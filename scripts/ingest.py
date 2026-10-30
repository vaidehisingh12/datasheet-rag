"""
Parse, chunk and index every document in the manifest.

Parsing is the slow part (Docling's layout and table models on a laptop CPU),
but its output is cached, so re-running this after changing the chunker only
redoes the cheap steps. The index is rebuilt from scratch each time so it never
holds chunks from an older version of the chunker.

Parsing runs in a separate worker process. On the first full run, Docling
segfaulted on document 8 after parsing fine for a while, and the same PDF
parsed without trouble on its own. A segfault can't be caught with
try/except - it just kills the process - so the worker is disposable: if it
dies, we note which document it was on, start a fresh worker, and carry on.
A document that takes down the worker twice is recorded as failed and skipped.

    python scripts/ingest.py
    python scripts/ingest.py --limit 5      # quick run on the first few docs
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import PARSED_DIR, is_parsed
from rag.retrieval import dense
from rag.settings import DATA_DIR, MANIFEST_PATH, RAW_DIR
from docling_core.types.doc import DoclingDocument

MAX_CRASHES_PER_DOC = 2


def pdf_path(row: dict) -> Path:
    return RAW_DIR / row["component_class"] / f"{row['doc_id']}.pdf"


def parse_all(rows: list[dict]) -> dict[str, str]:
    """
    Make sure every document has a cached parse. Returns {doc_id: reason}
    for the ones that couldn't be parsed.
    """
    failed: dict[str, str] = {}
    crashes: dict[str, int] = {}

    while True:
        pending = [r for r in rows if not is_parsed(r["doc_id"]) and r["doc_id"] not in failed]
        if not pending:
            return failed

        args = []
        for r in pending:
            args += [str(pdf_path(r)), r["doc_id"]]

        current, started_at = None, time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "rag.ingest.parse", *args],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8",
        )
        for line in proc.stdout:
            status, doc_id, *rest = line.strip().split(" ", 2)
            if status == "START":
                current, started_at = doc_id, time.perf_counter()
            elif status == "OK":
                done = sum(is_parsed(r["doc_id"]) for r in rows)
                print(f"parsed {doc_id} in {time.perf_counter() - started_at:.0f}s "
                      f"({done}/{len(rows)})", flush=True)
                current = None
            elif status == "ERROR":
                # An ordinary Python exception: a broken PDF, not a crash.
                failed[doc_id] = rest[0] if rest else "unknown error"
                print(f"FAILED {doc_id}: {failed[doc_id]}", flush=True)
                current = None
        proc.wait()

        if proc.returncode != 0 and current:
            crashes[current] = crashes.get(current, 0) + 1
            print(f"worker died (exit {proc.returncode}) on {current}, "
                  f"crash {crashes[current]}/{MAX_CRASHES_PER_DOC}", flush=True)
            if crashes[current] >= MAX_CRASHES_PER_DOC:
                failed[current] = f"parser crashed {crashes[current]} times"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    failed = parse_all(rows)

    tokenizer = dense.get_model().tokenizer

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    all_chunks = []
    for row in rows:
        if row["doc_id"] in failed:
            continue
        doc = DoclingDocument.load_from_json(PARSED_DIR / f"{row['doc_id']}.json")
        all_chunks.extend(chunk_document(doc, row["doc_id"], row["component_class"], count=count))

    tables = sum(c.kind == "table" for c in all_chunks)
    print(f"indexing {len(all_chunks)} chunks ({tables} tables) "
          f"from {len(rows) - len(failed)} documents...", flush=True)
    dense.reset_index()
    dense.index_chunks(all_chunks)
    dense.close()

    # Keep a plain copy of the chunks around. Handy for eyeballing what the
    # chunker did, and for BM25 later, which needs the raw text.
    with open(DATA_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.payload(), ensure_ascii=False) + "\n")

    failures_path = DATA_DIR / "ingest_failures.json"
    if failed:
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"{len(failed)} documents failed, see {failures_path}")
    elif failures_path.exists():
        failures_path.unlink()  # left over from an earlier run
    print("done")


if __name__ == "__main__":
    main()
