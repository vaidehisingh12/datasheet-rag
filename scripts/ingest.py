"""
Parse, chunk and index every document in the manifest.

Parsing is the slow part (Docling's layout and table models on a laptop CPU),
but its output is cached, so re-running this after changing the chunker only
redoes the cheap steps. The index is rebuilt from scratch each time so it never
holds chunks from an older version of the chunker.

    python scripts/ingest.py
    python scripts/ingest.py --limit 5      # quick run on the first few docs
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import make_converter, parse_pdf
from rag.retrieval import dense
from rag.settings import DATA_DIR, MANIFEST_PATH, RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    tokenizer = dense.get_model().tokenizer

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    converter = make_converter()
    all_chunks = []
    failed = []

    for i, row in enumerate(rows, 1):
        pdf = RAW_DIR / row["component_class"] / f"{row['doc_id']}.pdf"
        start = time.perf_counter()
        try:
            doc = parse_pdf(pdf, row["doc_id"], converter)
        except Exception as e:
            # A few PDFs in any scraped set are broken. Note them and move on
            # rather than losing a long run to one bad file.
            failed.append({"doc_id": row["doc_id"], "error": repr(e)})
            print(f"[{i}/{len(rows)}] {row['doc_id']}: FAILED ({e})", flush=True)
            continue

        chunks = chunk_document(doc, row["doc_id"], row["component_class"], count=count)
        all_chunks.extend(chunks)
        tables = sum(c.kind == "table" for c in chunks)
        print(f"[{i}/{len(rows)}] {row['doc_id']}: {len(chunks)} chunks "
              f"({tables} tables) in {time.perf_counter() - start:.1f}s", flush=True)

    print(f"indexing {len(all_chunks)} chunks...", flush=True)
    dense.reset_index()
    dense.index_chunks(all_chunks)
    dense.close()

    # Keep a plain copy of the chunks around. Handy for eyeballing what the
    # chunker did, and for BM25 later, which needs the raw text.
    with open(DATA_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.payload(), ensure_ascii=False) + "\n")

    if failed:
        with open(DATA_DIR / "ingest_failures.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"{len(failed)} documents failed, see ingest_failures.json")
    print("done")


if __name__ == "__main__":
    main()
