"""
Pick the datasheets that make up the corpus and pull them out of the archives.

Each Zenodo archive is split into train/dev/test folders, with a PDF and an
HTML version of every datasheet. I only sample from dev and test: those are
the documents the original authors labelled by hand, so if I ever want to
cross-check an answer against their gold labels, the data is there.

Sampling is seeded, so running this again gives the same corpus. The result
is written to corpus/manifest.csv (committed) and the files themselves are
extracted to RAW_DIR (not committed - see the licensing note in the README).

    python scripts/build_manifest.py              # ~30 per class
    python scripts/build_manifest.py --per-class 10
"""

import argparse
import csv
import random
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.settings import ARCHIVE_DIR, MANIFEST_PATH, RAW_DIR

SOURCE = "https://doi.org/10.5281/zenodo.4558344"
LICENCE = "CC BY-NC-SA 4.0"

CLASSES = {
    "transistor": "transistor_dataset.tar.xz",
    "opamp": "opamp_dataset.tar.xz",
    "connector": "circular_connector_dataset.tar.xz",
}

# Some datasheets are 100+ page family catalogues. They're slow to parse and
# I'd never read them closely enough to write questions about them, so skip
# anything over this size.
MAX_PDF_BYTES = 3_000_000
SEED = 4558344  # the Zenodo record number, so it's easy to remember

# Picked by the sampler but no use for a question set. Each one is replaced by
# a document drawn with a separate random generator (REPLACEMENT_SEED), so
# excluding something never changes the other picks - and never forces a
# re-parse of the documents that were already fine.
EXCLUDE = {
    "amphenolsinesystemscorp_rt0w7106snh-k":
        "mechanical drawing; only text is the title block",
    "amphenolsinesystemscorp_rt0w71626pnh-k":
        "mechanical drawing; only text is the title block",
    "teconnectivityampconnectors_213597-4":
        "scanned drawing with no text layer; OCR output is garbled",
}
REPLACEMENT_SEED = SEED + 1
# A replacement has to have a real text layer, so the same problem can't come
# straight back. Datasheets in the corpus have a few thousand characters even
# on one page; the drawings above have under 900.
MIN_REPLACEMENT_TEXT_CHARS = 2000


def list_candidates(archive: Path) -> dict[str, dict]:
    """
    One streaming pass over the archive, collecting dev/test datasheets that
    have both a PDF and an HTML version. Returns {stem: info}.
    """
    found: dict[str, dict] = {}
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            # expected shape: <split>/<pdf|html>/<file>
            if len(parts) != 3 or parts[0] not in ("dev", "test"):
                continue
            split, kind, filename = parts
            stem, ext = filename.rsplit(".", 1)
            entry = found.setdefault(stem, {"split": split})
            if ext.lower() == "pdf" and kind == "pdf":
                entry["pdf"] = member.name
                entry["pdf_bytes"] = member.size
            elif ext.lower() == "html" and kind == "html":
                entry["html"] = member.name
    return {
        stem: e for stem, e in found.items()
        if "pdf" in e and "html" in e and e["pdf_bytes"] <= MAX_PDF_BYTES
    }


def extract(archive: Path, wanted: dict[str, str]) -> None:
    """Second pass: write out just the members we picked. wanted = {member name: target path}."""
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            if member.name in wanted:
                target = Path(wanted[member.name])
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def text_chars(pdf: Path) -> int:
    """How much text the PDF's own text layer holds (no OCR)."""
    import pypdfium2 as pdfium  # comes with Docling

    doc = pdfium.PdfDocument(pdf)
    try:
        return sum(len(page.get_textpage().get_text_range()) for page in doc)
    finally:
        doc.close()


def pick_replacements(archive: Path, cls: str, pool: list[str], candidates: dict,
                      needed: int) -> list[str]:
    """
    Take documents from `pool` in order, skipping any without a usable text
    layer. Checking means extracting the PDF, so a handful of spares are
    pulled out to a scratch folder and checked there.
    """
    chosen: list[str] = []
    scratch = RAW_DIR / "_replacement_check"
    for start in range(0, len(pool), 10):
        batch = pool[start:start + 10]
        extract(archive, {candidates[s]["pdf"]: scratch / f"{s}.pdf" for s in batch})
        for stem in batch:
            chars = text_chars(scratch / f"{stem}.pdf")
            if chars >= MIN_REPLACEMENT_TEXT_CHARS:
                chosen.append(stem)
            else:
                print(f"{cls}: replacement {stem} skipped, only {chars} chars of text")
            if len(chosen) == needed:
                shutil.rmtree(scratch)
                return chosen
    shutil.rmtree(scratch, ignore_errors=True)
    raise SystemExit(f"{cls}: ran out of candidates with a usable text layer")


def make_doc_id(cls: str, stem: str) -> str:
    # Filenames have brackets, spaces etc. Keep IDs boring and filesystem-safe.
    return f"{cls}-{re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows = []

    for cls, archive_name in CLASSES.items():
        archive = ARCHIVE_DIR / archive_name
        if not archive.with_suffix(archive.suffix + ".ok").exists():
            raise SystemExit(f"{archive_name} isn't downloaded yet - run download_corpus.py")

        print(f"{cls}: scanning {archive_name}...", flush=True)
        candidates = list_candidates(archive)
        # Sort before sampling so the result doesn't depend on archive order.
        picked = rng.sample(sorted(candidates), min(args.per_class, len(candidates)))
        print(f"{cls}: {len(candidates)} candidates, picked {len(picked)}")

        dropped = [s for s in picked if s in EXCLUDE]
        if dropped:
            picked = [s for s in picked if s not in EXCLUDE]
            for stem in dropped:
                print(f"{cls}: excluded {stem} ({EXCLUDE[stem]})")
            pool = [s for s in sorted(candidates) if s not in picked and s not in EXCLUDE]
            random.Random(REPLACEMENT_SEED).shuffle(pool)
            replacements = pick_replacements(archive, cls, pool, candidates, len(dropped))
            print(f"{cls}: replaced with {', '.join(replacements)}")
            picked += replacements

        wanted = {}
        for stem in sorted(picked):
            info = candidates[stem]
            doc_id = make_doc_id(cls, stem)
            pdf_out = RAW_DIR / cls / f"{doc_id}.pdf"
            html_out = RAW_DIR / cls / f"{doc_id}.html"
            wanted[info["pdf"]] = pdf_out
            wanted[info["html"]] = html_out
            rows.append({
                "doc_id": doc_id,
                "component_class": cls,
                "tier": 1,
                "archive": archive_name,
                "split": info["split"],
                "pdf_member": info["pdf"],
                "html_member": info["html"],
                "pdf_kb": round(info["pdf_bytes"] / 1024),
                "licence": LICENCE,
                "source": SOURCE,
            })

        print(f"{cls}: extracting...", flush=True)
        extract(archive, wanted)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
