"""
Paths and model names used across the project.

The repo itself lives in OneDrive, but the raw datasheets (~3.7 GB of
archives) and the vector index do not belong there: OneDrive would try to
sync gigabytes of PDFs. So anything large goes under DATA_DIR, which defaults
to a folder outside the synced tree and can be moved with an env variable.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get("DATASHEET_RAG_DATA", Path.home() / "datasheet-rag-local")
)

ARCHIVE_DIR = DATA_DIR / "archives"   # .tar.xz files straight from Zenodo
RAW_DIR = DATA_DIR / "raw"            # PDFs picked for the corpus, one per manifest row
PARSED_DIR = DATA_DIR / "parsed"      # Docling output, one JSON per document
INDEX_DIR = DATA_DIR / "index"        # local Qdrant storage

MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.csv"

# Small English embedding model that runs fine on a laptop CPU.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "datasheets"
