"""
Keyword retrieval with BM25, mainly to catch part numbers.

Dense embeddings think every transistor's absolute maximum ratings table looks
about the same, which is true, and it's why "VCEO of the BC337" kept coming
back with some other transistor's table on top. BM25 cares about exact tokens,
so "BC337" vs "BC338" is a real difference to it.

Two details make that work:

  * Tokens are split at letter/digit boundaries as well as punctuation, and the
    whole token is kept too. Filenames like ZTX415STZ then still match a
    question that just says ZTX415, and BC337 vs BC338 differ on "337".
  * The document ID goes into every chunk's searchable text. A spec table
    often never mentions the part number - it's only in the page title - so
    without this a table can't match the part it belongs to.
"""

import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag.settings import DATA_DIR

_WORD = re.compile(r"[a-z0-9]+")
_PIECES = re.compile(r"[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text.lower()):
        tokens.append(word)
        pieces = _PIECES.findall(word)
        if len(pieces) > 1:
            tokens.extend(pieces)
    return tokens


# Switchable so the ablation can check how much of BM25's lead over dense
# comes from this metadata rather than from keyword matching itself.
INCLUDE_DOC_ID = True


def searchable_text(chunk: dict) -> str:
    # doc_id looks like "transistor-DiodesIncorporated_ZTX415STZ"
    parts = [chunk["section"], chunk["caption"], chunk["text"]]
    if INCLUDE_DOC_ID:
        parts.insert(0, chunk["doc_id"])
    return " ".join(parts)


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[dict]]:
    with open(DATA_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return BM25Okapi([tokenize(searchable_text(c)) for c in chunks]), chunks


def search(query: str, k: int = 5, component_class: str | None = None) -> list[dict]:
    bm25, chunks = _index()
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order:
        if component_class and chunks[i]["component_class"] != component_class:
            continue
        hits.append({**chunks[i], "score": float(scores[i])})
        if len(hits) == k:
            break
    return hits
