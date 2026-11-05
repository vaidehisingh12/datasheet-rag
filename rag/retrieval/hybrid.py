"""
Combining dense and BM25 results, and optionally reranking them.

Fusion is reciprocal rank fusion (RRF): each chunk scores 1 / (60 + rank) in
every list it appears in, and the scores are added up. It only looks at ranks,
which matters here because BM25 scores (0 to 40-ish) and cosine similarities
(0.5 to 0.9) aren't on any shared scale. Blending the raw scores would mean
inventing a weighting.

Reranking takes the fused top 30 and re-scores each (question, chunk) pair
with a cross-encoder, which reads both together instead of comparing two
separately computed vectors. It's much better at "does this chunk actually
answer the question", and much slower, so it only ever sees a shortlist.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.retrieval import bm25, dense

RRF_K = 60          # the standard constant from the original RRF paper
CANDIDATES = 50     # how deep to read each list before fusing
RERANK_DEPTH = 30   # how many fused results the cross-encoder re-scores

# Small and CPU-friendly: about 22M parameters.
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def fuse(*ranked_lists: list[dict]) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, 1):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(cid, hit)
    order = sorted(scores, key=scores.get, reverse=True)
    return [{**by_id[cid], "score": scores[cid]} for cid in order]


def search(query: str, k: int = 5, component_class: str | None = None) -> list[dict]:
    return fuse(
        dense.search(query, CANDIDATES, component_class),
        bm25.search(query, CANDIDATES, component_class),
    )[:k]


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER, device="cpu", max_length=512)


def rerank(query: str, hits: list[dict]) -> list[dict]:
    # Give the cross-encoder the same context a person would see: which
    # datasheet, which section, which table.
    pairs = [(query, f"{h['doc_id']} | {h['section']} | {h['caption']}\n{h['text']}") for h in hits]
    scores = _reranker().predict(pairs, batch_size=16, show_progress_bar=False)
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [{**hits[i], "score": float(scores[i])} for i in order]


def search_reranked(query: str, k: int = 5, component_class: str | None = None) -> list[dict]:
    return rerank(query, search(query, RERANK_DEPTH, component_class))[:k]
