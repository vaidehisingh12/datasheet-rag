"""
Retrieval metrics, plus the statistics needed to say whether a difference
between two setups is real.

Relevance is decided by the golden set: a retrieved chunk counts as a hit if
it's from the right document and covers one of the pages the answer is on.
Matching on (document, page) instead of chunk IDs means the golden set
survives changes to the chunker, which will happen a lot.

Why the statistics: with ~60 questions, one question flipping from miss to hit
moves hit rate by almost 2 points. A 3-point "improvement" can easily be
noise, so every comparison gets a bootstrap confidence interval, and paired
hit/miss outcomes get a McNemar test.
"""

import math
import random
from typing import Sequence


def is_relevant(hit: dict, gold: dict) -> bool:
    return hit["doc_id"] == gold["doc_id"] and bool(set(hit["pages"]) & set(gold["pages"]))


def first_relevant_rank(hits: Sequence[dict], gold: dict) -> int | None:
    """1-based rank of the first relevant hit, or None if nothing relevant came back."""
    for rank, hit in enumerate(hits, 1):
        if is_relevant(hit, gold):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(hits: Sequence[dict], gold: dict, k: int) -> float:
    """
    Binary-relevance nDCG. Several chunks can be relevant (a page often holds
    a table and some text), so unlike MRR this rewards getting more than one of
    them near the top. The ideal ranking assumes every relevant chunk we saw
    could have been ranked first.
    """
    rels = [1.0 if is_relevant(h, gold) else 0.0 for h in hits[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(int(sum(rels))))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` with a percentile bootstrap interval: (mean, low, high)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    low = means[int(alpha / 2 * n_resamples)]
    high = means[int((1 - alpha / 2) * n_resamples) - 1]
    return sum(values) / n, low, high


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], **kw) -> tuple[float, float, float]:
    """
    CI on mean(b) - mean(a) over the same questions. Pairing matters: both
    setups find the easy questions easy, so the per-question differences are
    much less noisy than comparing two independent intervals.
    """
    assert len(a) == len(b), "both setups must be scored on the same questions"
    return bootstrap_ci([y - x for x, y in zip(a, b)], **kw)


def mcnemar_p(a_hits: Sequence[float], b_hits: Sequence[float]) -> float:
    """
    Exact McNemar test on paired hit/miss outcomes. Only the questions where
    the two setups disagree carry information: if setup B fixes 6 questions
    and breaks 2, is that better than a coin flip? Returns the two-sided p-value.
    """
    b_only = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    a_only = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
