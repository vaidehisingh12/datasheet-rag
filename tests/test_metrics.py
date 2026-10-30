import pytest

from rag.eval.metrics import (
    bootstrap_ci,
    first_relevant_rank,
    hit_at_k,
    mcnemar_p,
    ndcg_at_k,
    paired_bootstrap_delta,
    reciprocal_rank,
)

GOLD = {"doc_id": "d1", "pages": [3]}


def hit(doc, *pages):
    return {"doc_id": doc, "pages": list(pages)}


def test_rank_matches_on_doc_and_page():
    hits = [hit("d2", 3), hit("d1", 1), hit("d1", 2, 3)]
    assert first_relevant_rank(hits, GOLD) == 3
    assert first_relevant_rank(hits[:2], GOLD) is None


def test_hit_and_rr():
    assert hit_at_k(3, 5) == 1.0
    assert hit_at_k(6, 5) == 0.0
    assert hit_at_k(None, 5) == 0.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_ndcg():
    assert ndcg_at_k([hit("d1", 3), hit("x", 1)], GOLD, 5) == 1.0
    assert ndcg_at_k([hit("x", 1)], GOLD, 5) == 0.0
    # relevant at rank 2 only -> 1/log2(3) out of an ideal of 1
    assert ndcg_at_k([hit("x", 1), hit("d1", 3)], GOLD, 5) == pytest.approx(0.6309, abs=1e-3)


def test_bootstrap_ci_contains_mean():
    values = [1.0] * 30 + [0.0] * 30
    mean, low, high = bootstrap_ci(values, n_resamples=2000)
    assert mean == 0.5
    assert low < 0.5 < high
    assert high - low < 0.35


def test_paired_delta_of_identical_setups_is_zero():
    a = [1.0, 0.0, 1.0, 1.0]
    assert paired_bootstrap_delta(a, a, n_resamples=500) == (0.0, 0.0, 0.0)


def test_mcnemar():
    same = [1, 0, 1, 0]
    assert mcnemar_p(same, same) == 1.0
    # B fixes 10 questions A got wrong and breaks none: clearly different
    a = [0] * 10 + [1] * 20
    b = [1] * 30
    assert mcnemar_p(a, b) < 0.01
    # 3 fixed, 2 broken: could easily be chance
    a = [0, 0, 0, 1, 1]
    b = [1, 1, 1, 0, 0]
    assert mcnemar_p(a, b) == 1.0
