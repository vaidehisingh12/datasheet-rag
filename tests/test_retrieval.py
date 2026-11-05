"""
Tests for the parts of retrieval that don't need a model or an index: the
BM25 tokenizer and rank fusion.
"""

from rag.retrieval.bm25 import tokenize


def test_part_numbers_match_how_people_write_them():
    # The archive filename has a packaging suffix that nobody types in a question.
    assert {"ztx", "415"} <= set(tokenize("transistor-DiodesIncorporated_ZTX415STZ"))
    assert {"ztx", "415"} <= set(tokenize("ZTX415"))


def test_similar_part_numbers_stay_distinguishable():
    assert "337" in tokenize("BC337") and "337" not in tokenize("BC338")


def test_whole_token_is_kept_alongside_pieces():
    assert tokenize("BC337-25") == ["bc337", "bc", "337", "25"]


def test_dotted_lemo_part_numbers():
    assert tokenize("SKP.M13.GLLG") == ["skp", "m13", "m", "13", "gllg"]


def test_rank_fusion_rewards_agreement():
    # Importing hybrid pulls in sentence-transformers, so keep it local to this test.
    from rag.retrieval.hybrid import fuse

    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    bm25 = [{"chunk_id": "c"}, {"chunk_id": "b"}, {"chunk_id": "d"}]
    order = [h["chunk_id"] for h in fuse(dense, bm25)]
    # b is second in both lists, so it beats a and d, which are only in one each
    assert order[0] in {"b", "c"}
    assert order.index("b") < order.index("a")
    assert order.index("b") < order.index("d")
    assert set(order) == {"a", "b", "c", "d"}
