import pytest

from scripts.evaluate_contriever import evaluate_query, ndcg_at_k


def test_csfcube_ndcg_at_10_uses_ideal_annotation_relevances():
    annotation = {
        "cands": ["best", "second", "third"],
        "relevance_max": [3, 2, 1],
    }
    assert evaluate_query(["best", "second", "third"], annotation)["ndcg_at_10"] == pytest.approx(1.0)


def test_csfcube_ndcg_at_10_penalizes_reversed_ranking():
    annotation = {
        "cands": ["best", "second", "third"],
        "relevance_max": [3, 2, 1],
    }
    score = evaluate_query(["third", "second", "best"], annotation)["ndcg_at_10"]
    assert 0.0 < score < 1.0


def test_csfcube_annotation_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="lengths differ"):
        evaluate_query(["d1"], {"cands": ["d1", "d2"], "relevance_max": [1]})


def test_ndcg_at_10_is_bounded_and_zero_when_no_relevant_labels():
    assert ndcg_at_k(["d1", "d2"], {"d3": 0}, 10) == pytest.approx(0.0)
