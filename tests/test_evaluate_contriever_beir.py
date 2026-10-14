import pytest

from scripts.evaluate_contriever_beir import (
    DIMENSION_TO_BUDGET,
    evaluate_rankings,
    ndcg_at_k,
)


def test_ndcg_at_10_perfect_ranking_is_one():
    qrels = {"d1": 3, "d2": 2, "d3": 1}
    assert ndcg_at_k(["d1", "d2", "d3"], qrels) == pytest.approx(1.0)


def test_ndcg_at_10_reversed_ranking_is_below_one():
    qrels = {"d1": 3, "d2": 2, "d3": 1}
    score = ndcg_at_k(["d3", "d2", "d1"], qrels)
    assert 0.0 < score < 1.0


def test_ndcg_zero_when_no_relevant_documents():
    assert ndcg_at_k(["d1", "d2"], {"d3": 0}) == pytest.approx(0.0)


def test_evaluator_requires_every_qrels_query():
    with pytest.raises(ValueError, match="missing"):
        evaluate_rankings(
            {"q1": ["d1"]},
            {"q1": {"d1": 1}, "q2": {"d2": 1}},
        )


def test_evaluator_returns_mean_and_query_count():
    metrics = evaluate_rankings(
        {"q1": ["d1", "d2"], "q2": ["d2", "d1"]},
        {"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    assert metrics["mean_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 2


def test_representation_budget_mapping():
    assert DIMENSION_TO_BUDGET == {
        768: "1.0",
        384: "0.5",
        192: "0.25",
        96: "0.125",
        48: "0.0625",
    }
