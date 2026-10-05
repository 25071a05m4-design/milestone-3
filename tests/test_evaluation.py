import math

import pytest

from scripts.evaluation import (
    dcg_at_k,
    mean_ndcg_at_k,
    ndcg_at_k,
)


def test_dcg_at_k():
    score = dcg_at_k([3, 2, 0], k=3)

    expected = (
        (2**3 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
    )

    assert score == pytest.approx(expected)


def test_ndcg_perfect_ranking():
    qrels = {
        "doc1": 3,
        "doc2": 2,
        "doc3": 1,
    }

    ranking = ["doc1", "doc2", "doc3"]

    assert ndcg_at_k(ranking, qrels, k=3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_is_lower():
    qrels = {
        "doc1": 3,
        "doc2": 2,
        "doc3": 1,
    }

    ranking = ["doc3", "doc2", "doc1"]

    score = ndcg_at_k(ranking, qrels, k=3)

    assert 0.0 < score < 1.0


def test_mean_ndcg():
    rankings = {
        "q1": ["d1", "d2"],
        "q2": ["d2", "d1"],
    }

    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
    }

    assert mean_ndcg_at_k(rankings, qrels, k=2) == pytest.approx(1.0)
