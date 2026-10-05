"""Evaluation utilities for Milestone 4.

The primary metric is NDCG@10.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """Compute discounted cumulative gain at k."""
    if k <= 0:
        return 0.0

    total = 0.0

    for rank, relevance in enumerate(relevances[:k], start=1):
        total += (2.0 ** float(relevance) - 1.0) / math.log2(rank + 1)

    return total


def ndcg_at_k(
    ranked_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
) -> float:
    """Compute NDCG@k for one query.

    Parameters
    ----------
    ranked_ids:
        Document IDs in predicted ranking order.
    qrels:
        Mapping from document ID to graded relevance.
    k:
        Evaluation cutoff.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    ranked_relevances = [
        float(qrels.get(str(doc_id), 0.0))
        for doc_id in ranked_ids[:k]
    ]

    actual_dcg = dcg_at_k(ranked_relevances, k)

    ideal_relevances = sorted(
        (float(value) for value in qrels.values()),
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def mean_ndcg_at_k(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> float:
    """Compute mean NDCG@k across queries with available qrels."""
    scores = []

    for query_id, ranked_ids in rankings.items():
        query_id = str(query_id)

        if query_id not in qrels:
            continue

        scores.append(
            ndcg_at_k(
                ranked_ids,
                qrels[query_id],
                k=k,
            )
        )

    if not scores:
        raise ValueError("No queries with both rankings and qrels were found")

    return sum(scores) / len(scores)