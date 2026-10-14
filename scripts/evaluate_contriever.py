"""Evaluate CSFCube Contriever/PCA rankings with correctly truncated NDCG@10.

The CSFCube annotations define three facets (background, method, result).
Each facet is evaluated independently on its annotated query set, then the
reported aggregate is the unweighted macro-average of the three facet means.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path("CSFCube")
RANKING_DIR = ROOT / "contriever_rankings"
OUTPUT_PATH = Path("results") / "csfcube_contriever.json"

FACETS = {
    "background": "test-pid2anns-csfcube-background.json",
    "method": "test-pid2anns-csfcube-method.json",
    "result": "test-pid2anns-csfcube-result.json",
}
DIMS = (768, 384, 192, 96, 48)
DIMENSION_TO_BUDGET = {
    768: "1.0",
    384: "0.5",
    192: "0.25",
    96: "0.125",
    48: "0.0625",
}


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    return sum(
        (2.0 ** float(rel) - 1.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    actual = [float(relevance.get(str(doc_id), 0.0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    return 0.0 if ideal_dcg <= 0.0 else dcg_at_k(actual, k) / ideal_dcg


def evaluate_query(ranking: Sequence[str], annotation: Mapping[str, object]) -> dict[str, float]:
    candidate_ids = [str(value) for value in annotation["cands"]]
    relevance_values = annotation["relevance_max"]
    if len(candidate_ids) != len(relevance_values):
        raise ValueError("CSFCube annotation cands/relevance_max lengths differ")
    relevance = {
        doc_id: float(value)
        for doc_id, value in zip(candidate_ids, relevance_values)
    }
    ranked_ids = [str(value) for value in ranking]
    ranked_rels = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids]
    binary = [1 if rel >= 2 else 0 for rel in ranked_rels]
    top20_binary = binary[:20]
    total_relevant = sum(binary)
    r = total_relevant
    r_precision = (sum(binary[:r]) / r) if r else 0.0
    return {
        "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
        "ndcg_at_20": ndcg_at_k(ranked_ids, relevance, 20),
        "p_at_20": sum(top20_binary) / 20.0,
        "recall_at_20": (sum(top20_binary) / total_relevant) if total_relevant else 0.0,
        "r_precision": r_precision,
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    annotations_by_facet = {
        facet: load_json(ROOT / filename)
        for facet, filename in FACETS.items()
    }
    output = {
        "dataset": "CSFCube",
        "method": "Contriever",
        "metric": "NDCG@10",
        "evaluation_protocol": "CSFCube three-facet candidate annotations",
        "aggregate": "unweighted macro-average of facet NDCG@10 means",
        "dimensions": list(DIMS),
        "results": {},
    }

    for dimension in DIMS:
        ranking_path = RANKING_DIR / f"contriever-{dimension}.json"
        rankings = load_json(ranking_path)
        if not isinstance(rankings, dict):
            raise ValueError(f"{ranking_path}: expected query-ID to ranked-document-ID mapping")
        rankings = {str(qid): [str(docid) for docid in docs] for qid, docs in rankings.items()}

        facet_means = {}
        facet_query_counts = {}
        facet_metrics = {}
        for facet, annotations in annotations_by_facet.items():
            expected = {str(qid) for qid in annotations}
            missing = sorted(expected - set(rankings))
            if missing:
                raise ValueError(
                    f"{ranking_path}: missing {len(missing)} {facet} queries; examples: {missing[:10]}"
                )
            per_query = [
                evaluate_query(rankings[str(qid)], annotation)
                for qid, annotation in annotations.items()
            ]
            if not per_query:
                raise ValueError(f"CSFCube {facet} facet has no annotated queries")
            facet_metrics[facet] = {
                key: sum(item[key] for item in per_query) / len(per_query)
                for key in per_query[0]
            }
            facet_means[facet] = facet_metrics[facet]["ndcg_at_10"]
            facet_query_counts[facet] = len(per_query)

        macro_ndcg10 = sum(facet_means.values()) / len(facet_means)
        output["results"][str(dimension)] = {
            **{facet: facet_means[facet] for facet in FACETS},
            "mean_ndcg_at_10": macro_ndcg10,
            "facet_metrics": facet_metrics,
            "queries": facet_query_counts,
        }
        print(
            f"CSFCube Contriever {dimension}d "
            f"(budget {DIMENSION_TO_BUDGET[dimension]}): "
            f"NDCG@10={macro_ndcg10:.6f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Saved validated result: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
