"""Evaluate saved Contriever rankings against BEIR test qrels.

This script intentionally evaluates ranking artifacts after retrieval. It does
not re-encode models or fit PCA. It writes the exact JSON schema consumed by
scripts/build_full_grid.py.

Example:
  python scripts/evaluate_contriever_beir.py --dataset LIMIT-50k
  python scripts/evaluate_contriever_beir.py --dataset SciDocs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from beir.datasets.data_loader import GenericDataLoader

DIMENSIONS = (768, 384, 192, 96, 48)
DIMENSION_TO_BUDGET = {
    768: "1.0",
    384: "0.5",
    192: "0.25",
    96: "0.125",
    48: "0.0625",
}
DEFAULT_DATA_DIRS = {
    "LIMIT-50k": r"E:\\ir_project\\data\\limit-50k",
    "SciDocs": r"E:\\ir_project\\data\\scidocs",
}
DEFAULT_RANKING_DIR = r"E:\\ir_project\\data\\contriever_beir"


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    return sum(
        (2.0 ** float(rel) - 1.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    actual = [float(qrels.get(str(doc_id), 0.0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(rel) for rel in qrels.values()), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(actual, k) / ideal_dcg


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, float | int]:
    """Evaluate every qrels query and reject missing/extra query IDs."""
    ranking_ids = {str(qid) for qid in rankings}
    qrel_ids = {str(qid) for qid in qrels}
    missing = sorted(qrel_ids - ranking_ids)
    if missing:
        raise ValueError(
            f"Rankings are missing {len(missing)} qrels queries; "
            f"examples: {missing[:10]}"
        )

    per_query = [
        ndcg_at_k(rankings[qid], qrels[qid], k=10)
        for qid in sorted(qrel_ids)
    ]
    if not per_query:
        raise ValueError("No qrels queries were available for evaluation.")
    return {
        "mean_ndcg_at_10": sum(per_query) / len(per_query),
        "num_queries": len(per_query),
    }


def evaluate_dataset(
    dataset: str,
    data_dir: str,
    ranking_dir: str,
    output_dir: str,
) -> Path:
    _, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    expected_qids = {str(qid) for qid in queries}
    qrel_qids = {str(qid) for qid in qrels}
    if expected_qids != qrel_qids:
        raise ValueError(
            f"{dataset}: query/qrels mismatch: "
            f"{len(expected_qids - qrel_qids)} queries without qrels, "
            f"{len(qrel_qids - expected_qids)} qrels IDs absent from queries"
        )

    slug = dataset.lower().replace("-", "_")
    result = {
        "dataset": dataset,
        "method": "Contriever",
        "metric": "NDCG@10",
        "evaluation_split": "test",
        "source": "saved_ranking_artifacts",
        "results": {},
    }
    ranking_root = Path(ranking_dir)
    for dimension in DIMENSIONS:
        path = ranking_root / f"{slug}_contriever_rankings_{dimension}d.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ranking artifact for {dataset}, {dimension}d: {path}"
            )
        with path.open(encoding="utf-8") as handle:
            rankings = json.load(handle)
        if not isinstance(rankings, dict):
            raise ValueError(f"{path}: expected a JSON object mapping query IDs to ranked document IDs")
        rankings = {str(qid): [str(doc_id) for doc_id in docs] for qid, docs in rankings.items()}
        # The ranking file must cover all test queries. Extra non-qrels queries
        # are ignored by the metric function only after coverage is checked.
        metrics = evaluate_rankings(rankings, qrels)
        result["results"][str(dimension)] = metrics
        print(
            f"{dataset} Contriever {dimension}d "
            f"(budget {DIMENSION_TO_BUDGET[dimension]}): "
            f"NDCG@10={metrics['mean_ndcg_at_10']:.6f} "
            f"queries={metrics['num_queries']}"
        )

    output_path = Path(output_dir) / f"{slug}_contriever.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\\n")
    print(f"Saved validated result: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DEFAULT_DATA_DIRS, required=True)
    parser.add_argument("--data-dir", help="Override local BEIR dataset directory")
    parser.add_argument("--ranking-dir", default=DEFAULT_RANKING_DIR)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    evaluate_dataset(
        args.dataset,
        args.data_dir or DEFAULT_DATA_DIRS[args.dataset],
        args.ranking_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
