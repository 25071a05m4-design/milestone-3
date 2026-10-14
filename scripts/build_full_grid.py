"""Collect supplied NDCG@10 result JSONs and build the Milestone 4 grid.

No score is inferred or filled in. Unsupported/missing results remain missing,
and the CLI refuses to write a misleading complete grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

DATASETS = ["CSFCube", "LIMIT-50k", "SciDocs"]
METHODS = ["MedCPT", "ColBERT", "SPLADE", "BM25", "Contriever"]
BUDGETS = ["1.0", "0.5", "0.25", "0.125", "0.0625"]
OUTPUT = os.path.join("results", "full_grid_results.csv")
DIMENSION_TO_BUDGET = {"768": "1.0", "384": "0.5", "192": "0.25", "96": "0.125", "48": "0.0625"}


def _dataset_name(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "csfcube": "CSFCube",
        "limit": "LIMIT-50k",
        "limit-50k": "LIMIT-50k",
        "limit 50k": "LIMIT-50k",
        "scidocs": "SciDocs",
    }
    return aliases.get(normalized)


def _method_name(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return {m.lower(): m for m in METHODS}.get(normalized)


def _score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _record_score(dataset: str, method: str, budget: str, value: Any,
                  results: dict[tuple[str, str, str], float], source: str) -> None:
    score = _score(value)
    if score is None:
        raise ValueError(f"{source}: invalid NDCG@10 score {value!r}")
    key = (dataset, method, budget)
    if key in results and not math.isclose(results[key], score, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Conflicting scores for {key}: {results[key]} vs {score} ({source})")
    results[key] = score


def ingest_result_object(obj: Mapping[str, Any], results: dict[tuple[str, str, str], float],
                         source: str = "<memory>") -> int:
    """Ingest supported result JSON schemas; return the number of score cells added."""
    metric = str(obj.get("metric", "NDCG@10")).upper().replace(" ", "")
    if metric not in {"NDCG@10", "NDCG@K"}:
        raise ValueError(f"{source}: expected NDCG@10 results, got metric={metric!r}")
    dataset = _dataset_name(obj.get("dataset"))
    method = _method_name(obj.get("method"))
    if not dataset or not method:
        return 0

    before = len(results)
    # Schema A: {"score": ..., "budget": "1.0"} (often full-budget BM25).
    if "score" in obj:
        budget = str(obj.get("budget", "1.0"))
        if budget in BUDGETS:
            _record_score(dataset, method, budget, obj["score"], results, source)

    # Schema B: {"scores": {"1.0": score, ...}}.
    scores = obj.get("scores")
    if isinstance(scores, Mapping):
        for budget, value in scores.items():
            budget = str(budget)
            if budget in BUDGETS:
                _record_score(dataset, method, budget, value, results, source)

    # Schema C: MedCPT {"budgets": {"1.0": {"ndcg@10": score}}}.
    budget_objects = obj.get("budgets")
    if isinstance(budget_objects, Mapping):
        for budget, details in budget_objects.items():
            if str(budget) not in BUDGETS or not isinstance(details, Mapping):
                continue
            value = details.get("ndcg@10", details.get("NDCG@10", details.get("score")))
            if value is not None:
                _record_score(dataset, method, str(budget), value, results, source)

    # Schema D: dimension-indexed dense PCA results, such as CSFCube Contriever.
    dimensional_results = obj.get("results")
    if isinstance(dimensional_results, Mapping):
        for dimension, details in dimensional_results.items():
            budget = DIMENSION_TO_BUDGET.get(str(dimension))
            if not budget or not isinstance(details, Mapping):
                continue
            value = details.get("mean_ndcg_at_10", details.get("ndcg@10", details.get("score")))
            if value is not None:
                _record_score(dataset, method, budget, value, results, source)

    return len(results) - before


def load_result_directory(results_dir: str | os.PathLike[str]) -> dict[tuple[str, str, str], float]:
    """Load all JSON files in a results directory in deterministic order."""
    directory = Path(results_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Results directory does not exist: {directory}")
    results: dict[tuple[str, str, str], float] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read result file {path}: {exc}") from exc
        if not isinstance(obj, Mapping):
            continue
        ingest_result_object(obj, results, str(path))
    return results


def missing_cells(results: Mapping[tuple[str, str, str], float]) -> list[tuple[str, str, str]]:
    return [
        (dataset, method, budget)
        for dataset in DATASETS
        for method in METHODS
        for budget in BUDGETS
        if (dataset, method, budget) not in results
    ]


def validate_results(results: Mapping[tuple[str, str, str], float]) -> None:
    """Verify that every required grid cell exists."""
    missing = missing_cells(results)
    if missing:
        preview = "\\n".join(f"  {d} / {m} / {b}" for d, m, b in missing[:20])
        extra = f"\\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise ValueError("Missing Milestone 4 result cells:\\n" + preview + extra)


def write_grid(results: Mapping[tuple[str, str, str], float], output_path: str = OUTPUT) -> None:
    """Write the validated 3x5x5 grid to CSV."""
    validate_results(results)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = ["dataset", "method", *BUDGETS]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                row = {"dataset": dataset, "method": method}
                for budget in BUDGETS:
                    row[budget] = results[(dataset, method, budget)]
                writer.writerow(row)
    print("Wrote:", output_path)
    print("Rows:", len(DATASETS) * len(METHODS))
    print("Score cells:", len(DATASETS) * len(METHODS) * len(BUDGETS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Directory containing result JSON files")
    parser.add_argument("--output", default=OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    results = load_result_directory(args.results_dir)
    print(f"Loaded {len(results)}/75 unique score cells from {args.results_dir}")
    missing = missing_cells(results)
    if missing:
        print(f"Cannot write complete grid: {len(missing)} score cells are missing.")
        for dataset, method, budget in missing[:30]:
            print(f"  MISSING {dataset} / {method} / {budget}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        return 2
    write_grid(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
