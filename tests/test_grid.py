import csv
import json

import pytest

from scripts.build_full_grid import (
    BUDGETS,
    DATASETS,
    METHODS,
    ingest_result_object,
    load_result_directory,
    missing_cells,
    validate_results,
    write_grid,
)


def complete_results():
    return {
        (dataset, method, budget): 0.5
        for dataset in DATASETS
        for method in METHODS
        for budget in BUDGETS
    }


def test_complete_grid_has_75_cells():
    results = complete_results()
    assert len(results) == 75
    validate_results(results)


def test_incomplete_grid_is_rejected():
    results = complete_results()
    results.pop(("CSFCube", "Contriever", "0.5"))
    with pytest.raises(ValueError, match="Missing"):
        validate_results(results)


def test_grid_writes_15_rows_and_5_budget_columns(tmp_path):
    output = tmp_path / "full_grid_results.csv"
    write_grid(complete_results(), str(output))
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert list(rows[0]) == ["dataset", "method", *BUDGETS]


def test_ingests_full_budget_bm25():
    results = {}
    count = ingest_result_object(
        {
            "dataset": "LIMIT-50k",
            "method": "BM25",
            "budget": "1.0",
            "metric": "NDCG@10",
            "score": 0.73,
        },
        results,
    )
    assert count == 1
    assert results[("LIMIT-50k", "BM25", "1.0")] == pytest.approx(0.73)


def test_ingests_scores_schema():
    results = {}
    ingest_result_object(
        {
            "dataset": "SciDocs",
            "method": "Contriever",
            "metric": "NDCG@10",
            "scores": {"1.0": 0.4, "0.5": 0.3},
        },
        results,
    )
    assert results[("SciDocs", "Contriever", "1.0")] == pytest.approx(0.4)
    assert results[("SciDocs", "Contriever", "0.5")] == pytest.approx(0.3)


def test_ingests_medcpt_budgets_schema():
    results = {}
    ingest_result_object(
        {
            "dataset": "LIMIT-50k",
            "method": "MedCPT",
            "metric": "NDCG@10",
            "budgets": {
                "1.0": {"dimension": 768, "ndcg@10": 0.5},
                "0.5": {"dimension": 384, "ndcg@10": 0.45},
            },
        },
        results,
    )
    assert results[("LIMIT-50k", "MedCPT", "1.0")] == pytest.approx(0.5)
    assert results[("LIMIT-50k", "MedCPT", "0.5")] == pytest.approx(0.45)


def test_ingests_dimension_keyed_csfcube_schema():
    results = {}
    ingest_result_object(
        {
            "dataset": "CSFCube",
            "method": "Contriever",
            "metric": "NDCG@10",
            "results": {
                "768": {"mean_ndcg_at_10": 0.25},
                "384": {"mean_ndcg_at_10": 0.2},
            },
        },
        results,
    )
    assert results[("CSFCube", "Contriever", "1.0")] == pytest.approx(0.25)
    assert results[("CSFCube", "Contriever", "0.5")] == pytest.approx(0.2)


def test_conflicting_duplicate_score_is_rejected():
    results = {}
    obj = {
        "dataset": "SciDocs",
        "method": "BM25",
        "budget": "1.0",
        "metric": "NDCG@10",
        "score": 0.2,
    }
    ingest_result_object(obj, results, "first.json")
    obj["score"] = 0.3
    with pytest.raises(ValueError, match="Conflicting"):
        ingest_result_object(obj, results, "second.json")


def test_invalid_score_is_rejected():
    results = {}
    with pytest.raises(ValueError, match="invalid"):
        ingest_result_object(
            {
                "dataset": "SciDocs",
                "method": "BM25",
                "budget": "1.0",
                "metric": "NDCG@10",
                "score": 1.5,
            },
            results,
            "bad.json",
        )


def test_load_result_directory_reads_json(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text(
        json.dumps({
            "dataset": "SciDocs",
            "method": "BM25",
            "budget": "1.0",
            "metric": "NDCG@10",
            "score": 0.15,
        }),
        encoding="utf-8",
    )
    results = load_result_directory(tmp_path)
    assert results[("SciDocs", "BM25", "1.0")] == pytest.approx(0.15)
    assert len(missing_cells(results)) == 74
