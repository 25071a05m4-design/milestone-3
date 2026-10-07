import csv

import pytest

from scripts.build_full_grid import (
    BUDGETS,
    DATASETS,
    METHODS,
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

