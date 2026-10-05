"""Build the Milestone 4 3x5x5 result grid.

The script intentionally does not invent missing scores. Every cell must
come from a supplied result. Missing values are reported as errors.
"""

from __future__ import annotations

import csv
import os
from typing import Mapping


DATASETS = [
    "CSFCube",
    "LIMIT-50k",
    "SciDocs",
]

METHODS = [
    "MedCPT",
    "ColBERT",
    "SPLADE",
    "BM25",
    "Contriever",
]

BUDGETS = [
    "1.0",
    "0.5",
    "0.25",
    "0.125",
    "0.0625",
]

OUTPUT = os.path.join("results", "full_grid_results.csv")


def validate_results(
    results: Mapping[tuple[str, str, str], float],
) -> None:
    """Verify that every required grid cell exists."""

    missing = []

    for dataset in DATASETS:
        for method in METHODS:
            for budget in BUDGETS:
                key = (dataset, method, budget)

                if key not in results:
                    missing.append(key)

    if missing:
        preview = "\n".join(
            f"  {dataset} / {method} / {budget}"
            for dataset, method, budget in missing[:20]
        )

        extra = ""
        if len(missing) > 20:
            extra = f"\n  ... and {len(missing) - 20} more"

        raise ValueError(
            "Missing Milestone 4 result cells:\n"
            + preview
            + extra
        )


def write_grid(
    results: Mapping[tuple[str, str, str], float],
    output_path: str = OUTPUT,
) -> None:
    """Write the validated 3x5x5 grid to CSV."""

    validate_results(results)

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "dataset",
        "method",
        *BUDGETS,
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for dataset in DATASETS:
            for method in METHODS:

                row = {
                    "dataset": dataset,
                    "method": method,
                }

                for budget in BUDGETS:
                    row[budget] = results[
                        (dataset, method, budget)
                    ]

                writer.writerow(row)

    print("Wrote:", output_path)
    print("Rows:", len(DATASETS) * len(METHODS))
    print("Score cells:", len(DATASETS) * len(METHODS) * len(BUDGETS))


if __name__ == "__main__":
    # Real experiment results will be supplied here later.
    #
    # Do NOT put placeholder numbers in the final submission.
    #
    # Example:
    #
    # results = {
    #     ("CSFCube", "Contriever", "1.0"): 0.123,
    #     ...
    # }
    #
    # write_grid(results)

    print(
        "Grid builder ready. "
        "No results were supplied, so no CSV was generated."
    )