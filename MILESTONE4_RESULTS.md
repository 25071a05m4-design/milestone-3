# Milestone 4 — Result ingestion and grid validation

This stage collects already-computed NDCG@10 results into the required 3-dataset × 5-method × 5-budget grid. It does **not** run retrievers, train neural models, or manufacture scores.

## Build the grid

From the repository root:

```bash
python scripts/build_full_grid.py --results-dir results --output results/full_grid_results.csv
```

The loader supports the result JSON layouts currently present in this repository:
- `score` with optional `budget` (single-budget results such as BM25)
- `scores` keyed by budget
- `budgets` keyed by budget with an `ndcg@10` field (MedCPT layout)
- `results` keyed by dimensions 768, 384, 192, 96, and 48 (CSFCube Contriever layout)

The script checks score ranges, catches conflicting duplicate scores, reports missing cells, and exits without writing a purportedly complete grid if any cell is missing. Missing values must be filled only by running and validating the corresponding experiment.

## Tests

```bash
python -m pytest -v
```

Tests cover the 75-cell requirement, CSV layout, the supported result schemas, duplicate conflicts, invalid scores, and missing results.

## Methodology checks before final reporting

- Confirm each result uses the same NDCG@10 relevance convention and query set as the dataset's intended evaluator. CSFCube has its own protocol; do not silently mix it with generic BEIR evaluation.
- A PCA dimension is mapped to the fraction of the original 768-dimensional dense vector: 768→1.0, 384→0.5, 192→0.25, 96→0.125, 48→0.0625. Verify this budget mapping matches the final paper's representation-budget definition.
- ColBERT is token-level late interaction; do not mean-pool its token vectors to pretend it is a single-vector retriever.
- SPLADE is sparse vocabulary-space; its budget must be defined by the pruning protocol, not by dense-vector dimension.
- BM25 is a lexical baseline, not a neural representation with epochs or a learned vector dimension. Decide and document whether it is a single full-budget reference or a constant control across budget columns. The builder intentionally does not copy its full-budget score into smaller-budget cells.
- The current checked-in result files are partial. Do not report a completed 75-cell grid until every required cell is backed by a reproducible result artifact.
