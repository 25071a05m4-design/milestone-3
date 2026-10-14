# Milestone 4 — Result ingestion and grid validation

This milestone combines reproducible evaluation code and a fail-closed result-grid builder. It does **not** fabricate scores, treat missing values as zero, or claim experiments have run when their artifacts are absent.

## 1. Run the existing full-budget BM25 baselines

From the repository root, in the configured Python environment:

```bash
python scripts/run_bm25.py --dataset CSFCube
python scripts/run_bm25.py --dataset LIMIT-50k
python scripts/run_bm25.py --dataset SciDocs
```

The current script writes full-budget JSON results under `results/`. Its configured dataset paths are local Windows paths; update them for your machine if needed.

## 2. Run Contriever retrieval and evaluate the saved rankings

The retrieval script creates 768-, 384-, 192-, 96-, and 48-dimensional ranking artifacts for LIMIT-50k and SciDocs. It uses the local Contriever model and local dataset paths configured near the top of `scripts/run_contriever_beir.py`.

```bash
python scripts/run_contriever_beir.py
python scripts/evaluate_contriever_beir.py --dataset LIMIT-50k
python scripts/evaluate_contriever_beir.py --dataset SciDocs
python scripts/evaluate_contriever.py
```

The BEIR evaluator validates test query coverage and writes `results/limit_50k_contriever.json` and `results/scidocs_contriever.json` in a schema the grid builder can ingest. The CSFCube evaluator separately validates all three annotated facets, computes true NDCG@10 against the complete facet annotation labels, and writes `results/csfcube_contriever.json`. It uses an explicitly documented unweighted macro-average of the background, method, and result facet means. The evaluators do not rerun embedding generation.

**Audit note:** the current checked-in LIMIT-50k and SciDocs Contriever NDCG@10 scores are extremely low. Re-run the ranking/evaluation pipeline and inspect query IDs, corpus IDs, embedding checkpoint/configuration, and top-10 rankings before interpreting those scores.

## 3. Build the 75-cell grid

```bash
python scripts/build_full_grid.py --results-dir results --output results/full_grid_results.csv
```

For an explicit **constant BM25 control** across all budget columns (BM25 itself is not truncated), use:

```bash
python scripts/build_full_grid.py --results-dir results --output results/full_grid_results.csv --bm25-control-all-budgets
```

This repeats each dataset's observed BM25 full-budget score as a reference control. It does not claim BM25 has been truncated or that its smaller-budget scores were measured. This mode requires all three full-budget BM25 files and must be described as a constant lexical control in any report.

The loader supports:
- `score` with optional `budget` (single-budget result, e.g. BM25)
- `scores` keyed by budget
- `budgets` keyed by budget with an `ndcg@10` field (MedCPT)
- `results` keyed by dimensions 768, 384, 192, 96, and 48 with `mean_ndcg_at_10` (Contriever dimension runs)

It checks score ranges, catches conflicting duplicate scores, reports missing cells, and exits without writing a purportedly complete grid if any cell is missing.

## 4. Tests

```bash
python -m pip install numpy pytest
python -m pytest -v
```

The GitHub Actions workflow `.github/workflows/milestone4-validation.yml` runs the lightweight unit tests on pushes and pull requests.

## 5. Methodology decisions required before final reporting

- Re-run the CSFCube Contriever evaluator before using its existing checked-in scores: the prior evaluator reported full-ranking NDCG under an ambiguous `ndcg` field and did not calculate NDCG@10 correctly. The updated evaluator uses top-10 DCG and the full annotation relevance set for ideal DCG.
- Confirm that each result uses the intended NDCG@10 relevance convention and query set. CSFCube has its own candidate-pool/facet protocol; do not silently mix it with generic BEIR evaluation.
- PCA dimensions map to dense representation fractions as 768→1.0, 384→0.5, 192→0.25, 96→0.125, 48→0.0625. Verify that this matches the final paper's budget definition.
- ColBERT is token-level late interaction; do not mean-pool token vectors to disguise it as a single-vector retriever.
- SPLADE is sparse vocabulary-space; define its budget via a documented term-pruning protocol, not dense-vector dimension.
- BM25 is a lexical baseline, not a neural representation with epochs or learned vector dimensions. Decide and document whether BM25 is only a full-budget reference or a constant control at each budget. The grid builder intentionally does not copy its score into smaller-budget cells.
- The currently checked-in JSONs provide only a partial grid (approximately 28 of 75 cells). ColBERT/SPLADE and additional dataset/budget outputs still need to be generated, validated, and saved before a full grid can be reported.
- Model runs rely on local datasets/checkpoints under `E:\\ir_project`; those experiments cannot be truthfully marked complete until they have run in the configured environment and the resulting artifacts are reviewed.
