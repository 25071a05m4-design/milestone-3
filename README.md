# Milestone 3 — Representation Truncation

## Objective

Implement dimensionality/truncation methods for dense, late-interaction, and sparse information-retrieval representations while avoiding query leakage.

## Implemented Methods

### 1. MedCPT / Contriever — Dense PCA

PCA is fitted using **corpus embeddings only**.

The resulting PCA projection is frozen and subsequently applied to query embeddings.

Queries are never used when fitting the PCA.

### 2. ColBERT — Token-Level PCA

ColBERT token representations are 128-dimensional.

The implementation treats corpus token vectors as individual observations, fits PCA on them, and projects each token vector into the retained dimensions.

### 3. SPLADE — Top-k Term Pruning

SPLADE produces sparse term-weight vectors.

For each vector, the implementation retains the `k` terms with the largest weights and removes the remaining terms.

Ties are resolved deterministically using the term ID.

## Leakage / Confound Test

A unit test verifies that query representations are not used to fit the corpus PCA.

The test checks that the projection used for queries remains determined by the corpus-fitting stage rather than being refitted using query data.

## Project Structure

```text
truncation_impl/
├── scripts/
│   └── truncate.py
├── tests/
│   └── test_truncate.py
└── README.md
```

## Testing

Run:

```bash
python -m pytest -v
```

All tests should pass before the implementation is used with real embedding data.
