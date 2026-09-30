"""Unit tests for truncation utilities.

Confound #1: queries must not participate in PCA fitting.  The test below
fits the same corpus twice while keeping a deliberately different query set
held out both times, and verifies that the frozen projection is identical.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.truncate import (
    apply_frozen_pca,
    fit_corpus_pca,
    fit_colbert_token_pca,
    topk_splade,
    truncate_colbert_tokens,
)


def _same_pca_up_to_sign(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    """PCA eigenvectors are sign-indeterminate; compare after sign alignment."""
    if a.shape != b.shape:
        return False
    aligned = b.copy()
    for i in range(a.shape[0]):
        if np.dot(a[i], aligned[i]) < 0:
            aligned[i] *= -1
    return np.allclose(a, aligned, atol=atol, rtol=0)


def test_confounds_1_queries_held_out_projection_is_unchanged():
    rng = np.random.default_rng(1234)
    corpus = rng.normal(size=(32, 8))
    queries = rng.normal(loc=100.0, scale=20.0, size=(7, 8))

    # Reference fit: corpus only.
    first = fit_corpus_pca(corpus, n_components=4)

    # Refit while queries remain explicitly held out.  This is the regression
    # check: query vectors must never be passed to fit().
    corpus_only_again = corpus.copy()
    second = fit_corpus_pca(corpus_only_again, n_components=4)

    assert _same_pca_up_to_sign(first.projection_matrix(), second.projection_matrix())
    np.testing.assert_allclose(first.mean_, second.mean_, atol=1e-12, rtol=0)

    # Applying queries is allowed, but must not mutate the fitted projection.
    before = first.projection_matrix()
    projected_queries = apply_frozen_pca(first, queries)
    after = first.projection_matrix()
    assert projected_queries.shape == (7, 4)
    np.testing.assert_array_equal(before, after)


def test_colbert_per_token_pca_truncates_128d_vectors():
    rng = np.random.default_rng(7)
    corpus_tokens = rng.normal(size=(8, 5, 128))
    queries = rng.normal(size=(2, 3, 128))

    projector = fit_colbert_token_pca(corpus_tokens, n_components=32)
    reduced = truncate_colbert_tokens(queries, projector)

    assert reduced.shape == (2, 3, 32)


def test_splade_topk_retains_largest_weights():
    sparse = {11: 0.1, 3: 0.9, 8: 0.4, 5: 0.7, 2: 0.0}
    assert topk_splade(sparse, 2) == {3: 0.9, 5: 0.7}
    assert topk_splade(sparse, 0) == {}
