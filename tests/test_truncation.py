import numpy as np

from scripts.truncate import (
    PCAProjector,
    fit_corpus_pca,
    apply_frozen_pca,
    fit_colbert_token_pca,
    truncate_colbert_tokens,
    topk_splade,
)


def test_splade_keeps_top_k_terms():
    weights = {
        10: 0.2,
        20: 1.5,
        30: 0.7,
        40: 3.0,
        50: 2.0,
    }

    result = topk_splade(weights, k=2)

    assert len(result) == 2
    assert set(result.keys()) == {40, 50}


def test_colbert_token_pca_reduces_dimension():
    rng = np.random.default_rng(42)

    # 10 documents × 8 tokens × 128 dimensions
    corpus_tokens = rng.normal(size=(10, 8, 128))

    projector = fit_colbert_token_pca(
        corpus_tokens,
        n_components=32,
    )

    reduced = truncate_colbert_tokens(
        corpus_tokens,
        projector,
    )

    assert reduced.shape == (10, 8, 32)


def test_pca_projection_is_frozen_and_query_independent():
    rng = np.random.default_rng(42)

    corpus = rng.normal(size=(50, 64))
    queries = rng.normal(size=(5, 64))

    # Fit using corpus ONLY.
    projector = fit_corpus_pca(
        corpus,
        n_components=16,
    )

    projection_before = projector.projection_matrix().copy()

    # Apply to queries.
    query_projection = apply_frozen_pca(
        projector,
        queries,
    )

    projection_after = projector.projection_matrix().copy()

    # Transforming queries must not change the PCA projection.
    np.testing.assert_array_equal(
        projection_before,
        projection_after,
    )

    assert query_projection.shape == (5, 16)