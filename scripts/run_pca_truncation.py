import numpy as np

from truncate import (
    fit_corpus_pca,
    apply_frozen_pca,
    fit_colbert_token_pca,
    truncate_colbert_tokens,
    topk_splade,
)


def run_dense_example():
    """Contriever / MedCPT corpus-only PCA."""
    rng = np.random.default_rng(0)

    corpus = rng.normal(size=(200, 768))
    queries = rng.normal(size=(10, 768))

    projector = fit_corpus_pca(
        corpus,
        n_components=128,
    )

    reduced_queries = apply_frozen_pca(
        projector,
        queries,
    )

    return reduced_queries


def run_colbert_example():
    """ColBERT per-token PCA truncation."""
    rng = np.random.default_rng(0)

    # [num_docs, num_tokens, 128]
    corpus_tokens = rng.normal(
        size=(20, 32, 128)
    )

    # PCA is fitted on the complete corpus token pool.
    projector = fit_colbert_token_pca(
        corpus_tokens,
        n_components=32,
    )

    # Tokens are projected and reshaped back.
    reduced_tokens = truncate_colbert_tokens(
        corpus_tokens,
        projector,
    )

    return reduced_tokens


def run_splade_example():
    """SPLADE top-k sparse-term pruning."""

    weights = {
        10: 0.2,
        25: 1.7,
        31: 0.8,
        42: 3.1,
        55: 0.4,
    }

    # Keep only the 3 highest-weight terms.
    truncated = topk_splade(
        weights,
        k=3,
    )

    return truncated


if __name__ == "__main__":

    dense = run_dense_example()
    colbert = run_colbert_example()
    splade = run_splade_example()

    print("Dense / Contriever / MedCPT:")
    print("  Original dimension: 768")
    print("  Truncated dimension:", dense.shape)

    print("\nColBERT:")
    print("  Original shape: (20, 32, 128)")
    print("  Truncated shape:", colbert.shape)

    print("\nSPLADE:")
    print("  Top-k terms:", splade)
    print("  Number of retained terms:", len(splade))