import numpy as np

from truncate import (
    fit_corpus_pca,
    apply_frozen_pca,
    fit_colbert_token_pca,
    truncate_colbert_tokens,
    topk_splade,
)


def run_dense_example():
    """Contriever / MedCPT corpus-only PCA example."""
    corpus = np.random.default_rng(0).normal(size=(200, 768))
    queries = np.random.default_rng(1).normal(size=(10, 768))

    projector = fit_corpus_pca(corpus, n_components=128)

    # Queries are transformed using the frozen corpus PCA.
    reduced_queries = apply_frozen_pca(projector, queries)

    return reduced_queries


def run_colbert_example():
    """ColBERT per-token PCA truncation example."""
    rng = np.random.default_rng(0)

    # 20 documents, 32 tokens, original dimension 128.
    corpus_tokens = rng.normal(size=(20, 32, 128))

    projector = fit_colbert_token_pca(
        corpus_tokens,
        n_components=32,
    )

    reduced_tokens = truncate_colbert_tokens(
        corpus_tokens,
        projector,
    )

    return reduced_tokens


def run_splade_example():
    """SPLADE top-k sparse-term pruning example."""
    weights = {
        10: 0.2,
        25: 1.7,
        31: 0.8,
        42: 3.1,
        55: 0.4,
    }

    return topk_splade(weights, k=3)


if __name__ == "__main__":
    dense = run_dense_example()
    colbert = run_colbert_example()
    splade = run_splade_example()

    print("Dense query shape:", dense.shape)
    print("ColBERT token shape:", colbert.shape)
    print("SPLADE top-k:", splade)