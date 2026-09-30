"""Dimensionality/truncation utilities for dense, late-interaction, and sparse IR.

The key leakage rule is intentional: PCA is fit on corpus representations only.
Query representations are transformed with the frozen corpus PCA and never used
when fitting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PCAProjector:
    """A frozen PCA projector.

    Attributes
    ----------
    mean_:
        Corpus-only feature mean, shape ``(D,)``.
    components_:
        Corpus-only principal directions, shape ``(n_components, D)``.
    explained_variance_:
        Variance captured by each retained component.
    """

    mean_: Array
    components_: Array
    explained_variance_: Array

    @classmethod
    def fit(cls, corpus: Array, n_components: int) -> "PCAProjector":
        """Fit PCA using *only* corpus vectors.

        Parameters
        ----------
        corpus: array of shape ``(n_items, D)``
        n_components: retained dimension, ``1 <= k <= D``
        """
        x = _as_2d_float(corpus, "corpus")
        n, d = x.shape
        if n == 0:
            raise ValueError("corpus must contain at least one vector")
        if not 1 <= n_components <= min(n, d):
            raise ValueError(
                f"n_components must be in [1, min(n_samples, n_features)] = "
                f"[1, {min(n, d)}], got {n_components}"
            )

        mean = x.mean(axis=0)
        centered = x - mean
        # SVD gives the PCA directions without materializing a D x D covariance.
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        denom = max(n - 1, 1)
        variance = (singular_values ** 2) / denom
        return cls(
            mean_=mean.copy(),
            components_=vt[:n_components].copy(),
            explained_variance_=variance[:n_components].copy(),
        )

    def transform(self, vectors: Array) -> Array:
        """Project vectors with the frozen corpus-only PCA."""
        x = _as_float(vectors, "vectors")
        if x.shape[-1] != self.mean_.shape[0]:
            raise ValueError(
                f"last dimension must be {self.mean_.shape[0]}, got {x.shape[-1]}"
            )
        return (x - self.mean_) @ self.components_.T

    def projection_matrix(self) -> Array:
        """Return a copy of the retained projection directions."""
        return self.components_.copy()


def fit_corpus_pca(corpus_embeddings: Array, n_components: int) -> PCAProjector:
    """Fit PCA on corpus embeddings only (MedCPT/Contriever use case)."""
    return PCAProjector.fit(corpus_embeddings, n_components)


def apply_frozen_pca(projector: PCAProjector, queries: Array) -> Array:
    """Apply a corpus-fitted PCA to query embeddings without refitting."""
    return projector.transform(queries)


def fit_colbert_token_pca(
    corpus_token_vectors: Array, n_components: int
) -> PCAProjector:
    """Fit PCA over corpus ColBERT token vectors.

    ``corpus_token_vectors`` is ``(n_docs, n_tokens, 128)`` (or any final
    feature dimension). Tokens are treated as individual observations, while
    their document/token structure is restored when transforming.
    """
    x = _as_3d_float(corpus_token_vectors, "corpus_token_vectors")
    n_docs, n_tokens, dim = x.shape
    projector = PCAProjector.fit(x.reshape(n_docs * n_tokens, dim), n_components)
    return projector


def truncate_colbert_tokens(
    token_vectors: Array, projector: PCAProjector
) -> Array:
    """PCA-truncate every ColBERT token vector using a frozen projector.

    Input shape: ``(..., n_tokens, D)``. Output shape:
    ``(..., n_tokens, n_components)``.
    """
    x = _as_float(token_vectors, "token_vectors")
    if x.ndim < 2:
        raise ValueError("token_vectors must have at least 2 dimensions")
    flat = x.reshape(-1, x.shape[-1])
    reduced = projector.transform(flat)
    return reduced.reshape(*x.shape[:-1], projector.components_.shape[0])


def topk_splade(
    weights: Mapping[int, float] | Sequence[tuple[int, float]], k: int
) -> dict[int, float]:
    """Keep the top-k SPLADE terms by sparse weight.

    SPLADE weights are normally non-negative; ranking by absolute value also
    makes the helper safe for signed sparse vectors. Ties are resolved by term
    id so results are deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    items = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    normalized = [(int(term), float(weight)) for term, weight in items if weight != 0]
    ranked = sorted(normalized, key=lambda tw: (-abs(tw[1]), tw[0]))
    return dict(ranked[:k])


def truncate_splade_batch(
    sparse_vectors: Iterable[Mapping[int, float] | Sequence[tuple[int, float]]], k: int
) -> list[dict[int, float]]:
    """Apply top-k term pruning independently to each SPLADE vector."""
    return [topk_splade(vector, k) for vector in sparse_vectors]


def _as_float(x: Array, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _as_2d_float(x: Array, name: str) -> Array:
    arr = _as_float(x, name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_items, dim), got {arr.shape}")
    return arr


def _as_3d_float(x: Array, name: str) -> Array:
    arr = _as_float(x, name)
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_items, n_tokens, dim), got {arr.shape}"
        )
    return arr
