from __future__ import annotations

import numpy as np


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds.

    Uses squared Euclidean distance. Intended for evaluation, not for gradients.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("`a` and `b` must be (n, d) and (m, d) with same d.")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")

    # (n, m) squared distances
    d2 = (
        np.sum(a**2, axis=1)[:, None]
        + np.sum(b**2, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    d2 = np.maximum(d2, 0.0)
    return float(np.mean(np.min(d2, axis=1)) + np.mean(np.min(d2, axis=0)))


def silhouette_score_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score with basic guards (optional dependency: scikit-learn already required)."""

    from sklearn.metrics import silhouette_score

    X = np.asarray(X, float)
    labels = np.asarray(labels)
    if X.shape[0] < 3:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))

