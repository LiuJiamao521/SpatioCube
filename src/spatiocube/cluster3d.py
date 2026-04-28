from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .core import SpatioCube
from .graph import build_3d_weighted_graph


@dataclass(frozen=True)
class DiffusionGMMResult:
    labels: np.ndarray
    embedding: np.ndarray
    responsibilities: np.ndarray
    n_iter: int


def _row_normalize(W: sp.csr_matrix, *, eps: float = 1e-12) -> sp.csr_matrix:
    W = W.tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    inv = 1.0 / (d + eps)
    return sp.diags(inv) @ W


def diffuse_features_ppr(
    W: sp.csr_matrix,
    X: np.ndarray,
    *,
    eta: float = 0.5,
    n_steps: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """Personalized-PageRank style diffusion.

    Iteration: Z_{t+1} = (1-eta) X + eta * A * Z_t, where A is row-normalized W.
    """
    if not (0.0 < float(eta) < 1.0):
        raise ValueError("`eta` must be in (0, 1).")
    X = np.asarray(X, dtype=float)
    A = _row_normalize(W, eps=eps)
    Z = X.copy()
    for _ in range(int(n_steps)):
        Z_next = (1.0 - float(eta)) * X + float(eta) * (A @ Z)
        rel = np.linalg.norm(Z_next - Z) / (np.linalg.norm(Z) + eps)
        Z = Z_next
        if rel < float(tol):
            break
    return Z


def _stack_expression_svd(
    cube: SpatioCube, *, svd_dim: int = 50, random_state: int = 0
) -> np.ndarray:
    """Concatenate expression matrices from slices and compute a shared SVD embedding."""
    from scipy.sparse import vstack as sp_vstack  # local import

    Xs = [a.X for a in cube.adatas]
    try:
        Xcat = sp_vstack(Xs)
    except Exception:
        Xcat = np.vstack([np.asarray(x) for x in Xs])

    n_comp = int(min(svd_dim, Xcat.shape[1] - 1)) if Xcat.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    return svd.fit_transform(Xcat)


def _log_gaussian_diag(Z: np.ndarray, mu: np.ndarray, var: np.ndarray, *, eps: float) -> np.ndarray:
    """Return log N(Z | mu, diag(var)) for all k. Shapes: Z(n,d), mu(k,d), var(k,d)."""
    Z = np.asarray(Z, float)
    mu = np.asarray(mu, float)
    var = np.asarray(var, float)
    var = np.maximum(var, eps)
    # (n,k,d)
    diff = Z[:, None, :] - mu[None, :, :]
    quad = np.sum((diff * diff) / var[None, :, :], axis=2)  # (n,k)
    logdet = np.sum(np.log(var), axis=1)[None, :]  # (1,k)
    d = Z.shape[1]
    return -0.5 * (quad + logdet + float(d) * np.log(2.0 * np.pi))


def _softmax_rowwise(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, float)
    L = L - np.max(L, axis=1, keepdims=True)
    P = np.exp(L)
    P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
    return P


def graph_regularized_gmm_mean_field(
    Z: np.ndarray,
    W: sp.csr_matrix,
    *,
    n_clusters: int,
    lam: float = 1.0,
    n_iter: int = 30,
    init: str = "kmeans",
    random_state: int = 0,
    cov_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean-field EM for a diagonal-cov GMM with Potts-like graph regularization.

    We optimize a mean-field approximation:
      q_i(k) ∝ exp( log p(Z_i | k) + lam * sum_j w_ij q_j(k) / deg_i )
    """
    Z = np.asarray(Z, float)
    n, d = Z.shape
    k = int(n_clusters)
    if k <= 1:
        raise ValueError("`n_clusters` must be >= 2.")
    if n < k:
        raise ValueError("Need n_obs >= n_clusters.")

    rng = np.random.default_rng(int(random_state))
    W = W.tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-12)

    # init responsibilities
    if init == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init="auto", random_state=int(random_state))
        lab0 = km.fit_predict(Z)
        Q = np.zeros((n, k), float)
        Q[np.arange(n), lab0] = 1.0
        Q = 0.95 * Q + 0.05 * (1.0 / k)  # avoid exact zeros
    elif init == "random":
        Q = rng.random((n, k))
        Q = Q / (Q.sum(axis=1, keepdims=True) + 1e-12)
    else:
        raise ValueError("`init` must be one of {'kmeans','random'}.")

    pi = Q.mean(axis=0)
    mu = (Q.T @ Z) / (Q.sum(axis=0)[:, None] + 1e-12)  # (k,d)
    # diagonal var per cluster
    var = np.zeros((k, d), float)
    for kk in range(k):
        diff = Z - mu[kk]
        w = Q[:, kk][:, None]
        var[kk] = (w * diff * diff).sum(axis=0) / (w.sum() + 1e-12) + float(cov_eps)

    last_labels: np.ndarray | None = None
    used_iter = 0

    for it in range(int(n_iter)):
        used_iter = it + 1
        # E-step: likelihood term + mean-field graph smoothing term
        logp = _log_gaussian_diag(Z, mu, var, eps=float(cov_eps))
        logp = logp + np.log(np.maximum(pi, 1e-12))[None, :]

        # mean-field neighbor message: (W @ Q) / deg
        msg = (W @ Q) / deg[:, None]
        L = logp + float(lam) * msg
        Q = _softmax_rowwise(L)

        # M-step
        Nk = Q.sum(axis=0) + 1e-12
        pi = Nk / float(n)
        mu = (Q.T @ Z) / Nk[:, None]
        for kk in range(k):
            diff = Z - mu[kk]
            w = Q[:, kk][:, None]
            var[kk] = (w * diff * diff).sum(axis=0) / (w.sum() + 1e-12) + float(cov_eps)

        labels = np.argmax(Q, axis=1)
        if last_labels is not None and np.array_equal(labels, last_labels):
            break
        last_labels = labels

    labels = np.argmax(Q, axis=1).astype(int, copy=False)
    return labels, Q, mu, used_iter


def cluster_3d_diffusion_gmm(
    cube: SpatioCube,
    *,
    n_clusters: int,
    svd_dim: int = 50,
    graph_n_intra: int = 15,
    graph_n_inter: int = 5,
    alpha_intra: float = 1.0,
    beta_map: float = 2.0,
    gamma_inter: float = 1.0,
    eta: float = 0.5,
    diffusion_steps: int = 50,
    diffusion_tol: float = 1e-6,
    lam: float = 1.0,
    em_iters: int = 30,
    init: str = "kmeans",
    random_state: int = 0,
    write_back: bool = True,
) -> DiffusionGMMResult:
    """3D clustering via cross-slice graph diffusion + graph-regularized GMM (mean-field EM).

    Returns labels for all spots across slices (global indexing). If `write_back=True`,
    writes to each slice's `obs[cube.cluster_key]`.
    """
    # 1) Shared expression embedding
    X = _stack_expression_svd(cube, svd_dim=svd_dim, random_state=random_state)

    # 2) Weighted 3D graph (intra + inter + mapping)
    W = build_3d_weighted_graph(
        cube,
        n_intra=graph_n_intra,
        n_inter=graph_n_inter,
        prefer_mapping=True,
        alpha_intra=alpha_intra,
        beta_map=beta_map,
        gamma_inter=gamma_inter,
    )

    # 3) Cross-slice diffusion
    Z = diffuse_features_ppr(
        W,
        X,
        eta=eta,
        n_steps=diffusion_steps,
        tol=diffusion_tol,
    )

    # 4) Graph-regularized clustering (mean-field EM)
    labels, Q, _, used_iter = graph_regularized_gmm_mean_field(
        Z,
        W,
        n_clusters=n_clusters,
        lam=lam,
        n_iter=em_iters,
        init=init,
        random_state=random_state,
    )

    if write_back:
        cube.set_clusters(labels)
        for a in cube.adatas:
            a.uns.setdefault("SpatioCube", {}).update(
                {
                    "cluster_method": "diffusion_gmm_mean_field",
                    "cluster_params": {
                        "n_clusters": int(n_clusters),
                        "svd_dim": int(svd_dim),
                        "graph_n_intra": int(graph_n_intra),
                        "graph_n_inter": int(graph_n_inter),
                        "alpha_intra": float(alpha_intra),
                        "beta_map": float(beta_map),
                        "gamma_inter": float(gamma_inter),
                        "eta": float(eta),
                        "diffusion_steps": int(diffusion_steps),
                        "diffusion_tol": float(diffusion_tol),
                        "lam": float(lam),
                        "em_iters": int(em_iters),
                        "init": str(init),
                        "random_state": int(random_state),
                    },
                }
            )

    return DiffusionGMMResult(
        labels=labels,
        embedding=Z,
        responsibilities=Q,
        n_iter=int(used_iter),
    )

