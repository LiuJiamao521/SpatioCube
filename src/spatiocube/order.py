from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class OrderConfig:
    subsample_n: int = 2000
    svd_dim: int = 50
    knn: int = 30
    # Use OT-based distance for ordering (robust to rotation/translation).
    use_ot: bool = True
    ot_reg: float = 0.05
    random_state: int = 0


def _subsample_idx(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=max_n, replace=False)


def _stack_X(adatas):
    from scipy.sparse import vstack as sp_vstack

    Xs = [a.X for a in adatas]
    try:
        return sp_vstack(Xs)
    except Exception:
        return np.vstack([np.asarray(x) for x in Xs])


def _require_pot():
    try:
        import ot  # type: ignore

        return ot
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Slice order inference with OT requires optional dependency POT. "
            "Install with `pip install .[align_ot]`."
        ) from e


def _held_karp_best_path(D: np.ndarray) -> list[int]:
    """Exact best Hamiltonian path (min sum of adjacent distances).

    Uses Held–Karp DP for path (not cycle). Complexity O(n^2 2^n), OK for n~10–15.
    """

    D = np.asarray(D, float)
    n = D.shape[0]
    if n <= 2:
        return list(range(n))

    # dp[mask][j] = best cost to reach j using nodes in mask
    # parent[mask][j] = previous node
    size = 1 << n
    dp = np.full((size, n), np.inf, float)
    parent = np.full((size, n), -1, int)

    for j in range(n):
        dp[1 << j, j] = 0.0

    for mask in range(size):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0:
                continue
            # try all i in prev_mask
            m = prev_mask
            while m:
                i = (m & -m).bit_length() - 1
                m &= m - 1
                cand = dp[prev_mask, i] + D[i, j]
                if cand < dp[mask, j]:
                    dp[mask, j] = cand
                    parent[mask, j] = i

    full = size - 1
    end = int(np.argmin(dp[full]))
    # backtrack
    path = []
    mask = full
    j = end
    while j != -1:
        path.append(j)
        pj = parent[mask, j]
        mask ^= 1 << j
        j = pj
    path.reverse()
    return path


def infer_slice_order(
    adatas: list,
    *,
    config: OrderConfig = OrderConfig(),
) -> list[int]:
    """Infer a linear slice order using expression-only similarity.

    Strategy:
    - Subsample each slice
    - Fit a shared TruncatedSVD on concatenated subsamples
    - Compute pairwise similarity via mutual-kNN overlap; convert to distance
    - Solve a greedy shortest path (TSP-like) to get an order
    """

    if len(adatas) < 2:
        return list(range(len(adatas)))

    rng = np.random.default_rng(config.random_state)
    subs = []
    subs_idx = []
    for a in adatas:
        idx = _subsample_idx(int(a.n_obs), config.subsample_n, rng)
        subs_idx.append(idx)
        subs.append(a[idx])

    X = _stack_X(subs)
    n_comp = int(min(config.svd_dim, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=config.random_state)
    E = svd.fit_transform(X)

    # split embeddings back per slice
    embeds = []
    offset = 0
    for a in subs:
        n = int(a.n_obs)
        embeds.append(E[offset : offset + n])
        offset += n

    n_slices = len(embeds)
    sim = np.eye(n_slices, dtype=float)
    for i in range(n_slices):
        for j in range(i + 1, n_slices):
            Ei = embeds[i]
            Ej = embeds[j]
            k_i = min(config.knn, Ej.shape[0])
            k_j = min(config.knn, Ei.shape[0])
            nn_j = NearestNeighbors(n_neighbors=k_i).fit(Ej)
            nn_i = NearestNeighbors(n_neighbors=k_j).fit(Ei)
            nbr_ij = nn_j.kneighbors(Ei, return_distance=False)  # (ni, k)
            nbr_ji = nn_i.kneighbors(Ej, return_distance=False)  # (nj, k)

            # mutual kNN count (approx): i->j edges that are also j->i
            # build boolean adjacency sets
            # For speed: represent each row's neighbor set as Python set on subsample sizes.
            sets_ji = [set(map(int, row)) for row in nbr_ji]
            mutual = 0
            for u in range(nbr_ij.shape[0]):
                for v in nbr_ij[u]:
                    if u in sets_ji[int(v)]:
                        mutual += 1
            denom = nbr_ij.size + 1e-12
            s = mutual / denom
            sim[i, j] = sim[j, i] = float(s)

    # distance (smaller = more similar)
    D_knn = 1.0 - sim
    np.fill_diagonal(D_knn, 0.0)

    if not config.use_ot:
        return _held_karp_best_path(D_knn)

    # OT-based pairwise distance on embeddings (more global + smoother),
    # robust to rotation/translation because it ignores XY altogether.
    ot = _require_pot()
    D = np.zeros((n_slices, n_slices), float)
    for i in range(n_slices):
        for j in range(i + 1, n_slices):
            Ei = embeds[i]
            Ej = embeds[j]
            ni, nj = Ei.shape[0], Ej.shape[0]
            a = np.full((ni,), 1.0 / ni)
            b = np.full((nj,), 1.0 / nj)
            # squared euclidean cost
            C = (
                np.sum(Ei**2, axis=1)[:, None]
                + np.sum(Ej**2, axis=1)[None, :]
                - 2.0 * (Ei @ Ej.T)
            )
            C = np.maximum(C, 0.0)
            Pi = ot.sinkhorn(a, b, C, reg=config.ot_reg)
            dist = float(np.sum(Pi * C))
            D[i, j] = D[j, i] = dist

    # Solve globally smooth best path
    return _held_karp_best_path(D)

