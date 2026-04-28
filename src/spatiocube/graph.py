from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .core import SpatioCube


def build_3d_adjacency(
    cube: SpatioCube,
    *,
    n_intra: int = 15,
    n_inter: int = 5,
    radius_intra: float | None = None,
    radius_inter: float | None = None,
    prefer_mapping: bool = True,
) -> sp.csr_matrix:
    """Build a sparse 3D adjacency graph over all spots across slices.

    - Intra-slice edges: KNN (or radius) within each slice on XY.
    - Inter-slice edges: connect only adjacent slices (i, i+1) using scaled 3D coords.
    """

    # Global indexing
    offsets = np.cumsum([0] + [a.n_obs for a in cube.adatas[:-1]])
    n_total = int(sum(a.n_obs for a in cube.adatas))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Intra-slice
    for si, a in enumerate(cube.adatas):
        base = int(offsets[si])
        xy = np.asarray(a.obsm[cube.spatial_key], float)
        if radius_intra is not None:
            nn = NearestNeighbors(radius=radius_intra).fit(xy)
            ind = nn.radius_neighbors(xy, return_distance=False)
            for i, nbrs in enumerate(ind):
                gi = base + i
                for j in nbrs:
                    if j == i:
                        continue
                    rows.append(gi)
                    cols.append(base + int(j))
                    data.append(1.0)
        else:
            k = min(n_intra + 1, xy.shape[0])
            nn = NearestNeighbors(n_neighbors=k).fit(xy)
            ind = nn.kneighbors(xy, return_distance=False)
            for i, nbrs in enumerate(ind):
                gi = base + i
                for j in nbrs[1:]:
                    rows.append(gi)
                    cols.append(base + int(j))
                    data.append(1.0)

    # Inter-slice: adjacent only
    for si in range(len(cube.adatas) - 1):
        a0 = cube.adatas[si]
        a1 = cube.adatas[si + 1]
        base0 = int(offsets[si])
        base1 = int(offsets[si + 1])

        # Prefer alignment-derived mapping edges if available (creates true cross-slice entities).
        if prefer_mapping:
            mp = a1.uns.get("SpatioCube", {}).get("map_to_prev")
            if isinstance(mp, dict) and "target_indices" in mp:
                tgt_idx = np.asarray(mp["target_indices"])
                if tgt_idx.ndim == 2:
                    for s in range(tgt_idx.shape[0]):
                        gs = base1 + s
                        for t in tgt_idx[s, : min(n_inter, tgt_idx.shape[1])]:
                            rows.append(gs)
                            cols.append(base0 + int(t))
                            data.append(1.0)
                    continue

        X0 = cube.scaled_spatial_3d(si)
        X1 = cube.scaled_spatial_3d(si + 1)

        if radius_inter is not None:
            nn = NearestNeighbors(radius=radius_inter).fit(X1)
            ind = nn.radius_neighbors(X0, return_distance=False)
            for i, nbrs in enumerate(ind):
                gi = base0 + i
                for j in nbrs:
                    rows.append(gi)
                    cols.append(base1 + int(j))
                    data.append(1.0)
        else:
            k = min(n_inter, X1.shape[0])
            nn = NearestNeighbors(n_neighbors=k).fit(X1)
            ind = nn.kneighbors(X0, return_distance=False)
            for i, nbrs in enumerate(ind):
                gi = base0 + i
                for j in nbrs:
                    rows.append(gi)
                    cols.append(base1 + int(j))
                    data.append(1.0)

            # also connect reverse direction
            nn2 = NearestNeighbors(n_neighbors=min(n_inter, X0.shape[0])).fit(X0)
            ind2 = nn2.kneighbors(X1, return_distance=False)
            for j, nbrs in enumerate(ind2):
                gj = base1 + j
                for i in nbrs:
                    rows.append(gj)
                    cols.append(base0 + int(i))
                    data.append(1.0)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    # Symmetrize and binarize
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A


def build_3d_weighted_graph(
    cube: SpatioCube,
    *,
    n_intra: int = 15,
    n_inter: int = 5,
    radius_intra: float | None = None,
    radius_inter: float | None = None,
    prefer_mapping: bool = True,
    alpha_intra: float = 1.0,
    beta_map: float = 2.0,
    gamma_inter: float = 1.0,
    sigma_intra: float | None = None,
    sigma_inter: float | None = None,
    eps: float = 1e-12,
) -> sp.csr_matrix:
    """Build a weighted 3D graph over all spots across slices.

    This is a weighted variant of :func:`build_3d_adjacency` intended for learning /
    diffusion-based methods.

    Edge groups:
    - Intra-slice: spatial neighbors within each slice (XY).
    - Inter-slice: adjacent slices only (scaled 3D coords).
    - Mapping edges: alignment-derived cross-slice correspondences (if present).

    Weights:
    - Intra/inter edges can use a Gaussian kernel on neighbor distances.
    - Mapping edges use a constant weight (beta_map) by default.
    """

    offsets = np.cumsum([0] + [a.n_obs for a in cube.adatas[:-1]])
    n_total = int(sum(a.n_obs for a in cube.adatas))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def _append_edges(base_i: int, base_j: int, ii: int, jj: int, w: float) -> None:
        if ii == jj and base_i == base_j:
            return
        rows.append(base_i + int(ii))
        cols.append(base_j + int(jj))
        data.append(float(w))

    # Intra-slice weighted edges
    for si, a in enumerate(cube.adatas):
        base = int(offsets[si])
        xy = np.asarray(a.obsm[cube.spatial_key], float)
        if xy.shape[0] == 0:
            continue

        if radius_intra is not None:
            nn = NearestNeighbors(radius=radius_intra).fit(xy)
            ind, dist = nn.radius_neighbors(xy, return_distance=True)
            if sigma_intra is None:
                # robust default from observed distances
                all_d = np.concatenate([d for d in dist if len(d) > 0]) if len(dist) else np.array([])
                sig = float(np.median(all_d)) if all_d.size else 1.0
            else:
                sig = float(sigma_intra)
            sig2 = max(sig * sig, eps)
            for i, (nbrs, dists) in enumerate(zip(ind, dist)):
                for j, dij in zip(nbrs, dists):
                    if int(j) == i:
                        continue
                    w = alpha_intra * float(np.exp(-(float(dij) ** 2) / sig2))
                    _append_edges(base, base, i, int(j), w)
        else:
            k = min(n_intra + 1, xy.shape[0])
            nn = NearestNeighbors(n_neighbors=k).fit(xy)
            dist, ind = nn.kneighbors(xy, return_distance=True)
            if sigma_intra is None:
                # Use median of non-self neighbor distances
                d = dist[:, 1:].ravel()
                sig = float(np.median(d)) if d.size else 1.0
            else:
                sig = float(sigma_intra)
            sig2 = max(sig * sig, eps)
            for i in range(ind.shape[0]):
                for j, dij in zip(ind[i, 1:], dist[i, 1:]):
                    w = alpha_intra * float(np.exp(-(float(dij) ** 2) / sig2))
                    _append_edges(base, base, i, int(j), w)

    # Inter-slice (adjacent only) + mapping edges
    for si in range(len(cube.adatas) - 1):
        a0 = cube.adatas[si]
        a1 = cube.adatas[si + 1]
        base0 = int(offsets[si])
        base1 = int(offsets[si + 1])

        # Mapping-derived edges (source slice = a1, target slice = a0)
        if prefer_mapping:
            mp = a1.uns.get("SpatioCube", {}).get("map_to_prev")
            if isinstance(mp, dict) and "target_indices" in mp:
                tgt_idx = np.asarray(mp["target_indices"])
                if tgt_idx.ndim == 2 and tgt_idx.shape[0] == int(a1.n_obs):
                    k = min(int(n_inter), tgt_idx.shape[1])
                    for s in range(tgt_idx.shape[0]):
                        for t in tgt_idx[s, :k]:
                            _append_edges(base1, base0, s, int(t), beta_map)

        X0 = cube.scaled_spatial_3d(si)
        X1 = cube.scaled_spatial_3d(si + 1)

        if X0.shape[0] == 0 or X1.shape[0] == 0:
            continue

        if radius_inter is not None:
            nn = NearestNeighbors(radius=radius_inter).fit(X1)
            ind, dist = nn.radius_neighbors(X0, return_distance=True)
            if sigma_inter is None:
                all_d = np.concatenate([d for d in dist if len(d) > 0]) if len(dist) else np.array([])
                sig = float(np.median(all_d)) if all_d.size else 1.0
            else:
                sig = float(sigma_inter)
            sig2 = max(sig * sig, eps)
            for i, (nbrs, dists) in enumerate(zip(ind, dist)):
                for j, dij in zip(nbrs, dists):
                    w = gamma_inter * float(np.exp(-(float(dij) ** 2) / sig2))
                    _append_edges(base0, base1, i, int(j), w)
        else:
            k = min(int(n_inter), X1.shape[0])
            nn = NearestNeighbors(n_neighbors=k).fit(X1)
            dist, ind = nn.kneighbors(X0, return_distance=True)
            if sigma_inter is None:
                d = dist.ravel()
                sig = float(np.median(d)) if d.size else 1.0
            else:
                sig = float(sigma_inter)
            sig2 = max(sig * sig, eps)
            for i in range(ind.shape[0]):
                for j, dij in zip(ind[i], dist[i]):
                    w = gamma_inter * float(np.exp(-(float(dij) ** 2) / sig2))
                    _append_edges(base0, base1, i, int(j), w)

            # reverse direction as well
            nn2 = NearestNeighbors(n_neighbors=min(int(n_inter), X0.shape[0])).fit(X0)
            dist2, ind2 = nn2.kneighbors(X1, return_distance=True)
            for j in range(ind2.shape[0]):
                for i, dij in zip(ind2[j], dist2[j]):
                    w = gamma_inter * float(np.exp(-(float(dij) ** 2) / sig2))
                    _append_edges(base1, base0, j, int(i), w)

    W = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    # Symmetrize and combine duplicates by summation
    W = (W + W.T).tocsr()
    W.eliminate_zeros()
    return W


def leiden_cluster(
    adjacency: sp.csr_matrix,
    *,
    resolution: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Run Leiden on a sparse adjacency matrix."""

    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Leiden clustering requires optional dependencies. "
            "Install with `pip install .[leiden]`."
        ) from e

    A = adjacency.tocsr()
    A.sort_indices()
    sources, targets = A.nonzero()
    weights = A.data.astype(float, copy=False)

    g = ig.Graph(n=A.shape[0], edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
    g.es["weight"] = weights.tolist()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=random_state,
    )
    return np.asarray(part.membership, dtype=int)

