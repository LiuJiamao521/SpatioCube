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

