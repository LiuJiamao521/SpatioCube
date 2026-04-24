from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .core import SpatioCube
from .metrics import chamfer_distance


def _require_pot():
    try:
        import ot  # type: ignore

        return ot
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Alignment via OT requires optional dependency POT. "
            "Install with `pip install .[align_ot]`."
        ) from e


@dataclass(frozen=True)
class AlignResult:
    chamfer_xy: float
    n_source: int
    n_target: int
    method: str


def _subsample_idx(n: int, max_n: int, *, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=max_n, replace=False)


def _procrustes_rigid_2d(
    src_xy: np.ndarray, tgt_xy: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Solve weighted rigid transform in 2D: R @ x + t."""

    src = np.asarray(src_xy, float)
    tgt = np.asarray(tgt_xy, float)
    if weights is None:
        w = np.ones((src.shape[0],), float)
    else:
        w = np.asarray(weights, float)
        w = np.maximum(w, 0.0)
    w = w / (w.sum() + 1e-12)

    mu_s = (src * w[:, None]).sum(axis=0)
    mu_t = (tgt * w[:, None]).sum(axis=0)
    X = src - mu_s
    Y = tgt - mu_t

    # Weighted covariance
    C = (X * w[:, None]).T @ Y
    U, _, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_t - R @ mu_s
    return R, t


def align_adjacent_slices_ot(
    cube: SpatioCube,
    *,
    lambda_z: float | None = None,
    subsample_n: int = 2000,
    paired_subsample: bool = True,
    svd_dim: int = 50,
    expr_knn: int | None = 50,
    transport: str = "emd",
    emd_max_n: int = 2000,
    ot_reg: float = 0.05,
    ot_method: str = "sinkhorn_stabilized",
    ot_num_iter_max: int = 20000,
    ot_stop_thr: float = 1e-9,
    clip_quantile: float | None = None,
    n_iter: int = 3,
    random_state: int = 0,
) -> list[AlignResult]:
    """Coarse-to-fine adjacent-slice rigid alignment in a shared 2D frame.

    - Build a shared expression embedding (truncated SVD on concatenated slices).
    - Estimate a soft/hard correspondence matrix `Pi` between subsampled spots.
    - Fit a 2D rigid transform (rotation + translation) and apply it to the source slice.

    By default uses a **linear assignment** transport (exact one-to-one on the subsample)
    when sizes are equal and not too large; otherwise falls back to **Sinkhorn** (requires POT).

    Writes back updated 2D coordinates into each slice's `obsm[spatial_key]`,
    and updates `spatial_3d` via `cube.write_back()`.
    """

    if lambda_z is not None:
        cube.lambda_z = float(lambda_z)

    rng = np.random.default_rng(random_state)
    results: list[AlignResult] = []
    requested_transport = str(transport).lower()
    ot = None

    # We align slice i+1 into slice i coordinate system (cumulative).
    for i in range(len(cube.adatas) - 1):
        src = cube.adatas[i + 1]
        tgt = cube.adatas[i]

        Xs = src.X
        Xt = tgt.X

        # Use a shared truncated SVD so embeddings live in same space.
        n_comp = int(min(svd_dim, Xs.shape[1] - 1, Xt.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        from scipy.sparse import vstack as sp_vstack  # local import

        try:
            Xcat = sp_vstack([Xs, Xt])
        except Exception:
            Xcat = np.vstack([np.asarray(Xs), np.asarray(Xt)])
        Ecat = svd.fit_transform(Xcat)
        Es = Ecat[: Xs.shape[0]]
        Et = Ecat[Xs.shape[0] :]

        src_xy = np.asarray(src.obsm[cube.spatial_key], float)
        tgt_xy = np.asarray(tgt.obsm[cube.spatial_key], float)

        ns_full, nt_full = int(src_xy.shape[0]), int(tgt_xy.shape[0])
        if paired_subsample and ns_full == nt_full:
            isrc = _subsample_idx(ns_full, subsample_n, rng=rng)
            itgt = isrc
        elif paired_subsample:
            # Unequal slice sizes: subsample sources, then pair each source embedding
            # to its nearest target embedding (still cheap at subsample_n scale).
            isrc = _subsample_idx(ns_full, subsample_n, rng=rng)
            Et_all = Et
            nn_pair = NearestNeighbors(n_neighbors=1).fit(Et_all)
            itgt = nn_pair.kneighbors(Es[isrc], return_distance=False).ravel().astype(int, copy=False)
            itgt = np.unique(itgt)
            if itgt.size < min(ns_full, subsample_n) // 2:
                # Fallback if pairing collapses too aggressively
                itgt = _subsample_idx(nt_full, subsample_n, rng=rng)
        else:
            isrc = _subsample_idx(ns_full, subsample_n, rng=rng)
            itgt = _subsample_idx(nt_full, subsample_n, rng=rng)

        Es_s = Es[isrc]
        Et_s = Et[itgt]
        src_xy_s = src_xy[isrc]
        tgt_xy_s = tgt_xy[itgt]

        ns, nt = int(src_xy_s.shape[0]), int(tgt_xy_s.shape[0])

        transport_mode = requested_transport
        if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"}:
            if ns != nt or ns > int(emd_max_n):
                transport_mode = "sinkhorn"
                warnings.warn(
                    "Falling back to Sinkhorn OT for this adjacent pair: "
                    f"requested transport={requested_transport!r} but subsample sizes "
                    f"({ns}, {nt}) are incompatible with linear assignment or exceed "
                    f"`emd_max_n={int(emd_max_n)}`.",
                    RuntimeWarning,
                )

        if transport_mode in {"sinkhorn", "ot"} and ot is None:
            ot = _require_pot()

        # Expression cost matrix in shared embedding space.
        # For moderate subsample sizes, a dense squared-distance matrix is more stable
        # than KNN-sparsified costs (which can accidentally exclude the true match when
        # subsampled source/target indices are not aligned).
        if expr_knn is None or expr_knn <= 0 or expr_knn >= nt:
            # (ns, d) vs (nt, d) -> (ns, nt)
            Es2 = np.sum(Es_s * Es_s, axis=1, keepdims=True)
            Et2 = np.sum(Et_s * Et_s, axis=1, keepdims=True).T
            cross = Es_s @ Et_s.T
            C = np.maximum(Es2 + Et2 - 2.0 * cross, 0.0)
        else:
            # Candidate restriction via expression KNN in shared embedding space.
            nn = NearestNeighbors(n_neighbors=min(int(expr_knn), Et_s.shape[0])).fit(Et_s)
            knn_idx = nn.kneighbors(Es_s, return_distance=False)  # (ns, k)

            C = np.full((ns, nt), np.inf, dtype=float)
            for r in range(ns):
                cand = knn_idx[r]
                diff = Et_s[cand] - Es_s[r]
                C[r, cand] = np.sum(diff * diff, axis=1)

        # Optional clipping: keep only lower-cost entries per row.
        if clip_quantile is not None and 0.0 < float(clip_quantile) < 1.0:
            thr = np.zeros((ns, 1), dtype=float)
            for r in range(ns):
                row = C[r]
                row_f = row[np.isfinite(row)]
                if row_f.size == 0:
                    thr[r, 0] = np.inf
                else:
                    thr[r, 0] = float(np.quantile(row_f, clip_quantile))
            C = np.where((C <= thr) & np.isfinite(C), C, np.inf)

        finite = np.isfinite(C)
        if not np.any(finite):
            raise RuntimeError("OT cost matrix has no finite entries; check `expr_knn` / embeddings.")
        # Replace masked/clipped entries with a *row-local* moderate penalty.
        # Using a global cap derived from max(C[finite]) can make every row's penalty
        # enormous after row-wise shifting, which destroys Sinkhorn conditioning.
        cap_rows = np.zeros((ns, 1), dtype=float)
        for r in range(ns):
            row = C[r]
            row_f = row[np.isfinite(row)]
            mx = float(np.max(row_f)) if row_f.size else 1.0
            if not np.isfinite(mx) or mx <= 0.0:
                mx = 1.0
            cap_rows[r, 0] = max(10.0 * mx, 1.0)
        C = np.where(np.isfinite(C), C, cap_rows)

        # Stabilize Sinkhorn: row-wise shift does not change the optimal coupling for
        # entropic OT with uniform marginals, but prevents huge dynamic range when
        # many entries are masked to a large constant.
        C = C - np.min(C, axis=1, keepdims=True)

        # Break symmetries for discrete solvers (linear assignment): when many permutations
        # achieve the same expression cost (common under paired subsampling), prefer the
        # identity pairing i -> i (within the subsampled index order).
        if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"} and ns == nt:
            max_finite = float(np.max(C[np.isfinite(C)])) if np.any(np.isfinite(C)) else 1.0
            if not np.isfinite(max_finite) or max_finite <= 0.0:
                max_finite = 1.0
            # Must be large enough to dominate typical numerical noise in the expression cost,
            # otherwise `linear_sum_assignment` may pick an arbitrary zero-cost permutation.
            eps = 1e-6 * max(max_finite, 1.0)
            idx = np.arange(ns, dtype=float)[:, None] - np.arange(nt, dtype=float)[None, :]
            C = C + eps * (idx * idx)

        a = np.full((ns,), 1.0 / ns)
        b = np.full((nt,), 1.0 / nt)

        # Cumulative rigid transform in the original source coordinate frame:
        # x' = x @ R_acc.T + t_acc
        R_acc = np.eye(2, dtype=float)
        t_acc = np.zeros(2, dtype=float)
        src_xy_s0 = np.asarray(src_xy_s, dtype=float, order="C")

        for _ in range(n_iter):
            if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"}:
                if ns != nt:
                    raise ValueError(
                        "`transport='emd'` (linear assignment) requires equal subsample sizes; "
                        "try increasing `subsample_n`, disabling `paired_subsample`, or use `transport='sinkhorn'`."
                    )
                r_ind, c_ind = linear_sum_assignment(C)
                Pi = np.zeros((ns, nt), dtype=float)
                Pi[r_ind, c_ind] = 1.0
                # Match `sinkhorn` convention: rows sum to `a`.
                Pi = Pi * a[:, None]
            elif transport_mode in {"sinkhorn", "ot"}:
                assert ot is not None
                Pi = ot.sinkhorn(
                    a,
                    b,
                    C,
                    reg=ot_reg,
                    method=str(ot_method),
                    numItermax=int(ot_num_iter_max),
                    stopThr=float(ot_stop_thr),
                    warn=False,
                )
            else:
                raise ValueError("`transport` must be one of {'emd','sinkhorn'}.")

            # Barycentric mapping in target coordinates. Note: when `Pi` rows sum to a
            # constant (e.g. uniform OT marginals), `Pi @ Y` scales `Y` by that constant;
            # Procrustes expects true coordinates, so we rescale per-row.
            w = Pi.sum(axis=1)
            mapped = (Pi @ tgt_xy_s) / (w[:, None] + 1e-12)
            src_cur = (src_xy_s0 @ R_acc.T) + t_acc[None, :]
            R_delta, t_delta = _procrustes_rigid_2d(src_cur, mapped, weights=w)
            R_acc = R_delta @ R_acc
            t_acc = (R_delta @ t_acc) + t_delta

        # Apply final transform to full-resolution source slice
        src_xy_full = (src_xy @ R_acc.T) + t_acc[None, :]
        src.obsm[cube.spatial_key] = src_xy_full

        # Save a lightweight full-resolution mapping (source->target) for clustering:
        # for each source spot, connect to top-k nearest targets in embedding space.
        nn_full = NearestNeighbors(n_neighbors=min(5, Et.shape[0])).fit(Et)
        tgt_nbrs = nn_full.kneighbors(Es, return_distance=False)  # (n_src, k)
        src.uns.setdefault("SpatioCube", {})["map_to_prev"] = {
            "target_slice_index": i,
            "target_indices": tgt_nbrs.astype(np.int32, copy=False),
        }

        cube.write_back()
        src_xy_s_final = (src_xy_s0 @ R_acc.T) + t_acc[None, :]
        chamfer = chamfer_distance(src_xy_s_final, tgt_xy_s)
        align_method = "coarse_emd_rigid" if transport_mode in {
            "emd",
            "lap",
            "linear_assignment",
            "linear_sum_assignment",
        } else "coarse_sinkhorn_rigid"

        results.append(
            AlignResult(
                chamfer_xy=chamfer,
                n_source=int(src.n_obs),
                n_target=int(tgt.n_obs),
                method=align_method,
            )
        )

    # Record metadata
    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(
            {
                "align_method": "coarse_adjacent_rigid",
                "align_params": {
                    "subsample_n": subsample_n,
                    "paired_subsample": paired_subsample,
                    "svd_dim": svd_dim,
                    "expr_knn": expr_knn,
                    "transport_requested": str(transport),
                    "emd_max_n": int(emd_max_n),
                    "ot_reg": ot_reg,
                    "ot_method": str(ot_method),
                    "ot_num_iter_max": int(ot_num_iter_max),
                    "ot_stop_thr": float(ot_stop_thr),
                    "clip_quantile": clip_quantile,
                    "n_iter": n_iter,
                    "random_state": random_state,
                },
            }
        )
    return results

