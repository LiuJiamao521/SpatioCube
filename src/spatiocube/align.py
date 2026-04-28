from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import least_squares
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


def _stack_X(adatas):
    from scipy.sparse import vstack as sp_vstack

    Xs = [a.X for a in adatas]
    try:
        return sp_vstack(Xs)
    except Exception:
        return np.vstack([np.asarray(x) for x in Xs])


def _fit_global_svd_and_transform(
    cube: SpatioCube,
    *,
    subsample_n: int,
    svd_dim: int,
    random_state: int,
    feature_mode: Literal["svd", "svd_smooth"] = "svd",
    smooth_k: int = 15,
    smooth_alpha: float = 0.7,
    smooth_steps: int = 2,
) -> list[np.ndarray]:
    """Fit one SVD on concatenated subsamples, then transform full X for each slice.

    This avoids re-fitting SVD for every adjacent pair, which is a major runtime cost
    on large datasets (many slices, many genes).
    """
    rng = np.random.default_rng(int(random_state))
    subs = []
    for a in cube.adatas:
        idx = _subsample_idx(int(a.n_obs), int(subsample_n), rng=rng)
        subs.append(a[idx])
    X_fit = _stack_X(subs)
    if X_fit.shape[1] <= 1:
        n_comp = 1
    else:
        n_comp = int(min(int(svd_dim), int(X_fit.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=int(random_state))
    svd.fit(X_fit)
    embeds = []
    for a in cube.adatas:
        embeds.append(svd.transform(a.X))

    if feature_mode == "svd":
        return embeds
    if feature_mode != "svd_smooth":
        raise ValueError("`feature_mode` must be one of {'svd','svd_smooth'}.")

    # Spatial smoothing per slice (no clustering labels; purely continuous features).
    # E_{t+1} = (1-alpha) E0 + alpha * P * E_t, where P is row-normalized spatial kNN adjacency.
    smoothed: list[np.ndarray] = []
    for a, E0 in zip(cube.adatas, embeds):
        xy = np.asarray(a.obsm[cube.spatial_key], float)
        n = int(xy.shape[0])
        if n == 0:
            smoothed.append(E0)
            continue
        k = int(min(max(3, smooth_k), n - 1)) if n > 1 else 0
        if k <= 0:
            smoothed.append(E0)
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
        ind = nn.kneighbors(xy, return_distance=False)[:, 1:]  # drop self
        # Row-normalized adjacency as an implicit operator (avoid building sparse matrix).
        E = np.asarray(E0, float)
        E_init = E.copy()
        alpha = float(smooth_alpha)
        steps = int(max(1, smooth_steps))
        for _ in range(steps):
            # neighbor mean
            E_nbr = np.mean(E[ind], axis=1)
            E = (1.0 - alpha) * E_init + alpha * E_nbr
        smoothed.append(E.astype(np.float32, copy=False))
    return smoothed


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


def _procrustes_similarity_2d(
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve weighted similarity transform in 2D: s * (R @ x) + t.

    This helps when organ size gradually changes across slices (global isotropic scaling).
    """
    src = np.asarray(src_xy, float)
    tgt = np.asarray(tgt_xy, float)
    if weights is None:
        w = np.ones((src.shape[0],), float)
    else:
        w = np.asarray(weights, float)
        w = np.maximum(w, 0.0)
    w = w / (w.sum() + eps)

    mu_s = (src * w[:, None]).sum(axis=0)
    mu_t = (tgt * w[:, None]).sum(axis=0)
    X = src - mu_s
    Y = tgt - mu_t

    C = (X * w[:, None]).T @ Y
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Weighted variance of source
    var_s = float((w[:, None] * (X * X)).sum())
    if not np.isfinite(var_s) or var_s <= eps:
        s = 1.0
    else:
        s = float(np.sum(S) / (var_s + eps))
    s = float(np.clip(s, float(scale_min), float(scale_max)))
    t = mu_t - s * (R @ mu_s)
    return s, R, t


def _alignment_pairs(
    n_slices: int,
    strategy: Literal["sequential", "middle_out"],
    anchor_index: int | None,
) -> list[tuple[int, int]]:
    """Return list of (i_src, i_tgt): transform slice i_src into slice i_tgt frame."""

    if n_slices < 2:
        return []
    if strategy == "sequential":
        return [(i + 1, i) for i in range(n_slices - 1)]
    if strategy == "middle_out":
        # Anchor at floor((n-1)/2) so n==2 yields pairs [(1,0)] (not empty).
        default_anchor = (n_slices - 1) // 2
        m = default_anchor if anchor_index is None else int(anchor_index)
        m = max(0, min(m, n_slices - 1))
        pairs: list[tuple[int, int]] = []
        # Left: bring slices m-1..0 into the frame of their right neighbor (already linked to m).
        for j in range(m - 1, -1, -1):
            pairs.append((j, j + 1))
        # Right: classic chain from anchor toward the end (same as sequential but starting at m).
        for j in range(m, n_slices - 1):
            pairs.append((j + 1, j))
        return pairs
    raise ValueError("`strategy` must be one of {'sequential','middle_out'}.")


def _store_map_to_prev(
    cube: SpatioCube,
    i_src: int,
    i_tgt: int,
    Es: np.ndarray,
    Et: np.ndarray,
) -> None:
    """Store KNN mapping on the *higher slice index* for graph edges (si, si+1).

    `build_3d_weighted_graph` / `build_3d_adjacency` expect `map_to_prev` on
    `cube.adatas[k]` with `target_slice_index == k-1` for each k>=1.
    """

    k_hi = max(i_src, i_tgt)
    k_lo = min(i_src, i_tgt)
    if k_hi <= 0:
        return
    if k_hi != k_lo + 1:
        raise ValueError("Internal: map_to_prev only defined for consecutive slice indices.")

    Es_hi = Es if i_src == k_hi else Et
    Et_lo = Et if i_tgt == k_lo else Es
    # Neighbors in `lo` for each spot on `hi`
    nn_full = NearestNeighbors(n_neighbors=min(5, Et_lo.shape[0])).fit(Et_lo)
    tgt_nbrs = nn_full.kneighbors(Es_hi, return_distance=False)
    cube.adatas[k_hi].uns.setdefault("SpatioCube", {})["map_to_prev"] = {
        "target_slice_index": int(k_lo),
        "target_indices": tgt_nbrs.astype(np.int32, copy=False),
    }


def _align_pair_ot(
    cube: SpatioCube,
    i_src: int,
    i_tgt: int,
    *,
    rng: np.random.Generator,
    subsample_n: int,
    paired_subsample: bool,
    svd_dim: int,
    expr_knn: int | None,
    requested_transport: str,
    emd_max_n: int,
    ot_reg: float,
    ot_method: str,
    ot_num_iter_max: int,
    ot_stop_thr: float,
    clip_quantile: float | None,
    n_iter: int,
    random_state: int,
    ot_mod: object | None,
    embeds: list[np.ndarray] | None,
    store_adjacent_mapping: bool = True,
    allow_scale: bool = False,
    scale_min: float = 0.98,
    scale_max: float = 1.02,
    spatial_weight: float = 0.0,
    spatial_sigma: float | None = None,
    transport_unbalanced: bool = False,
    ot_reg_m: float = 10.0,
) -> tuple[AlignResult, object | None]:
    """Align slice i_src into slice i_tgt coordinate frame (2D rigid after OT)."""

    src = cube.adatas[i_src]
    tgt = cube.adatas[i_tgt]

    if embeds is None:
        Xs = src.X
        Xt = tgt.X
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
    else:
        Es = embeds[i_src]
        Et = embeds[i_tgt]

    src_xy = np.asarray(src.obsm[cube.spatial_key], float)
    tgt_xy = np.asarray(tgt.obsm[cube.spatial_key], float)

    ns_full, nt_full = int(src_xy.shape[0]), int(tgt_xy.shape[0])
    if paired_subsample and ns_full == nt_full:
        isrc = _subsample_idx(ns_full, subsample_n, rng=rng)
        itgt = isrc
    elif paired_subsample:
        isrc = _subsample_idx(ns_full, subsample_n, rng=rng)
        Et_all = Et
        nn_pair = NearestNeighbors(n_neighbors=1).fit(Et_all)
        itgt = nn_pair.kneighbors(Es[isrc], return_distance=False).ravel().astype(int, copy=False)
        itgt = np.unique(itgt)
        if itgt.size < min(ns_full, subsample_n) // 2:
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

    ot = ot_mod
    if transport_mode in {"sinkhorn", "ot"} and ot is None:
        ot = _require_pot()

    if expr_knn is None or expr_knn <= 0 or expr_knn >= nt:
        Es2 = np.sum(Es_s * Es_s, axis=1, keepdims=True)
        Et2 = np.sum(Et_s * Et_s, axis=1, keepdims=True).T
        cross = Es_s @ Et_s.T
        C = np.maximum(Es2 + Et2 - 2.0 * cross, 0.0)
    else:
        nn = NearestNeighbors(n_neighbors=min(int(expr_knn), Et_s.shape[0])).fit(Et_s)
        knn_idx = nn.kneighbors(Es_s, return_distance=False)

        C = np.full((ns, nt), np.inf, dtype=float)
        for r in range(ns):
            cand = knn_idx[r]
            diff = Et_s[cand] - Es_s[r]
            C[r, cand] = np.sum(diff * diff, axis=1)

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
    cap_rows = np.zeros((ns, 1), dtype=float)
    for r in range(ns):
        row = C[r]
        row_f = row[np.isfinite(row)]
        mx = float(np.max(row_f)) if row_f.size else 1.0
        if not np.isfinite(mx) or mx <= 0.0:
            mx = 1.0
        cap_rows[r, 0] = max(10.0 * mx, 1.0)
    C = np.where(np.isfinite(C), C, cap_rows)

    C = C - np.min(C, axis=1, keepdims=True)

    if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"} and ns == nt:
        max_finite = float(np.max(C[np.isfinite(C)])) if np.any(np.isfinite(C)) else 1.0
        if not np.isfinite(max_finite) or max_finite <= 0.0:
            max_finite = 1.0
        eps = 1e-6 * max(max_finite, 1.0)
        idx = np.arange(ns, dtype=float)[:, None] - np.arange(nt, dtype=float)[None, :]
        C = C + eps * (idx * idx)

    a = np.full((ns,), 1.0 / ns)
    b = np.full((nt,), 1.0 / nt)

    # Spatial gating scale (in target XY units). When provided, we combine expression cost with
    # a spatial cost based on the current transformed source positions (helps partial overlap).
    if spatial_weight > 0.0:
        if spatial_sigma is None:
            # Use a robust default from target kNN distances
            k_sig = min(10, max(3, min(nt, 10)))
            nn_sig = NearestNeighbors(n_neighbors=k_sig).fit(tgt_xy_s)
            d_sig, _ = nn_sig.kneighbors(tgt_xy_s, return_distance=True)
            sig = float(np.median(d_sig[:, 1:])) if d_sig.size else 1.0
        else:
            sig = float(spatial_sigma)
        sig2 = max(sig * sig, 1e-12)
    else:
        sig2 = 1.0

    R_acc = np.eye(2, dtype=float)
    s_acc = 1.0
    t_acc = np.zeros(2, dtype=float)
    src_xy_s0 = np.asarray(src_xy_s, dtype=float, order="C")

    assert ot is not None or transport_mode not in {"sinkhorn", "ot"}
    for _ in range(n_iter):
        # Combine expression cost with spatial cost around current estimate to suppress
        # spurious matches from non-overlapping regions.
        if spatial_weight > 0.0:
            src_cur = (s_acc * (src_xy_s0 @ R_acc.T)) + t_acc[None, :]
            # squared distance matrix between current src and target subsample
            Ss2 = np.sum(src_cur * src_cur, axis=1, keepdims=True)
            Tt2 = np.sum(tgt_xy_s * tgt_xy_s, axis=1, keepdims=True).T
            cross_xy = src_cur @ tgt_xy_s.T
            Cxy = np.maximum(Ss2 + Tt2 - 2.0 * cross_xy, 0.0) / sig2
            C_use = C + float(spatial_weight) * Cxy
        else:
            C_use = C

        if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"}:
            if ns != nt:
                raise ValueError(
                    "`transport='emd'` (linear assignment) requires equal subsample sizes; "
                    "try increasing `subsample_n`, disabling `paired_subsample`, or use `transport='sinkhorn'`."
                )
            r_ind, c_ind = linear_sum_assignment(C_use)
            Pi = np.zeros((ns, nt), dtype=float)
            Pi[r_ind, c_ind] = 1.0
            Pi = Pi * a[:, None]
        elif transport_mode in {"sinkhorn", "ot"}:
            assert ot is not None
            if transport_unbalanced:
                # Allow unmatched mass (partial overlap / size change).
                # Uses unbalanced Sinkhorn: POT ot.unbalanced.sinkhorn_unbalanced
                Pi = ot.unbalanced.sinkhorn_unbalanced(
                    a,
                    b,
                    C_use,
                    reg=float(ot_reg),
                    reg_m=float(ot_reg_m),
                    method=str(ot_method),
                    numItermax=int(ot_num_iter_max),
                    stopThr=float(ot_stop_thr),
                    warn=False,
                )
            else:
                Pi = ot.sinkhorn(
                    a,
                    b,
                    C_use,
                    reg=ot_reg,
                    method=str(ot_method),
                    numItermax=int(ot_num_iter_max),
                    stopThr=float(ot_stop_thr),
                    warn=False,
                )
        else:
            raise ValueError("`transport` must be one of {'emd','sinkhorn'}.")

        w = Pi.sum(axis=1)
        mapped = (Pi @ tgt_xy_s) / (w[:, None] + 1e-12)
        src_cur = (s_acc * (src_xy_s0 @ R_acc.T)) + t_acc[None, :]
        if allow_scale:
            s_delta, R_delta, t_delta = _procrustes_similarity_2d(
                src_cur,
                mapped,
                weights=w,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            s_acc = float(s_delta) * s_acc
            R_acc = R_delta @ R_acc
            t_acc = (float(s_delta) * (R_delta @ t_acc)) + t_delta
        else:
            R_delta, t_delta = _procrustes_rigid_2d(src_cur, mapped, weights=w)
            R_acc = R_delta @ R_acc
            t_acc = (R_delta @ t_acc) + t_delta

    src_xy_full = (s_acc * (src_xy @ R_acc.T)) + t_acc[None, :]
    src.obsm[cube.spatial_key] = src_xy_full

    if store_adjacent_mapping:
        _store_map_to_prev(cube, i_src, i_tgt, Es, Et)

    cube.write_back()
    src_xy_s_final = (src_xy_s0 @ R_acc.T) + t_acc[None, :]
    chamfer = chamfer_distance(src_xy_s_final, tgt_xy_s)
    align_method = "coarse_emd_rigid" if transport_mode in {
        "emd",
        "lap",
        "linear_assignment",
        "linear_sum_assignment",
    } else "coarse_sinkhorn_rigid"
    if allow_scale:
        align_method = align_method.replace("rigid", "similarity")
    if spatial_weight > 0.0:
        align_method = align_method + "_spatial"
    if transport_unbalanced:
        align_method = align_method + "_unbalanced"

    return (
        AlignResult(
            chamfer_xy=chamfer,
            n_source=int(src.n_obs),
            n_target=int(tgt.n_obs),
            method=align_method,
        ),
        ot,
    )


def _rotmat(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def _refresh_adjacent_map_to_prev_from_embedding(
    cube: SpatioCube,
    embeds: list[np.ndarray],
    *,
    n_neighbors: int = 5,
) -> None:
    """(Re)build `map_to_prev` for all adjacent pairs using embedding KNN."""
    n_slices = len(cube.adatas)
    for k in range(1, n_slices):
        Es = embeds[k]
        Et = embeds[k - 1]
        nn = NearestNeighbors(n_neighbors=min(int(n_neighbors), Et.shape[0])).fit(Et)
        idx = nn.kneighbors(Es, return_distance=False)
        cube.adatas[k].uns.setdefault("SpatioCube", {})["map_to_prev"] = {
            "target_slice_index": int(k - 1),
            "target_indices": idx.astype(np.int32, copy=False),
        }


def align_slices_to_anchor_ot(
    cube: SpatioCube,
    *,
    anchor_index: int | None = None,
    subsample_n: int = 2000,
    svd_dim: int = 50,
    feature_mode: Literal["svd", "svd_smooth"] = "svd_smooth",
    smooth_k: int = 15,
    smooth_alpha: float = 0.7,
    smooth_steps: int = 2,
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
    shared_embedding: bool = True,
    refresh_mapping: bool = True,
) -> dict[str, object]:
    """Global rigid alignment by directly registering every slice to a single anchor slice.

    Compared with chaining adjacent pairs, this star topology avoids accumulated drift
    (often seen as a spiral/twist across long series).
    """
    n_slices = len(cube.adatas)
    if n_slices < 2:
        return {"status": "skipped", "reason": "n_slices<2"}

    if anchor_index is None:
        anchor_index = (n_slices - 1) // 2
    anchor_index = int(anchor_index)
    if anchor_index < 0 or anchor_index >= n_slices:
        raise ValueError("`anchor_index` out of range.")

    rng = np.random.default_rng(int(random_state))
    requested_transport = str(transport).lower()
    ot: object | None = None

    embeds: list[np.ndarray] | None = None
    if shared_embedding:
        embeds = _fit_global_svd_and_transform(
            cube,
            subsample_n=subsample_n,
            svd_dim=svd_dim,
            random_state=random_state,
            feature_mode=feature_mode,
            smooth_k=smooth_k,
            smooth_alpha=smooth_alpha,
            smooth_steps=smooth_steps,
        )

    results: dict[int, AlignResult] = {}
    for i in range(n_slices):
        if i == anchor_index:
            continue
        res, ot = _align_pair_ot(
            cube,
            i_src=i,
            i_tgt=anchor_index,
            rng=rng,
            subsample_n=subsample_n,
            paired_subsample=False,
            svd_dim=svd_dim,
            expr_knn=expr_knn,
            requested_transport=requested_transport,
            emd_max_n=emd_max_n,
            ot_reg=ot_reg,
            ot_method=ot_method,
            ot_num_iter_max=ot_num_iter_max,
            ot_stop_thr=ot_stop_thr,
            clip_quantile=clip_quantile,
            n_iter=n_iter,
            random_state=random_state,
            ot_mod=ot,
            embeds=embeds,
            store_adjacent_mapping=False,
            # Do NOT change slice physical size by default.
            allow_scale=False,
            spatial_weight=0.2,
            spatial_sigma=None,
            transport_unbalanced=True,
            ot_reg_m=10.0,
        )
        results[i] = res

    cube.write_back()

    if refresh_mapping:
        if embeds is None:
            embeds = _fit_global_svd_and_transform(
                cube,
                subsample_n=subsample_n,
                svd_dim=svd_dim,
                random_state=random_state,
                feature_mode=feature_mode,
                smooth_k=smooth_k,
                smooth_alpha=smooth_alpha,
                smooth_steps=smooth_steps,
            )
        _refresh_adjacent_map_to_prev_from_embedding(cube, embeds, n_neighbors=5)

    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(
            {
                "align_global_method": "anchor_ot_rigid",
                "align_global_params": {
                    "anchor_index": int(anchor_index),
                    "subsample_n": int(subsample_n),
                    "svd_dim": int(svd_dim),
                    "expr_knn": expr_knn,
                    "transport": str(transport),
                    "emd_max_n": int(emd_max_n),
                    "ot_reg": float(ot_reg),
                    "ot_method": str(ot_method),
                    "ot_num_iter_max": int(ot_num_iter_max),
                    "ot_stop_thr": float(ot_stop_thr),
                    "clip_quantile": clip_quantile,
                    "n_iter": int(n_iter),
                    "random_state": int(random_state),
                    "shared_embedding": bool(shared_embedding),
                    "refresh_mapping": bool(refresh_mapping),
                },
            }
        )

    return {"status": "ok", "anchor_index": int(anchor_index), "results": results}


def align_global_slices_ba_ot(
    cube: SpatioCube,
    *,
    subsample_n: int = 2000,
    svd_dim: int = 50,
    expr_knn: int | None = 50,
    transport: str = "emd",
    emd_max_n: int = 2000,
    ot_reg: float = 0.05,
    ot_method: str = "sinkhorn_stabilized",
    ot_num_iter_max: int = 20000,
    ot_stop_thr: float = 1e-9,
    clip_quantile: float | None = None,
    random_state: int = 0,
    anchor_index: int = 0,
    lam_smooth: float = 0.0,
    max_nfev: int = 50,
    base_key: str | None = None,
) -> dict[str, object]:
    """Global rigid bundle adjustment (pose graph) over all slices using expression OT correspondences.

    Motivation:
    - Pairwise chaining (sequential) accumulates small errors -> global drift / spiral.
    - This function solves a global least-squares problem over all slice poses
      (rotation + translation per slice) using only **adjacent-slice** correspondences.

    Model (2D rigid pose per slice i):
      x_global = R(theta_i) @ x_local + t_i

    We build soft correspondences between adjacent slices via expression OT on a subsample,
    then optimize all {theta_i, t_i} jointly with `scipy.optimize.least_squares`.

    Notes:
    - Gauge freedom is fixed by anchoring slice `anchor_index` to identity pose.
    - By default, coordinates are transformed from `base_key` if provided; otherwise uses
      `cube.spatial_raw_key` when available; else uses current `cube.spatial_key`.
    """

    n_slices = len(cube.adatas)
    if n_slices < 2:
        return {"status": "skipped", "reason": "n_slices<2"}

    rng = np.random.default_rng(int(random_state))
    requested_transport = str(transport).lower()
    ot: object | None = None

    if base_key is None:
        if cube.spatial_raw_key is not None and all(cube.spatial_raw_key in a.obsm for a in cube.adatas):
            base_key = cube.spatial_raw_key
        else:
            base_key = cube.spatial_key

    # Precompute correspondences for each adjacent pair k<->k+1.
    # Store: (k_hi, k_lo, src_pts, tgt_pts, w)
    corrs: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for k_lo in range(n_slices - 1):
        k_hi = k_lo + 1
        a_lo = cube.adatas[k_lo]
        a_hi = cube.adatas[k_hi]

        Xs = a_hi.X
        Xt = a_lo.X
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

        src_xy = np.asarray(a_hi.obsm[base_key], float)
        tgt_xy = np.asarray(a_lo.obsm[base_key], float)
        ns_full, nt_full = int(src_xy.shape[0]), int(tgt_xy.shape[0])
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

        if transport_mode in {"sinkhorn", "ot"} and ot is None:
            ot = _require_pot()

        # Cost in shared embedding
        if expr_knn is None or expr_knn <= 0 or expr_knn >= nt:
            Es2 = np.sum(Es_s * Es_s, axis=1, keepdims=True)
            Et2 = np.sum(Et_s * Et_s, axis=1, keepdims=True).T
            cross = Es_s @ Et_s.T
            C = np.maximum(Es2 + Et2 - 2.0 * cross, 0.0)
        else:
            nn = NearestNeighbors(n_neighbors=min(int(expr_knn), Et_s.shape[0])).fit(Et_s)
            knn_idx = nn.kneighbors(Es_s, return_distance=False)
            C = np.full((ns, nt), np.inf, dtype=float)
            for r in range(ns):
                cand = knn_idx[r]
                diff = Et_s[cand] - Es_s[r]
                C[r, cand] = np.sum(diff * diff, axis=1)

        # Optional clipping: keep only lower-cost entries per row.
        # Must be done carefully: aggressive clipping can wipe out all finite entries for a row.
        if clip_quantile is not None and 0.0 < float(clip_quantile) < 1.0:
            C0 = C
            thr = np.zeros((ns, 1), dtype=float)
            for r in range(ns):
                row = C0[r]
                row_f = row[np.isfinite(row)]
                if row_f.size == 0:
                    thr[r, 0] = np.inf
                else:
                    thr[r, 0] = float(np.quantile(row_f, float(clip_quantile)))
            C = np.where((C0 <= thr) & np.isfinite(C0), C0, np.inf)
            # Fallback: if clipping wipes everything, revert to unclipped costs.
            if not np.any(np.isfinite(C)):
                C = C0

        finite = np.isfinite(C)
        if not np.any(finite):
            continue
        cap_rows = np.zeros((ns, 1), dtype=float)
        for r in range(ns):
            row_f = C[r][np.isfinite(C[r])]
            mx = float(np.max(row_f)) if row_f.size else 1.0
            cap_rows[r, 0] = max(10.0 * mx, 1.0)
        C = np.where(np.isfinite(C), C, cap_rows)
        C = C - np.min(C, axis=1, keepdims=True)

        a = np.full((ns,), 1.0 / ns)
        b = np.full((nt,), 1.0 / nt)
        if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"}:
            if ns != nt:
                transport_mode = "sinkhorn"
        if transport_mode in {"emd", "lap", "linear_assignment", "linear_sum_assignment"}:
            r_ind, c_ind = linear_sum_assignment(C)
            Pi = np.zeros((ns, nt), dtype=float)
            Pi[r_ind, c_ind] = 1.0
            Pi = Pi * a[:, None]
        else:
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

        w = Pi.sum(axis=1)
        mapped = (Pi @ tgt_xy_s) / (w[:, None] + 1e-12)  # points in slice-lo local coords
        # Keep rows with non-trivial mass.
        # Note: under uniform marginals (common in Sinkhorn), w can be (almost) constant,
        # so quantile-based filtering may accidentally drop **all** rows.
        keep = w > 0.0
        if np.any(keep):
            corrs.append((k_hi, k_lo, src_xy_s[keep], mapped[keep], w[keep]))

    if len(corrs) == 0:
        return {"status": "failed", "reason": "no correspondences"}

    # Parameter vector: poses for all slices except anchor.
    # pose = (theta, tx, ty)
    anchor_index = int(anchor_index)
    if anchor_index < 0 or anchor_index >= n_slices:
        raise ValueError("`anchor_index` out of range.")

    def pack(poses: list[tuple[float, float, float]]) -> np.ndarray:
        out = []
        for i in range(n_slices):
            if i == anchor_index:
                continue
            out.extend(list(poses[i]))
        return np.asarray(out, dtype=float)

    def unpack(x: np.ndarray) -> list[tuple[float, float, float]]:
        poses = [(0.0, 0.0, 0.0) for _ in range(n_slices)]
        idx = 0
        for i in range(n_slices):
            if i == anchor_index:
                poses[i] = (0.0, 0.0, 0.0)
                continue
            poses[i] = (float(x[idx]), float(x[idx + 1]), float(x[idx + 2]))
            idx += 3
        return poses

    x0 = np.zeros(3 * (n_slices - 1), dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        poses = unpack(x)
        res = []
        for k_hi, k_lo, P_hi, Q_lo, ww in corrs:
            th_hi, tx_hi, ty_hi = poses[k_hi]
            th_lo, tx_lo, ty_lo = poses[k_lo]
            R_hi = _rotmat(th_hi)
            R_lo = _rotmat(th_lo)
            Ph = (P_hi @ R_hi.T) + np.array([tx_hi, ty_hi])[None, :]
            Ql = (Q_lo @ R_lo.T) + np.array([tx_lo, ty_lo])[None, :]
            d = Ph - Ql
            w_s = np.sqrt(np.maximum(ww, 0.0))[:, None]
            res.append((d * w_s).ravel())
        if lam_smooth > 0:
            # Encourage small inter-slice pose differences (optional)
            poses = unpack(x)
            for i in range(n_slices - 1):
                if i == anchor_index or (i + 1) == anchor_index:
                    continue
                th0, tx0_, ty0_ = poses[i]
                th1, tx1_, ty1_ = poses[i + 1]
                res.append(
                    (float(lam_smooth) * np.array([th1 - th0, tx1_ - tx0_, ty1_ - ty0_], float))
                )
        return np.concatenate(res, axis=0)

    sol = least_squares(residuals, x0, max_nfev=int(max_nfev))
    poses_opt = unpack(sol.x)

    # Apply optimized poses to full-resolution coords from base_key, write into cube.spatial_key
    for i, a in enumerate(cube.adatas):
        th, tx, ty = poses_opt[i]
        R = _rotmat(th)
        xy0 = np.asarray(a.obsm[base_key], float)
        a.obsm[cube.spatial_key] = (xy0 @ R.T) + np.array([tx, ty])[None, :]

    cube.write_back()

    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(
            {
                "align_global_method": "ba_ot_rigid",
                "align_global_params": {
                    "subsample_n": int(subsample_n),
                    "svd_dim": int(svd_dim),
                    "expr_knn": expr_knn,
                    "transport": str(transport),
                    "emd_max_n": int(emd_max_n),
                    "ot_reg": float(ot_reg),
                    "ot_method": str(ot_method),
                    "ot_num_iter_max": int(ot_num_iter_max),
                    "ot_stop_thr": float(ot_stop_thr),
                    "clip_quantile": clip_quantile,
                    "anchor_index": int(anchor_index),
                    "lam_smooth": float(lam_smooth),
                    "max_nfev": int(max_nfev),
                    "base_key": str(base_key),
                },
                "align_global_result": {
                    "success": bool(sol.success),
                    "cost": float(sol.cost),
                    "nfev": int(sol.nfev),
                    "status": int(sol.status),
                    "message": str(sol.message),
                },
            }
        )

    return {
        "status": "ok" if sol.success else "failed",
        "success": bool(sol.success),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "poses": poses_opt,
    }


def align_adjacent_slices_ot(
    cube: SpatioCube,
    *,
    lambda_z: float | None = None,
    subsample_n: int = 2000,
    paired_subsample: bool = True,
    svd_dim: int = 50,
    feature_mode: Literal["svd", "svd_smooth"] = "svd_smooth",
    smooth_k: int = 15,
    smooth_alpha: float = 0.7,
    smooth_steps: int = 2,
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
    strategy: Literal["sequential", "middle_out"] = "sequential",
    anchor_index: int | None = None,
    shared_embedding: bool = True,
) -> list[AlignResult]:
    """Coarse-to-fine rigid alignment (rotation + translation) using OT + Procrustes.

    **Slice order vs alignment**
    - `infer_slice_order` (optional) uses **global** pairwise expression distances + a best-path
      solver to order slices along the series.
    - This function aligns **one pair at a time** in that order. With ``strategy="sequential"``,
      each slice is chained into slice ``0``'s frame, so small per-pair errors can **accumulate**
      (often seen as a slow twist / drift in 3D).
    - With ``strategy="middle_out"``, alignment propagates from a central **anchor** slice toward
      both ends, which typically **reduces accumulated drift** for long series.

    Writes updated 2D coordinates to each slice's `obsm[spatial_key]` and refreshes `spatial_3d`.
    """

    if lambda_z is not None:
        cube.lambda_z = float(lambda_z)

    rng = np.random.default_rng(random_state)
    results: list[AlignResult] = []
    requested_transport = str(transport).lower()
    ot: object | None = None
    embeds: list[np.ndarray] | None = None

    if shared_embedding:
        embeds = _fit_global_svd_and_transform(
            cube,
            subsample_n=subsample_n,
            svd_dim=svd_dim,
            random_state=random_state,
            feature_mode=feature_mode,
            smooth_k=smooth_k,
            smooth_alpha=smooth_alpha,
            smooth_steps=smooth_steps,
        )

    pairs = _alignment_pairs(len(cube.adatas), strategy, anchor_index)
    for i_src, i_tgt in pairs:
        res, ot = _align_pair_ot(
            cube,
            i_src,
            i_tgt,
            rng=rng,
            subsample_n=subsample_n,
            paired_subsample=paired_subsample,
            svd_dim=svd_dim,
            expr_knn=expr_knn,
            requested_transport=requested_transport,
            emd_max_n=emd_max_n,
            ot_reg=ot_reg,
            ot_method=ot_method,
            ot_num_iter_max=ot_num_iter_max,
            ot_stop_thr=ot_stop_thr,
            clip_quantile=clip_quantile,
            n_iter=n_iter,
            random_state=random_state,
            ot_mod=ot,
            embeds=embeds,
            store_adjacent_mapping=True,
            # Do NOT change slice physical size by default.
            allow_scale=False,
            spatial_weight=0.2,
            spatial_sigma=None,
            transport_unbalanced=True,
            ot_reg_m=10.0,
        )
        results.append(res)

    meta = {
        "align_method": "coarse_adjacent_rigid",
        "align_params": {
            "strategy": strategy,
            "anchor_index": anchor_index,
                "shared_embedding": bool(shared_embedding),
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
    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(meta)
    return results
