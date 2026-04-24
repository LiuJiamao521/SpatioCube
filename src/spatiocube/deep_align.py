from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .align import _procrustes_rigid_2d, _require_pot, _subsample_idx
from .core import SpatioCube


def jade_like_align_adjacent(
    cube: SpatioCube,
    *,
    subsample_n: int = 2000,
    svd_dim: int = 50,
    ot_reg: float = 0.05,
    roundtrip_steps: int = 5,
    n_iter: int = 2,
    random_state: int = 0,
) -> None:
    """A lightweight JADE-inspired roundtrip alignment (no neural nets).

    This is **not** a full JADE reimplementation. It keeps the core idea:
    alternate between (soft) alignment in an embedding space and refining
    the embedding via roundtrip barycentric updates.
    """

    ot = _require_pot()
    rng = np.random.default_rng(random_state)

    for i in range(len(cube.adatas) - 1):
        src = cube.adatas[i + 1]
        tgt = cube.adatas[i]

        Xs = src.X
        Xt = tgt.X
        n_comp = int(min(svd_dim, Xs.shape[1] - 1, Xt.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        from scipy.sparse import vstack as sp_vstack

        try:
            Xcat = sp_vstack([Xs, Xt])
        except Exception:
            Xcat = np.vstack([np.asarray(Xs), np.asarray(Xt)])
        Ecat = svd.fit_transform(Xcat)
        Es = Ecat[: Xs.shape[0]]
        Et = Ecat[Xs.shape[0] :]

        src_xy = np.asarray(src.obsm[cube.spatial_key], float)
        tgt_xy = np.asarray(tgt.obsm[cube.spatial_key], float)

        isrc = _subsample_idx(src_xy.shape[0], subsample_n, rng=rng)
        itgt = _subsample_idx(tgt_xy.shape[0], subsample_n, rng=rng)

        Es_s = Es[isrc]
        Et_s = Et[itgt]
        src_xy_s = src_xy[isrc]
        tgt_xy_s = tgt_xy[itgt]

        a = np.full((Es_s.shape[0],), 1.0 / Es_s.shape[0])
        b = np.full((Et_s.shape[0],), 1.0 / Et_s.shape[0])

        R = np.eye(2)
        t = np.zeros(2)

        for _ in range(n_iter):
            # Roundtrip embedding refinement
            Hs = Es_s.copy()
            Ht = Et_s.copy()
            for _rt in range(roundtrip_steps):
                # attention-like similarity; stable softmax per-row
                logits = -(Hs @ Ht.T) / np.sqrt(Hs.shape[1])
                logits = logits - logits.max(axis=1, keepdims=True)
                C = np.exp(logits)
                C = C / (C.sum(axis=1, keepdims=True) + 1e-12)

                # Sinkhorn to get doubly-stochastic plan
                Pi = ot.sinkhorn(a, b, -np.log(C + 1e-12), reg=ot_reg)

                Hs = Pi @ Ht
                Ht = Pi.T @ Hs

            # Use final Pi to compute barycentric mapping in XY
            Pi = ot.sinkhorn(a, b, -np.log(C + 1e-12), reg=ot_reg)
            mapped = Pi @ tgt_xy_s
            w = Pi.sum(axis=1)
            R, t = _procrustes_rigid_2d(src_xy_s, mapped, weights=w)
            src_xy_s = (src_xy_s @ R.T) + t[None, :]

        src_xy_full = (src_xy @ R.T) + t[None, :]
        src.obsm[cube.spatial_key] = src_xy_full

    cube.write_back()
    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(
            {
                "align_method": "jade_like_roundtrip",
                "align_params": {
                    "subsample_n": subsample_n,
                    "svd_dim": svd_dim,
                    "ot_reg": ot_reg,
                    "roundtrip_steps": roundtrip_steps,
                    "n_iter": n_iter,
                    "random_state": random_state,
                },
            }
        )

