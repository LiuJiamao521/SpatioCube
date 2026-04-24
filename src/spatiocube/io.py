from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


@dataclass(frozen=True)
class SpatialKeys:
    x: str = "coor_x_ad2"
    y: str = "coor_y_ad2"
    obsm_spatial: str = "spatial"


def _ensure_obsm_spatial(
    adata: ad.AnnData,
    *,
    spatial_keys: SpatialKeys = SpatialKeys(),
    overwrite: bool = False,
) -> None:
    """Ensure `adata.obsm['spatial']` exists, pulling from obs x/y columns if needed."""

    if (not overwrite) and (spatial_keys.obsm_spatial in adata.obsm):
        return

    missing = [c for c in (spatial_keys.x, spatial_keys.y) if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            "Cannot create `obsm['spatial']` because required obs columns are missing: "
            + ", ".join(missing)
        )

    xy = adata.obs[[spatial_keys.x, spatial_keys.y]].to_numpy(dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected 2D spatial coordinates with shape (n_obs, 2).")
    adata.obsm[spatial_keys.obsm_spatial] = xy


def read_merged_h5ad(
    path: str | Path | None = None,
    *,
    env_key: str = "SPATIOCUBE_MOUSEBRAIN_H5AD",
    spatial_keys: SpatialKeys = SpatialKeys(),
    ensure_spatial: bool = True,
) -> ad.AnnData:
    """Read a merged `.h5ad` file and ensure expected spatial keys.

    If `path` is None, this checks environment variable `env_key`.
    """

    if path is None:
        env_val = None
        try:
            import os

            env_val = os.environ.get(env_key)
        except Exception:
            env_val = None
        if not env_val:
            raise ValueError(
                f"`path` is required unless environment variable {env_key} is set."
            )
        path = env_val

    adata = sc.read_h5ad(str(path))
    if ensure_spatial:
        _ensure_obsm_spatial(adata, spatial_keys=spatial_keys, overwrite=False)
    return adata


def split_anndata_by_obs(
    adata: ad.AnnData,
    *,
    slice_key: str = "sampleid",
    sort: Literal["lexicographic", "none"] = "lexicographic",
) -> list[ad.AnnData]:
    """Split a merged AnnData into slices by `adata.obs[slice_key]`.

    Returns a list of AnnData views copied into independent objects.
    """

    if slice_key not in adata.obs.columns:
        raise KeyError(f"`slice_key='{slice_key}'` not found in `adata.obs`.")

    ser = adata.obs[slice_key].astype(str)
    groups = ser.groupby(ser).groups  # dict[slice_id, index]
    slice_ids = list(groups.keys())
    if sort == "lexicographic":
        slice_ids = sorted(slice_ids)
    elif sort == "none":
        pass
    else:
        raise ValueError("`sort` must be one of {'lexicographic','none'}.")

    out: list[ad.AnnData] = []
    for sid in slice_ids:
        idx = groups[sid]
        adata_i = adata[idx].copy()
        # Keep slice id in obs for downstream writing.
        adata_i.obs[slice_key] = pd.Categorical([sid] * adata_i.n_obs)
        out.append(adata_i)
    return out

