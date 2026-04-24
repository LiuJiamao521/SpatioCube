from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np

from ._types import SliceInfo
from .io import SpatialKeys, _ensure_obsm_spatial, split_anndata_by_obs
from .order import OrderConfig, infer_slice_order


@dataclass
class SpatioCube:
    """Container for multi-slice spatial transcriptomics in a shared 3D frame."""

    adatas: list[ad.AnnData]
    z_positions: np.ndarray  # shape (n_slices,)
    slice_key: str = "sampleid"
    lambda_z: float = 0.01
    spatial_key: str = "spatial"
    spatial_3d_key: str = "spatial_3d"
    cluster_key: str = "SpatioCube_cluster"
    spatial_raw_key: str | None = "spatial_xy_raw"

    def __post_init__(self) -> None:
        if len(self.adatas) == 0:
            raise ValueError("`adatas` must be non-empty.")
        if self.z_positions.shape != (len(self.adatas),):
            raise ValueError("`z_positions` must have shape (n_slices,).")
        if self.lambda_z <= 0:
            raise ValueError("`lambda_z` must be positive.")

        for a in self.adatas:
            if self.spatial_key not in a.obsm:
                raise KeyError(f"Missing `adata.obsm['{self.spatial_key}']` in a slice.")
            xy = np.asarray(a.obsm[self.spatial_key])
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(
                    f"`adata.obsm['{self.spatial_key}']` must have shape (n_obs, 2)."
                )

    def snapshot_spatial_xy_raw(self, *, key: str | None = None) -> None:
        """Store a per-slice copy of the current 2D coordinates before alignment.

        This is written to `adata.obsm[key]` (default: `spatial_raw_key` on the cube).
        """

        raw_key = self.spatial_raw_key if key is None else key
        if raw_key is None:
            return
        for a in self.adatas:
            if raw_key in a.obsm:
                continue
            xy = np.asarray(a.obsm[self.spatial_key], dtype=float)
            a.obsm[raw_key] = np.array(xy, copy=True)

    @classmethod
    def from_merged_h5ad(
        cls,
        adata: ad.AnnData,
        *,
        slice_key: str = "sampleid",
        z_positions: np.ndarray | None = None,
        z_spacing: float = 1.0,
        z_base: float = 0.0,
        lambda_z: float = 0.01,
        source_path: str | None = None,
        spatial_keys: SpatialKeys = SpatialKeys(),
        ensure_spatial: bool = True,
        sort_slices: str = "lexicographic",
        order_mode: str = "obs",
        order_config: OrderConfig = OrderConfig(),
        spatial_raw_key: str | None = "spatial_xy_raw",
    ) -> "SpatioCube":
        if ensure_spatial:
            _ensure_obsm_spatial(adata, spatial_keys=spatial_keys, overwrite=False)
        adatas = split_anndata_by_obs(adata, slice_key=slice_key, sort=sort_slices)  # copies
        if order_mode == "infer":
            perm = infer_slice_order(adatas, config=order_config)
            adatas = [adatas[i] for i in perm]
        elif order_mode == "obs":
            pass
        else:
            raise ValueError("`order_mode` must be one of {'obs','infer'}.")
        if z_positions is None:
            n = len(adatas)
            z_positions = float(z_base) + float(z_spacing) * np.arange(n, dtype=float)
        z_positions = np.asarray(z_positions, dtype=float)
        cube = cls(
            adatas=adatas,
            z_positions=z_positions,
            slice_key=slice_key,
            lambda_z=lambda_z,
            spatial_key=spatial_keys.obsm_spatial,
            spatial_raw_key=spatial_raw_key,
        )
        cube.snapshot_spatial_xy_raw()
        cube._write_uns_metadata(
            source="from_merged_h5ad",
            extra={
                "slice_key": slice_key,
                "source_path": source_path,
                "z_spacing": float(z_spacing),
                "z_base": float(z_base),
                "spatial_raw_key": spatial_raw_key,
            },
        )
        return cube

    @classmethod
    def from_adata_list(
        cls,
        adatas: list[ad.AnnData],
        *,
        z_positions: np.ndarray,
        slice_key: str = "sampleid",
        lambda_z: float = 0.01,
        spatial_key: str = "spatial",
        spatial_raw_key: str | None = "spatial_xy_raw",
    ) -> "SpatioCube":
        cube = cls(
            adatas=[a.copy() for a in adatas],
            z_positions=np.asarray(z_positions, dtype=float),
            slice_key=slice_key,
            lambda_z=lambda_z,
            spatial_key=spatial_key,
            spatial_raw_key=spatial_raw_key,
        )
        cube.snapshot_spatial_xy_raw()
        cube._write_uns_metadata(
            source="from_adata_list",
            extra={"slice_key": slice_key, "spatial_raw_key": spatial_raw_key},
        )
        return cube

    def slice_infos(self) -> list[SliceInfo]:
        infos: list[SliceInfo] = []
        for a, z in zip(self.adatas, self.z_positions):
            key = "unknown"
            if self.slice_key in a.obs.columns:
                vals = a.obs[self.slice_key].astype(str).unique()
                if len(vals) == 1:
                    key = vals[0]
            infos.append(SliceInfo(key=key, z=float(z), n_obs=int(a.n_obs)))
        return infos

    def write_back(self) -> None:
        """Ensure `spatial_3d` exists (xy + z) in each AnnData."""
        for a, z in zip(self.adatas, self.z_positions):
            xy = np.asarray(a.obsm[self.spatial_key], dtype=float)
            zcol = np.full((xy.shape[0], 1), float(z), dtype=float)
            a.obsm[self.spatial_3d_key] = np.concatenate([xy, zcol], axis=1)

    def set_clusters(self, labels: np.ndarray) -> None:
        """Write global labels back into each slice's `obs[cluster_key]`."""
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("`labels` must be a 1D array.")
        if labels.shape[0] != sum(a.n_obs for a in self.adatas):
            raise ValueError("`labels` length must equal total number of spots.")

        offset = 0
        for a in self.adatas:
            n = int(a.n_obs)
            a.obs[self.cluster_key] = labels[offset : offset + n]
            offset += n

    def scaled_spatial_3d(self, slice_idx: int) -> np.ndarray:
        """Return 3D coords with z scaled by sqrt(lambda_z) to match metric.

        Under metric d^2 = ||dx||^2 + lambda_z * dz^2, scaling z by sqrt(lambda_z)
        makes Euclidean distance equivalent.
        """

        a = self.adatas[slice_idx]
        if self.spatial_3d_key not in a.obsm:
            self.write_back()
        xyz = np.asarray(a.obsm[self.spatial_3d_key], dtype=float)
        out = xyz.copy()
        out[:, 2] *= float(np.sqrt(self.lambda_z))
        return out

    def _write_uns_metadata(self, *, source: str, extra: dict[str, Any] | None = None) -> None:
        meta: dict[str, Any] = {
            "source": source,
            "slice_key": self.slice_key,
            "lambda_z": float(self.lambda_z),
            "spatial_key": self.spatial_key,
            "spatial_raw_key": self.spatial_raw_key,
            "spatial_3d_key": self.spatial_3d_key,
            "cluster_key": self.cluster_key,
            "n_slices": len(self.adatas),
            "z_positions": np.asarray(self.z_positions, dtype=float).tolist(),
            "slice_order": [i.key for i in self.slice_infos()],
        }
        if extra:
            meta.update(extra)
        # Write to each slice for now; later we can keep a separate container-level state.
        for a in self.adatas:
            a.uns.setdefault("SpatioCube", {}).update(meta)

