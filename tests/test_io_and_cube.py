import numpy as np
import pandas as pd
import anndata as ad

from spatiocube.io import SpatialKeys, _ensure_obsm_spatial, split_anndata_by_obs
from spatiocube.core import SpatioCube


def _toy_adata(n: int, p: int, sampleid: str):
    X = np.random.RandomState(0).randn(n, p).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sampleid": [sampleid] * n,
            "coor_x_ad2": np.linspace(0, 10, n),
            "coor_y_ad2": np.linspace(0, 5, n),
        },
        index=[f"{sampleid}-{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(p)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_ensure_obsm_spatial_from_obs():
    a = _toy_adata(5, 3, "T300")
    assert "spatial" not in a.obsm
    _ensure_obsm_spatial(a, spatial_keys=SpatialKeys(), overwrite=False)
    assert "spatial" in a.obsm
    assert a.obsm["spatial"].shape == (5, 2)


def test_split_anndata_by_sampleid():
    a0 = _toy_adata(4, 3, "T300")
    a1 = _toy_adata(6, 3, "T301")
    merged = ad.concat([a0, a1], axis=0, merge="same")
    _ensure_obsm_spatial(merged, spatial_keys=SpatialKeys(), overwrite=False)

    slices = split_anndata_by_obs(merged, slice_key="sampleid")
    assert len(slices) == 2
    assert all("spatial" in s.obsm for s in slices)


def test_spatiocube_write_back_spatial3d():
    a0 = _toy_adata(4, 3, "T300")
    a1 = _toy_adata(6, 3, "T301")
    for a in (a0, a1):
        _ensure_obsm_spatial(a, spatial_keys=SpatialKeys(), overwrite=False)
    cube = SpatioCube.from_adata_list([a0, a1], z_positions=np.array([0.0, 1.0]))
    cube.write_back()
    assert all("spatial_3d" in a.obsm for a in cube.adatas)
    assert cube.adatas[0].obsm["spatial_3d"].shape[1] == 3


def test_spatiocube_z_spacing_and_spatial_raw_snapshot():
    a0 = _toy_adata(4, 3, "T300")
    a1 = _toy_adata(6, 3, "T301")
    merged = ad.concat([a0, a1], axis=0, merge="same")
    _ensure_obsm_spatial(merged, spatial_keys=SpatialKeys(), overwrite=False)

    cube = SpatioCube.from_merged_h5ad(
        merged,
        slice_key="sampleid",
        order_mode="obs",
        z_spacing=3.0,
        z_base=2.0,
        spatial_raw_key="spatial_xy_raw",
    )
    assert np.allclose(cube.z_positions, np.array([2.0, 5.0], dtype=float))
    for a in cube.adatas:
        assert "spatial_xy_raw" in a.obsm
        assert np.allclose(a.obsm["spatial_xy_raw"], a.obsm["spatial"])


def test_align_adjacent_slices_ot_recover_known_rigid_transform():
    from spatiocube.align import align_adjacent_slices_ot

    rng = np.random.RandomState(0)
    n = 40
    p = 12
    # Build an expression matrix where rows are identical across the two slices, but each row
    # is unique (helps OT / kNN identify the correct correspondence even if XY is misaligned).
    xy0 = rng.randn(n, 2) * 5.0
    tail = rng.randn(n, p - 2).astype(np.float32)
    tiebreak = (1e-6 * np.arange(n, dtype=np.float32))[:, None]
    X = np.concatenate([(0.1 * xy0).astype(np.float32), tail], axis=1) + tiebreak

    theta = np.deg2rad(37.0)
    R_true = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    t_true = np.array([12.3, -4.5], dtype=float)

    xy1 = (xy0 @ R_true.T) + t_true[None, :]

    obs0 = pd.DataFrame({"sampleid": ["S0"] * n})
    obs1 = pd.DataFrame({"sampleid": ["S1"] * n})
    var = pd.DataFrame(index=[f"g{i}" for i in range(p)])
    a0 = ad.AnnData(X=X.copy(), obs=obs0, var=var)
    a1 = ad.AnnData(X=X.copy(), obs=obs1, var=var)
    a0.obsm["spatial"] = np.asarray(xy0, dtype=float)
    a1.obsm["spatial"] = np.asarray(xy1, dtype=float)

    cube = SpatioCube.from_adata_list(
        [a0, a1],
        z_positions=np.array([0.0, 1.0]),
        spatial_raw_key=None,
    )

    align_adjacent_slices_ot(
        cube,
        subsample_n=min(2000, n),
        svd_dim=min(10, p - 1),
        expr_knn=None,
        transport="emd",
        n_iter=3,
        random_state=0,
    )

    xy1_hat = np.asarray(cube.adatas[1].obsm["spatial"], dtype=float)
    err = np.max(np.abs(xy1_hat - xy0))
    assert err < 1e-2

