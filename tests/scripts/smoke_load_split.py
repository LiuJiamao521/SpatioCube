from __future__ import annotations

import os

import spatiocube as scb


def main() -> None:
    # Either pass explicit path, or set environment variable:
    # export SPATIOCUBE_MOUSEBRAIN_H5AD=/cluster3/labData/jiamao/MouseBrain/xxx.h5ad
    print("SPATIOCUBE_MOUSEBRAIN_H5AD =", os.environ.get("SPATIOCUBE_MOUSEBRAIN_H5AD"))
    adata = scb.read_merged_h5ad()
    cube = scb.SpatioCube.from_merged_h5ad(
        adata,
        slice_key="sampleid",
        lambda_z=0.01,
        # Separate slices along z for 3D visualization (tune to your physical spacing / aesthetics).
        z_spacing=50.0,
    )
    print("n_slices =", len(cube.adatas))
    print("slice_infos =", cube.slice_infos()[:3])


if __name__ == "__main__":
    main()

