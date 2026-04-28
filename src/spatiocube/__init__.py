from __future__ import annotations

from .core import SpatioCube
from .cluster3d import cluster_3d_diffusion_gmm
from .contrastive import contrastive_embed_3d
from .deep_align import jade_like_align_adjacent
from .graph import build_3d_adjacency, build_3d_weighted_graph, leiden_cluster
from .metrics import chamfer_distance
from .order import infer_slice_order, OrderConfig
from .viz import plotly_pointcloud
from .io import read_merged_h5ad, split_anndata_by_obs
from .align import align_global_slices_ba_ot
from .align import align_slices_to_anchor_ot
from .sparkx import run_sparkx, SparkXResult

__all__ = [
    "SpatioCube",
    "read_merged_h5ad",
    "split_anndata_by_obs",
    "build_3d_adjacency",
    "build_3d_weighted_graph",
    "leiden_cluster",
    "cluster_3d_diffusion_gmm",
    "contrastive_embed_3d",
    "jade_like_align_adjacent",
    "align_global_slices_ba_ot",
    "align_slices_to_anchor_ot",
    "run_sparkx",
    "SparkXResult",
    "chamfer_distance",
    "plotly_pointcloud",
    "infer_slice_order",
    "OrderConfig",
    "__version__",
]

__version__ = "0.0.0"

