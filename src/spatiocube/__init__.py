from __future__ import annotations

from .core import SpatioCube
from .contrastive import contrastive_embed_3d
from .deep_align import jade_like_align_adjacent
from .graph import build_3d_adjacency, leiden_cluster
from .metrics import chamfer_distance
from .order import infer_slice_order, OrderConfig
from .viz import plotly_pointcloud
from .io import read_merged_h5ad, split_anndata_by_obs

__all__ = [
    "SpatioCube",
    "read_merged_h5ad",
    "split_anndata_by_obs",
    "build_3d_adjacency",
    "leiden_cluster",
    "contrastive_embed_3d",
    "jade_like_align_adjacent",
    "chamfer_distance",
    "plotly_pointcloud",
    "infer_slice_order",
    "OrderConfig",
    "__version__",
]

__version__ = "0.0.0"

