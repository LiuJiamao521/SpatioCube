from __future__ import annotations

from typing import Any

import numpy as np


def plotly_pointcloud(
    xyz: np.ndarray,
    *,
    color: np.ndarray | None = None,
    size: float = 2.0,
    title: str = "SpatioCube 3D",
    **kwargs: Any,
):
    """Plot a 3D point cloud with Plotly (optional dependency)."""

    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Plotly is required. Install with `pip install .[viz]`.") from e

    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("`xyz` must be shape (n, 3).")

    marker: dict[str, Any] = {"size": size}
    if color is not None:
        marker["color"] = np.asarray(color)
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=marker,
                **kwargs,
            )
        ]
    )
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    return fig


def pyvista_pointcloud(
    xyz: np.ndarray,
    *,
    scalars: np.ndarray | None = None,
    point_size: float = 3.0,
    **kwargs: Any,
):
    """Render a 3D point cloud using PyVista (optional dependency)."""

    try:
        import pyvista as pv  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyVista is required. Install with `pip install .[viz]`.") from e

    xyz = np.asarray(xyz, float)
    cloud = pv.PolyData(xyz)
    if scalars is not None:
        cloud["scalars"] = np.asarray(scalars)

    p = pv.Plotter(**kwargs)
    p.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size, scalars="scalars")
    return p

