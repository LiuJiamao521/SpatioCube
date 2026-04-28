from __future__ import annotations

from typing import Any

import numpy as np

# Okabe–Ito colorblind-safe palette (no red-green pairs; black last for optional use)
_OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#332288",  # dark purple (substitute for black on dark backgrounds)
)


def okabe_ito_colors(n_labels: int) -> list[str]:
    """Return n hex colors by cycling Okabe–Ito (colorblind-friendly)."""

    n = int(n_labels)
    if n <= 0:
        return []
    base = list(_OKABE_ITO)
    return [base[i % len(base)] for i in range(n)]


def plotly_pointcloud(
    xyz: np.ndarray,
    *,
    color: np.ndarray | None = None,
    size: float = 2.0,
    title: str = "SpatioCube 3D",
    discrete_colors: bool = True,
    **kwargs: Any,
):
    """Plot a 3D point cloud with Plotly (optional dependency).

    When ``color`` encodes categorical labels (integers or few unique strings) and
    ``discrete_colors=True`` (default), uses **Okabe–Ito** discrete colors and one trace
    per category so the legend shows class names (no Viridis gradient).
    """

    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Plotly is required. Install with `pip install .[viz]`.") from e

    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("`xyz` must be shape (n, 3).")

    if color is None:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker={"size": size},
                    **kwargs,
                )
            ]
        )
        fig.update_layout(title=title, scene=dict(aspectmode="data"))
        return fig

    c = np.asarray(color)
    use_discrete = bool(discrete_colors)
    if use_discrete:
        if c.dtype.kind in "iuf" and np.unique(c).size > 48:
            use_discrete = False
        elif c.dtype.kind not in "iuf" and np.unique(c.astype(str)).size > 48:
            use_discrete = False

    if not use_discrete:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker={
                        "size": size,
                        "color": c,
                        "colorscale": "Viridis",
                        "showscale": True,
                    },
                    **kwargs,
                )
            ]
        )
        fig.update_layout(title=title, scene=dict(aspectmode="data"))
        return fig

    # Discrete: one trace per label for clear legend
    if c.dtype.kind in "iuf":
        lab = c.astype(int, copy=False).ravel()
    else:
        s = c.astype(str).ravel()
        uniq = np.unique(s)
        mapping = {u: i for i, u in enumerate(uniq)}
        lab = np.array([mapping[str(x)] for x in s], dtype=int)

    traces = []
    u = np.unique(lab)
    palette = okabe_ito_colors(len(u))
    for i, k in enumerate(u):
        m = lab == k
        traces.append(
            go.Scatter3d(
                x=xyz[m, 0],
                y=xyz[m, 1],
                z=xyz[m, 2],
                mode="markers",
                name=str(int(k)),
                marker={"size": size, "color": palette[i]},
                legendgroup=str(int(k)),
                showlegend=True,
                **kwargs,
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        legend=dict(itemsizing="constant", traceorder="normal"),
    )
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
