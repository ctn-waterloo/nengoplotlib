"""Decoded-connection vector / phase-portrait plots.

For a connection ``pre -> post`` where ``post.dimensions == pre.dimensions``
(typically a recurrent connection on ``pre``), this samples the input space
of ``pre`` on a regular grid, evaluates the connection's decoded output at
every grid point, and plots the displacement ``output - input`` as a vector
field. That field is the right-hand side of the discrete-time dynamics the
connection induces (modulo synapse + transform), so for a recurrent
connection it's the network's phase portrait.

Supports 1D, 2D, and 3D ``pre`` dimensions. Higher-dimensional ensembles
need a projection (e.g. PCA) that's outside the scope of this function.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _grid_inputs(dims, S, bounds):
    """Build a (n_pts, dims) grid of evaluation points and the meshgrid axes."""
    lo, hi = bounds
    axis = np.linspace(lo, hi, S)
    grids = np.meshgrid(*([axis] * dims), indexing="xy")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    return pts, grids


def plot_phase_portrait(
    pre,
    conn,
    sim=None,
    network=None,
    S: int = 21,
    bounds=(-1.0, 1.0),
    plot_type: str = "quiver",
    cmap: str = "viridis",
    scale: Optional[float] = None,
    density: float = 1.5,
    ax=None,
    figsize=(4, 4),
    dpi=None,
):
    """Plot the decoded vector field of a connection from ``pre``.

    Parameters
    ----------
    pre : nengo.Ensemble
        Source population whose tuning curves drive the decoded output.
    conn : nengo.Connection
        Connection from *pre*. ``conn.size_out`` must equal
        ``pre.dimensions``; usually this is a recurrent connection.
    sim : nengo.Simulator, optional
        Built simulator. If None, *network* must be given and a temporary
        simulator is built (no time stepping required).
    network : nengo.Network, optional
        Used only when *sim* is None.
    S : int
        Grid resolution per dimension. Total grid points is ``S**dims``.
    bounds : (float, float)
        Input-space bounds along every dimension.
    plot_type : {'quiver', 'stream'}
        ``'stream'`` is 2D-only. 1D and 3D always use quiver-style arrows.
    cmap : str or Colormap
        Used to color arrows / streamlines by velocity magnitude.
    scale : float, optional
        Quiver scale (smaller -> longer arrows). Auto if None.
    density : float
        Streamplot density. Ignored for ``plot_type='quiver'``.

    ax : matplotlib axes, optional
        Existing axes to draw into. For 3D, must be a 3D axes.
    figsize, dpi : matplotlib figure args, used when *ax* is None.

    Returns
    -------
    (fig, ax, field) : tuple
        *field* is the underlying quiver / streamplot artist (or list of
        Line3DCollection-like objects for 3D), useful for restyling.
    """
    import nengo  # local: keep nengo as an optional dep of the package

    dims = int(pre.dimensions)
    if dims not in (1, 2, 3):
        raise ValueError(
            f"plot_phase_portrait supports pre.dimensions in {{1, 2, 3}}; "
            f"got {dims}. Project to a lower-dim subspace first."
        )
    if conn.size_out != dims:
        raise ValueError(
            f"conn.size_out ({conn.size_out}) must equal pre.dimensions ({dims}) "
            "so the decoded output can be plotted in pre's input space."
        )
    if plot_type == "stream" and dims != 2:
        raise ValueError("plot_type='stream' is only supported for 2D ensembles")
    if plot_type not in ("quiver", "stream"):
        raise ValueError(f"plot_type must be 'quiver' or 'stream'; got {plot_type!r}")

    if sim is None:
        if network is None:
            raise ValueError("plot_phase_portrait needs either sim or network")
        sim = nengo.Simulator(network, progress_bar=False)

    pts, grids = _grid_inputs(dims, S, bounds)
    _, activity = nengo.utils.ensemble.tuning_curves(pre, sim, inputs=pts)

    decoder = sim.data[conn].weights
    if decoder.ndim == 1:  # 1D output edge case
        decoder = decoder[None, :]
    output = decoder @ activity.T  # (dims, S**dims)

    # displacement field
    deltas = [
        output[d].reshape([S] * dims) - grids[d]
        for d in range(dims)
    ]
    speed = np.sqrt(sum(d * d for d in deltas))

    # ---- figure / axes ----
    if ax is None:
        if dims == 3:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    field = None

    if dims == 1:
        x = grids[0]
        dx = deltas[0]
        zeros = np.zeros_like(x)
        field = ax.quiver(
            x, zeros, dx, zeros,
            speed, cmap=cmap, scale=scale,
            angles="xy", scale_units="xy",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("dx")
        # Also draw the dx-vs-x curve for context
        ax.plot(x, dx, color="0.4", lw=0.8, alpha=0.6, zorder=0)
        ax.set_ylim(min(0, dx.min()) - 0.1, max(0, dx.max()) + 0.1)
    elif dims == 2:
        xx, yy = grids
        dx, dy = deltas
        if plot_type == "quiver":
            field = ax.quiver(
                xx, yy, dx, dy, speed,
                cmap=cmap, scale=scale,
            )
        else:
            field = ax.streamplot(
                xx, yy, dx, dy,
                color=speed, cmap=cmap, density=density,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")
    else:  # dims == 3
        xx, yy, zz = grids
        dx, dy, dz = deltas
        # mpl's 3D quiver accepts a flat color array per arrow via `colors=`,
        # but it's clunky; we color by speed via a ScalarMappable explicitly.
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=speed.min(), vmax=speed.max())
        smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        # each arrow segment needs two endpoints in 3D quiver; rgba per arrow
        colors = smap.to_rgba(speed.ravel())
        field = ax.quiver(
            xx, yy, zz, dx, dy, dz,
            length=1.0, normalize=False, colors=colors,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # if show_axes and dims <= 2:
    #     ax.axhline(0, c="k", alpha=0.2, lw=0.8)
    #     ax.axvline(0, c="k", alpha=0.2, lw=0.8)

    return fig, ax, field
