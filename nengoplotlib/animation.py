"""Animated spike / activity visualizations.

Two flavors:

  * :func:`plot_scrolling_raster` -- a raster whose x-axis scrolls forward
    in time. Recent spikes are colored by their synapse-filtered amplitude.
  * :func:`plot_grid_animation` -- color each cell in a 1D / 2D layout by the
    instantaneous filtered activity of the corresponding neuron(s). Hex or
    rect grid; positions come from a :class:`NeuronSorter` or are passed in
    directly.

Both return a :class:`matplotlib.animation.FuncAnimation`; call ``.save(...)``
or ``.to_jshtml()`` on the result.
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, RegularPolygon
from scipy.signal import lfilter


def _lowpass(X, dt, tau):
    """One-pole IIR lowpass along axis 0. Matches ``nengo.synapses.Lowpass``."""
    if tau is None or tau <= 0:
        return X
    alpha = dt / (dt + tau)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], X, axis=0)


def plot_scrolling_raster(
    t,
    spikes,
    plot_step=100,
    window=1.0,
    lookahead=0.0,
    tau=0.01,
    cmap="Greys",
    interval=200,
    dpi=200,
    figsize=(5, 2),
    ax=None,
):
    """Scrolling-window spike raster, animated.

    Parameters
    ----------
    t : (n_time,) array
    spikes : (n_time, n_neurons) array
    plot_step : int
        Number of timesteps advanced per animation frame.
    window : float
        Width of the visible x-axis window, seconds.
    lookahead : float
        Extra seconds of empty axis kept to the right of the latest spike.
    tau : float, optional
        Synapse time constant (seconds) used to color each spike by its
        filtered amplitude. ``None`` disables filtering (uniform black).
    cmap : str or Colormap
    interval : int
        Milliseconds between frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    t = np.asarray(t)
    spikes = np.asarray(spikes)
    dt = (t[-1] - t[0]) / (len(t) - 1)

    filtered = _lowpass(spikes, dt, tau) if tau else spikes

    norm = mpl.colors.Normalize(vmin=filtered.min(), vmax=filtered.max())
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron #")
    scat = ax.scatter([], [], marker=2, color="k", s=0.5)

    n_per_window = int((window - dt) / dt)
    n_frames = spikes.shape[0] // plot_step

    def init():
        ax.set_xlim(0, window)
        ax.set_ylim(0, spikes.shape[1])
        return (scat,)

    def update(frame):
        idx = frame * plot_step
        if idx >= len(t):
            return (scat,)
        now = t[idx]
        if now < window - lookahead:
            rel_time_idx, neuron_idx = np.where(spikes[:idx, :] > 0)
            abs_time_idx = rel_time_idx
        else:
            start = max(0, idx - n_per_window + int(lookahead / dt))
            rel_time_idx, neuron_idx = np.where(spikes[start:idx, :] > 0)
            abs_time_idx = rel_time_idx + start
            ax.set_xlim(t[start], t[idx] + lookahead)
        scat.set_offsets(np.column_stack((t[abs_time_idx], neuron_idx)))
        scat.set_facecolor(smap.to_rgba(filtered[abs_time_idx, neuron_idx]))
        return (scat,)

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=False, repeat=False,
    )
    fig.tight_layout()
    return ani


def plot_grid_animation(
    t,
    X,
    positions=None,
    sorter=None,
    patches=None,
    topology="hex",
    plot_step=100,
    tau=0.01,
    cmap="Greys",
    interval=200,
    dpi=200,
    figsize=(3, 3),
    edgecolor=None,
    linewidth=None,
    vmin=None,
    vmax=None,
    ax=None,
):
    """Animate per-neuron activity on a 2D layout.

    Three ways to specify the layout, in order of precedence:

    1. *sorter* with a ``.patches`` attribute (Voronoi parcellation): the
       caller-supplied patches are recolored every frame. Their natural
       shape is preserved.
    2. *patches* explicitly: a list of ``matplotlib.patches.Patch`` aligned
       with the columns of (post-transform) *X*.
    3. *positions* + *topology*: hex or rect grid built on the fly.

    Parameters
    ----------
    t : (n_time,) array
    X : (n_time, n_neurons) array
        Spike-like or filtered activity. When *sorter* is given, *X* is the
        raw / unsorted matrix -- it's passed through ``sorter.transform``.
    positions : (n_cells, 2) array, optional
        Cell centers. Required when *sorter* is None and *patches* is None.
    sorter : NeuronSorter, optional
        Replaces *positions* and pre-merges *X*. If the sorter exposes a
        ``patches`` list (Voronoi methods), those patches are used.
    patches : list of matplotlib.patches.Patch, optional
        Per-cell patches. ``None`` entries are dropped along with their
        corresponding columns of *X* / rows of *positions*.
    topology : {'hex', 'rect'}
        Grid shape when patches are auto-built.
    plot_step : int
    tau : float, optional
    cmap : str or Colormap
    edgecolor, linewidth : optional
        Override the default patch outline. Defaults are matched to the
        topology (none for hex/rect; soft cream edge for custom patches).
    vmin, vmax : float, optional

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    t = np.asarray(t)
    X = np.asarray(X, dtype=float)

    sorter_patches = None
    if sorter is not None:
        X = sorter.transform(X)
        positions = sorter.merged_positions
        if positions.shape[1] == 1:
            positions = np.column_stack([positions[:, 0], np.zeros(len(positions))])
        if sorter.som is not None and sorter.som.topology in ("hex", "rect"):
            topology = sorter.som.topology
        sorter_patches = getattr(sorter, "patches", None)

    if patches is None and sorter_patches is not None:
        patches = sorter_patches

    if patches is not None:
        # Drop None entries (degenerate cells) and align X / positions.
        valid = [i for i, p in enumerate(patches) if p is not None]
        patches = [patches[i] for i in valid]
        X = X[:, valid]
        if positions is not None:
            positions = np.asarray(positions, dtype=float)[valid]
        topology = "custom"

    if positions is None and patches is None:
        raise ValueError(
            "plot_grid_animation needs either positions, sorter, or patches"
        )
    if positions is not None:
        positions = np.asarray(positions, dtype=float)

    dt = (t[-1] - t[0]) / (len(t) - 1)
    filtered = _lowpass(X, dt, tau) if tau else X

    if vmin is None:
        vmin = filtered.min()
    if vmax is None:
        vmax = filtered.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if topology == "custom":
        # Use the supplied patches; size axes to their union bbox.
        all_xy = np.vstack([p.get_xy() for p in patches])
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())
    else:
        radius = 1.0 / np.sqrt(3) if topology == "hex" else 0.5
        pad = radius
        ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
        ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")

    if topology == "custom":
        pass  # use supplied patches as-is
    elif topology == "hex":
        patches = [
            RegularPolygon(
                p, numVertices=6, radius=radius,
                orientation=0.0, facecolor="white", edgecolor="white", lw=0,
            )
            for p in positions
        ]
    elif topology == "rect":
        patches = [
            Rectangle(
                (p[0] - radius, p[1] - radius), 2 * radius, 2 * radius,
                facecolor="white", edgecolor="white", lw=0,
            )
            for p in positions
        ]
    else:
        raise ValueError(f"unknown topology {topology!r}")

    if edgecolor is None:
        edgecolor = "#ece5d6" if topology == "custom" else "white"
    if linewidth is None:
        linewidth = 0.4 if topology == "custom" else 0
    pc = PatchCollection(
        patches, cmap=cmap, edgecolor=edgecolor, linewidth=linewidth,
    )
    ax.add_collection(pc)

    n_frames = filtered.shape[0] // plot_step

    def init():
        pc.set_facecolor(smap.to_rgba(np.zeros(filtered.shape[1])))
        return (pc,)

    def update(frame):
        idx = frame * plot_step
        if idx >= filtered.shape[0]:
            return (pc,)
        pc.set_facecolor(smap.to_rgba(filtered[idx]))
        return (pc,)

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=False, repeat=False,
    )
    fig.tight_layout()
    return ani
