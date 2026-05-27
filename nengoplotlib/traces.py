"""Stacked-offset activity traces (calcium-imaging style).

Each neuron's trace is normalized to its own max, shifted vertically by a
fixed offset, and (optionally) drawn over a white filled region underneath so
overlapping traces don't smear into each other.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_traces(
    t,
    X,
    offset=0.5,
    cmap=None,
    fill=True,
    normalize=True,
    linewidth=0.8,
    alpha=0.7,
    color="k",
    ax=None,
):
    """Plot stacked activity traces.

    Parameters
    ----------
    t : (n_time,) array
    X : (n_time, n_neurons) array
        Smoothed activity. Sort/filter beforehand if desired.
    offset : float
        Vertical gap between successive traces (in the same units the traces
        are normalized to, so the default 0.5 leaves traces touching when
        ``normalize=True``).
    cmap : str, Colormap, or None
        If given, each trace is colored with a cycling colormap entry.
        Otherwise *color* is used.
    fill : bool
        Draw a white ``fill_between`` under each trace so the trace above
        visually occludes the trace below.
    normalize : bool
        Divide each trace by its max before plotting.
    color : color
        Used when *cmap* is None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    t = np.asarray(t)
    X = np.asarray(X, dtype=float)
    if ax is None:
        _, ax = plt.subplots()

    cmap_obj = plt.get_cmap(cmap) if cmap is not None else None
    n = X.shape[1]

    for i in range(n):
        trace = X[:, i]
        if normalize:
            m = np.max(trace)
            if m > 0:
                trace = trace / m
        baseline = i * offset
        shifted = trace + baseline
        z_fill = n - 2 * i
        z_line = z_fill + 1
        c = color if cmap_obj is None else cmap_obj(i % cmap_obj.N)
        if fill:
            ax.fill_between(
                t, baseline, shifted,
                color="white", alpha=alpha, zorder=z_fill,
            )
        ax.plot(
            t, shifted,
            color=c, linewidth=linewidth, alpha=alpha,
            zorder=z_line, clip_on=False,
        )

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-offset, n * offset)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
