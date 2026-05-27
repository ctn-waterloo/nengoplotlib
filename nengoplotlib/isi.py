"""Interspike-interval (ISI) histograms.

Given a spike matrix ``(n_time, n_neurons)``, compute the per-neuron ISIs
(in seconds) and plot them. Two modes:

  * ``per_neuron=False`` (default): pool ISIs across the chosen neurons into
    a single histogram. Use this for population-level views.
  * ``per_neuron=True``: small-multiples grid, one histogram per neuron.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def _spike_times(column, dt):
    """Return spike times (seconds) for a single (n_time,) column."""
    idx = np.flatnonzero(column > 0)
    return idx * dt


def _isis(column, dt):
    times = _spike_times(column, dt)
    if times.size < 2:
        return np.empty(0)
    return np.diff(times)


def plot_isi(
    spikes,
    dt: float = 0.001,
    neuron_idxs=None,
    bins: int = 50,
    max_isi: Optional[float] = None,
    log_y: bool = False,
    per_neuron: bool = False,
    color: str = "C0",
    ax=None,
    figsize=None,
    dpi=None,
):
    """Plot interspike-interval histograms.

    Parameters
    ----------
    spikes : (n_time, n_neurons) array
    dt : float
        Simulation timestep in seconds.
    neuron_idxs : sequence of int, optional
        Subset of neurons to include. Default: all.
    bins : int
        Histogram bins.
    max_isi : float, optional
        Drop ISIs longer than this (seconds) before histogramming. Useful for
        clipping the tail of low-rate neurons.
    log_y : bool
        Logarithmic y-axis (useful for heavy-tailed ISI distributions).
    per_neuron : bool
        If True, draw a small-multiples grid with one histogram per neuron.
        If False, pool all selected neurons' ISIs into one histogram.
    color : color
        Bar color for the pooled histogram. Per-neuron mode uses the default
        matplotlib color cycle.
    ax : matplotlib axes, optional
        Existing axes to draw into. In pooled mode, used directly. In
        per-neuron mode, the axes is removed and its ``SubplotSpec`` is used
        as the parent for the small-multiples grid.
    figsize, dpi : matplotlib figure args, used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be a (n_time, n_neurons) array")

    if neuron_idxs is None:
        neuron_idxs = np.arange(spikes.shape[1])
    else:
        neuron_idxs = np.asarray(neuron_idxs)

    if per_neuron:
        n = len(neuron_idxs)
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        if ax is None:
            if figsize is None:
                figsize = (3 * cols, 2 * rows)
            fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
        else:
            fig = ax.figure
            parent_spec = ax.get_subplotspec()
            ax.remove()
            inner = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=parent_spec)
            axes = np.empty((rows, cols), dtype=object)
            for r in range(rows):
                for c in range(cols):
                    axes[r, c] = fig.add_subplot(inner[r, c])
        for k, j in enumerate(neuron_idxs):
            sub_ax = axes.flat[k]
            isis = _isis(spikes[:, j], dt)
            if max_isi is not None:
                isis = isis[isis <= max_isi]
            if isis.size:
                sub_ax.hist(isis, bins=bins, color=f"C{k % 10}", edgecolor="none")
            sub_ax.set_title(f"neuron {j}", fontsize=9)
            sub_ax.set_xlabel("ISI [s]")
            if log_y:
                sub_ax.set_yscale("log")
            sub_ax.spines[["right", "top"]].set_visible(False)
        for k in range(len(neuron_idxs), rows * cols):
            axes.flat[k].set_visible(False)
        if ax is None:
            fig.tight_layout()
        return fig

    # pooled
    all_isis = np.concatenate([_isis(spikes[:, j], dt) for j in neuron_idxs])
    if max_isi is not None:
        all_isis = all_isis[all_isis <= max_isi]

    if ax is None:
        if figsize is None:
            figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if all_isis.size:
        ax.hist(all_isis, bins=bins, color=color, edgecolor="none")
    ax.set_xlabel("ISI [s]")
    ax.set_ylabel("count")
    if log_y:
        ax.set_yscale("log")
    ax.spines[["right", "top"]].set_visible(False)
    return fig
