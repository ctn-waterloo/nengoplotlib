"""Per-neuron multi-trial raster + smoothed firing-rate (PSTH).

For each chosen neuron, three rows are drawn:

  1. Spike raster, one trial per y-row.
  2. Trial-averaged firing rate (Gaussian-smoothed).
  3. A short colored bar encoding the same rate as intensity, for compact
     side-by-side visual comparison.

Modeled after figures like Salz et al. (2016).
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def plot_psth(
    spike_trials,
    t=None,
    dt=0.001,
    neuron_idxs=None,
    smoothing_sigma=2,
    cmap="jet",
    ax=None,
    dpi=None,
    figsize=None,
):
    """Plot a per-neuron multi-trial raster + PSTH stack.

    Parameters
    ----------
    spike_trials : list of (n_time, n_neurons) arrays, or 3D array
        One spike matrix per trial. A single 2D array is treated as one trial.
        A 3D array of shape ``(n_trials, n_time, n_neurons)`` works too.
    t : (n_time,) array, optional
        Time axis. If None, built from *dt*.
    dt : float
        Timestep, used only when *t* is None and when computing instantaneous
        firing rates.
    neuron_idxs : sequence of int, optional
        Subset of neurons to plot. Default: all neurons in the data.
    smoothing_sigma : float
        Gaussian sigma (in samples) for smoothing the mean firing rate.
    cmap : str or Colormap
        Colormap for the per-neuron intensity bar.
    ax : matplotlib.axes.Axes, optional
        Existing axes whose subplot region will host the nested raster/rate/bar
        grid. The supplied axes is removed and its ``SubplotSpec`` is used as
        the parent for the internal gridspec. When None, a new figure is
        created.
    dpi, figsize : matplotlib figure args, used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(spike_trials, np.ndarray) and spike_trials.ndim == 3:
        trials = [spike_trials[i] for i in range(spike_trials.shape[0])]
    elif isinstance(spike_trials, np.ndarray) and spike_trials.ndim == 2:
        trials = [spike_trials]
    else:
        trials = [np.asarray(tr) for tr in spike_trials]

    if neuron_idxs is not None:
        trials = [tr[:, list(neuron_idxs)] for tr in trials]

    n_trials = len(trials)
    n_time, n_neurons = trials[0].shape
    if t is None:
        t = dt * np.arange(1, n_time + 1)
    else:
        t = np.asarray(t)

    if ax is None:
        if figsize is None:
            figsize = (5, 1.5 * n_neurons)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        outer = gridspec.GridSpec(n_neurons, 1, figure=fig, hspace=0.2)
    else:
        fig = ax.figure
        parent_spec = ax.get_subplotspec()
        ax.remove()
        outer = gridspec.GridSpecFromSubplotSpec(
            n_neurons, 1, subplot_spec=parent_spec, hspace=0.2,
        )

    cmap_obj = plt.get_cmap(cmap)

    for k in range(n_neurons):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[k],
            height_ratios=[1, 0.5, 0.2], hspace=0.05,
        )
        ax_raster = fig.add_subplot(inner[0])
        ax_rate = fig.add_subplot(inner[1], sharex=ax_raster)
        ax_bar = fig.add_subplot(inner[2], sharex=ax_raster)

        for ax in (ax_raster, ax_rate, ax_bar):
            ax.spines[["right", "top", "bottom"]].set_visible(False)

        neuron_data = np.array([tr[:, k] for tr in trials])
        trial_idx, time_idx = np.where(neuron_data > 0)
        ax_raster.scatter(t[time_idx], trial_idx, marker=2, s=2, color="k")
        ax_raster.set_yticks([])
        ax_raster.set_xticks([])
        ax_raster.set_ylabel("Trial")

        mean_rate = np.mean(neuron_data > 0, axis=0) / dt
        smoothed = gaussian_filter1d(mean_rate, sigma=smoothing_sigma)
        ax_rate.plot(t, smoothed, "k", linewidth=1.5)
        ax_rate.set_ylabel("Rate [Hz]")
        ax_rate.set_xticks([])

        span = np.ptp(smoothed) + 1e-8
        normed = (smoothed - smoothed.min()) / span
        normed = np.repeat(normed[:, None], 10, axis=1)
        ax_bar.pcolormesh(
            t, np.arange(10), normed.T, cmap=cmap_obj, shading="auto",
        )
        ax_bar.set_xlim(t[0], t[-1])
        ax_bar.set_yticks([])

    ax_bar.set_xlabel("Time [s]")
    ax_bar.spines[["bottom"]].set_visible(True)
    if ax is None:
        fig.tight_layout()
    return fig
