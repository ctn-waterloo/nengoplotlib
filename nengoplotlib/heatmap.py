"""Color heatmap of neural activity over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(t, X, cmap="jet", ax=None, **kwargs):
    """Plot a (time x neuron) activity heatmap with ``pcolormesh``.

    Parameters
    ----------
    t : (n_time,) array
    X : (n_time, n_neurons) array
        Already-filtered activity. To smooth raw spikes first, use
        :func:`nengoplotlib.sorting.smooth`.
    cmap : str or Colormap
    ax : matplotlib.axes.Axes, optional
    kwargs : dict
        Forwarded to ``pcolormesh``.

    Returns
    -------
    matplotlib.collections.QuadMesh
    """
    t = np.asarray(t)
    X = np.asarray(X)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("shading", "auto")
    return ax.pcolormesh(t, np.arange(X.shape[1]), X.T, cmap=cmap, **kwargs)
