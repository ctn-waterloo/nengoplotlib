"""Spike raster plots.

The :func:`plot_spikes` function is adapted from
``nengo_extras.plot_spikes.plot_spikes`` (Applied Brain Research et al.). The
original module also provided the spike-ordering / merging helpers that now
live in :mod:`nengoplotlib.sorting`.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


_GRAY_R_A = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gray_r_a", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
)


def plot_spikes(t, spikes, contrast_scale=1.0, ax=None, **kwargs):
    """Plot a spike raster with an alpha-channel grayscale colormap.

    Uses :meth:`matplotlib.axes.Axes.imshow`. The default colormap has a
    transparent background so other artists can be drawn underneath.

    Parameters
    ----------
    t : (n,) array
        Equidistant time indices.
    spikes : (n, m) array
        Spike (or activity) data for *m* neurons at *n* time points.
    contrast_scale : float
        Multiplies the imshow ``vmax``; lower values darken the raster.
    ax : matplotlib.axes.Axes, optional
    kwargs : dict
        Forwarded to ``imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    t = np.asarray(t)
    spikes = np.asarray(spikes)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("cmap", _GRAY_R_A)
    kwargs.setdefault("interpolation", "nearest")
    kwargs.setdefault("extent", (t[0], t[-1], 0.0, spikes.shape[1]))

    img = ax.imshow(spikes.T, **kwargs)
    img.set_clim(0.0, np.max(spikes) * contrast_scale)
    return img
