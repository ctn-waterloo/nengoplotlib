"""Feature builders for neuron sorting.

Sorting algorithms need a per-neuron feature vector. Most of the time that's
just the smoothed spike train; sometimes the caller wants to sort on encoders
or another externally-computed matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(X, t=None, dt=None, filter_width=0.02):
    """Gaussian-smooth columns of *X* along the time axis.

    Either *t* (time array) or *dt* (timestep) must be given so the filter
    width in seconds can be converted to samples.
    """
    X = np.asarray(X, dtype=float)
    if dt is None:
        if t is None:
            raise ValueError("smooth requires either t or dt")
        t = np.asarray(t)
        dt = (t[-1] - t[0]) / (len(t) - 1)
    return gaussian_filter1d(X, filter_width / dt, axis=0)


def auto_features(X, t=None, smoothing=None):
    """Return the (n_neurons, n_features) matrix used for sorting.

    If *smoothing* is a positive number, smooths *X* along time first (*t*
    required). Otherwise *X* is used as-is. The result is transposed so each
    row is one neuron.
    """
    X = np.asarray(X, dtype=float)
    if smoothing is not None and smoothing > 0:
        X = smooth(X, t=t, filter_width=smoothing)
    return X.T
