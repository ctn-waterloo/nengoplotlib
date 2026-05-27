"""Hierarchical-clustering 1D order for neuron sorting.

Derived from ``nengo_extras.plot_spikes.cluster``.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

from .features import auto_features


def sort_cluster(X, t=None, smoothing=0.002, metric="euclidean", method="single"):
    """Return a 1D ordering of columns of *X* via hierarchical clustering.

    Parameters
    ----------
    X : (n_time, n_neurons) array
    t : (n_time,) array, optional
        Required only when *smoothing* > 0 (to convert filter width to samples).
    smoothing : float, optional
        Gaussian filter width in seconds applied before computing similarity.
        ``None`` or 0 disables smoothing.
    metric, method : str
        Forwarded to ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    order : (n_neurons,) int array
    X_sorted : (n_time, n_neurons) array
    """
    features = auto_features(X, t=t, smoothing=smoothing)
    order = np.asarray(to_tree(linkage(features, metric=metric, method=method)).pre_order())
    return order, np.asarray(X)[:, order]
