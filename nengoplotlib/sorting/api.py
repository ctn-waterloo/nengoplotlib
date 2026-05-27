"""One-shot helpers that mirror the original nengo_extras ergonomics."""

from __future__ import annotations

from .base import NeuronSorter


def sort_neurons(X, t=None, method="cluster", ndim=1, **kwargs):
    """Fit a :class:`NeuronSorter` and immediately transform *X*.

    Returns
    -------
    X_sorted : (n_time, n_out) array
    sorter : NeuronSorter
        Keep this around if you want to apply the same sort to other trials.
    """
    sorter = NeuronSorter(method=method, ndim=ndim, **kwargs)
    X_out = sorter.fit_transform(X, t=t)
    return X_out, sorter
