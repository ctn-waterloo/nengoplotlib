"""Sub-sampling utilities for spike / activity matrices.

Derived from ``nengo_extras.plot_spikes``; kept here so the sorting subsystem
is self-contained.
"""

from __future__ import annotations

import numpy as np

from .features import smooth


def sample_by_variance(X, num, t=None, dt=None, filter_width=0.02):
    """Return the *num* columns of *X* with the highest smoothed variance.

    Returns
    -------
    selected : (n_time, num) array
    indices  : (num,) int array
        The column indices selected, in order of decreasing variance.
    """
    X = np.asarray(X)
    if X.shape[1] <= num:
        return X, np.arange(X.shape[1])
    filtered = smooth(X, t=t, dt=dt, filter_width=filter_width)
    indices = np.argsort(np.var(filtered, axis=0))[-1:-num - 1:-1]
    return X[:, indices], indices


def sample_by_activity(X, num, blocksize=None):
    """Return the *num* columns of *X* with the highest total activity.

    With *blocksize*, splits columns into contiguous blocks and picks the most
    active within each block (preserves rough ordering when the input is
    already sorted).
    """
    X = np.asarray(X)
    n_neurons = X.shape[1]
    if n_neurons <= num:
        return X, np.arange(n_neurons)

    if blocksize is None:
        blocksize = n_neurons

    n_blocks = int(np.ceil(n_neurons / blocksize))
    n_sel = int(np.ceil(num / n_blocks))

    selected_cols = []
    indices = []
    for i in range(n_blocks):
        lo, hi = i * blocksize, (i + 1) * blocksize
        block = X[:, lo:hi]
        activity = np.sum(block, axis=0)
        order = np.argsort(activity)[-1:-n_sel - 1:-1]
        selected_cols.append(block[:, order])
        indices.append(lo + order)

    return np.concatenate(selected_cols, axis=1), np.concatenate(indices)


def sample_random(X, num, random_state=None):
    """Return *num* random columns of *X*."""
    X = np.asarray(X)
    if X.shape[1] <= num:
        return X, np.arange(X.shape[1])
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[1], size=num, replace=False)
    return X[:, indices], indices
