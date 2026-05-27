"""Merging neighbors after sorting.

In 1D, average contiguous blocks of an already-ordered matrix. In 2D, average
neurons that share a SOM cell (or any cell-assignment array).
"""

from __future__ import annotations

import numpy as np


def merge_1d(X, n_out, order=None):
    """Block-average columns of *X* down to *n_out* columns.

    Parameters
    ----------
    X : (n_time, n_neurons) array
    n_out : int
        Target number of columns. If ``n_out >= n_neurons``, *X* is returned
        unchanged (with positions ``0..n_neurons-1``).
    order : (n_neurons,) array, optional
        If given, *X* is permuted by *order* before block-averaging.

    Returns
    -------
    merged : (n_time, n_out) array
    positions : (n_out,) float array
        Centers of each block in the post-sort coordinate (0..n_neurons-1).
    """
    X = np.asarray(X, dtype=float)
    if order is not None:
        X = X[:, np.asarray(order)]
    n_neurons = X.shape[1]
    if n_neurons <= n_out:
        return X, np.arange(n_neurons, dtype=float)

    # ``np.array_split`` gives exactly n_out contiguous slices that cover
    # 0..n_neurons-1; some slices are one element longer than others when
    # n_neurons isn't a clean multiple of n_out. This avoids the
    # ``ceil(n/n_out) * n_out > n_neurons`` overflow that produces empty
    # trailing blocks (and NaN columns).
    splits = np.array_split(np.arange(n_neurons), n_out)
    merged = np.column_stack([X[:, s].mean(axis=1) for s in splits])
    centers = np.array([s.mean() for s in splits])
    return merged, centers


def merge_2d(X, cell_assignments, n_cells, cell_centers=None):
    """Average columns of *X* by 2D cell assignment.

    Parameters
    ----------
    X : (n_time, n_neurons) array
    cell_assignments : (n_neurons,) int array
        Cell index for each input neuron.
    n_cells : int
        Total number of cells (some may be empty).
    cell_centers : (n_cells, 2) array, optional
        2D coordinates of each cell. If given, the returned positions array
        contains the centers of *non-empty* cells in the same order as the
        merged columns.

    Returns
    -------
    merged : (n_time, n_nonempty) array
        One column per non-empty cell.
    positions : (n_nonempty, 2) array, only if cell_centers given
    nonempty : (n_nonempty,) int array
        Indices of the cells that had at least one neuron assigned.
    """
    X = np.asarray(X, dtype=float)
    cell_assignments = np.asarray(cell_assignments)

    counts = np.bincount(cell_assignments, minlength=n_cells)
    nonempty = np.flatnonzero(counts > 0)

    sums = np.zeros((X.shape[0], n_cells))
    np.add.at(sums.T, cell_assignments, X.T)
    merged = sums[:, nonempty] / counts[nonempty]

    if cell_centers is not None:
        return merged, np.asarray(cell_centers)[nonempty], nonempty
    return merged, nonempty
