"""Connection weight / decoder matrix heatmaps.

Accepts either a raw weights array or a ``(nengo.Connection, nengo.Simulator)``
pair. Optional row / column sorters from :mod:`nengoplotlib.sorting` keep
similar neurons adjacent, which makes structure in the weight matrix easier
to spot.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def _resolve_order(order_arg, n):
    """Coerce a permutation array or NeuronSorter into an int array of length n."""
    if order_arg is None:
        return None
    if hasattr(order_arg, "order") and order_arg.order is not None:
        order = np.asarray(order_arg.order).ravel()
    else:
        order = np.asarray(order_arg).ravel()
    if order.shape != (n,):
        raise ValueError(
            f"order length {order.shape} doesn't match axis size {n}"
        )
    return order


def plot_weight_matrix(
    weights,
    sim=None,
    row_order=None,
    col_order=None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    aspect: Union[str, float] = "auto",
    colorbar: bool = True,
    xlabel: str = "pre",
    ylabel: str = "post",
    ax=None,
    figsize=(5, 4),
    dpi=None,
):
    """Plot a (post x pre) connection weight or decoder matrix as a heatmap.

    Parameters
    ----------
    weights : (n_post, n_pre) array or nengo.Connection
        If a connection is passed, *sim* must also be given; the function
        reads ``sim.data[conn].weights``.
    sim : nengo.Simulator, optional
        Required only when *weights* is a connection.
    row_order, col_order : array or NeuronSorter, optional
        Permutation applied before plotting. A :class:`NeuronSorter`'s
        ``.order`` (1D permutation) is used. Pass ``None`` to keep the
        original order.
    cmap : str or Colormap
        Default ``'RdBu_r'`` (diverging) because connection weights and
        decoders are usually signed.
    symmetric : bool
        If True and *vmin*/*vmax* aren't both given, sets a colormap range
        symmetric about 0 (max(|w|)).
    vmin, vmax : float, optional
        Override color limits.
    aspect : str or float
        Forwarded to ``imshow``. ``'auto'`` lets cells be rectangular.
    colorbar : bool
        Attach a colorbar.

    Returns
    -------
    (fig, ax, image) : tuple
    """
    if hasattr(weights, "size_in") and hasattr(weights, "size_out"):
        # looks like a nengo.Connection
        if sim is None:
            raise ValueError("a Connection requires sim=...")
        W = np.asarray(sim.data[weights].weights)
    else:
        W = np.asarray(weights)

    if W.ndim != 2:
        raise ValueError(f"weights must be 2D; got shape {W.shape}")

    n_post, n_pre = W.shape

    r = _resolve_order(row_order, n_post)
    c = _resolve_order(col_order, n_pre)
    if r is not None:
        W = W[r, :]
    if c is not None:
        W = W[:, c]

    if vmin is None or vmax is None:
        if symmetric:
            lim = float(np.max(np.abs(W))) or 1.0
            vmin = -lim if vmin is None else vmin
            vmax = lim if vmax is None else vmax
        else:
            vmin = float(W.min()) if vmin is None else vmin
            vmax = float(W.max()) if vmax is None else vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    img = ax.imshow(W, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        fig.colorbar(img, ax=ax, label="weight")

    return fig, ax, img
