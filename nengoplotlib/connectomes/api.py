"""Top-level connectome plot entry point.

Orchestrates the pipeline:

    build_tree -> collapse_passthroughs -> compute_sizes
                 -> (optional) equalize -> assign_angles
                 -> build_leaf_index + extract_connections
                 -> plot._plot_connectome
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import nengo

from .connections import build_leaf_index, extract_connections
from .layout import assign_angles, equalize
from .plot import _plot_connectome
from .sizing import by_n_descendants, by_n_neurons, compute_sizes
from .tree import Node, build_tree, collapse_passthroughs


SizeFn = Callable[[Node], float]


def plot_connectome(
    model: nengo.Network,
    size_by: Union[str, SizeFn] = "n_neurons",
    equalize_at: Optional[int] = None,
    connection_at: Optional[int] = None,
    label_depth: int = 0,
    labels: str = "legend",
    max_depth: int = 10,
    total_width: float = 0.2,
    connection_type: str = "arc",
    arc_params: Optional[dict] = None,
    font_size: float = 12,
    legend_loc: str = "center right",
    ax=None,
):
    """Plot a connectome-style circular diagram of a Nengo model.

    Parameters
    ----------
    model : nengo.Network
        Model to visualize.
    size_by : {'n_neurons', 'n_children'} or callable
        Wedge-size metric. ``'n_neurons'`` uses each ensemble's neuron count;
        ``'n_children'`` uses the number of leaf-ensemble descendants
        (internals = sum of children, leaves = 1). A callable
        ``f(Node) -> float`` is applied at every leaf for fully custom sizing.
    equalize_at : int, optional
        0-based depth at which to force wedges to be equal-sized; ancestors
        and descendants are warped proportionally. ``None`` (default) skips
        equalization. ``equalize_at=0`` equalizes the outermost ring.
    connection_at : int, optional
        0-based depth to aggregate connections to. ``None`` (default) uses
        the deepest rendered depth -- i.e. real ensemble-to-ensemble arcs.
        ``connection_at=0`` collapses every projection into a single
        top-level inter-area arc.
    label_depth : int
        0-based deepest ring to label. ``0`` labels only the outermost ring;
        higher values label inner rings too. Ignored when ``labels='none'``.
    labels : {'legend', 'inline', 'none'}
        Where to put labels. ``'legend'`` is an indented hierarchical legend
        beside the plot; ``'inline'`` writes labels directly on the wedges;
        ``'none'`` disables labels entirely.
    max_depth : int
        Maximum number of rings to render. Useful to cap rendering on very
        deep networks. Defaults to 10 (well beyond typical depths).
    total_width : float
        Radial thickness of the entire ring stack.
    connection_type : {'arc', 'bundled'}
        ``'arc'`` draws a curve directly between endpoints; ``'bundled'``
        routes the curve through the lowest common ancestor for hierarchical
        edge bundling. Bundled requires ``connection_at > 0``.
    arc_params : dict, optional
        Style overrides for the connection arcs. Recognized keys: ``lw``,
        ``alpha``, ``color``. Any missing key falls back to per-edge
        defaults (``lw``/``alpha`` scale with the edge's share of its
        source's outgoing weight; ``color`` is a darkened version of the
        source's top-level color).
    font_size : float
        Font size used for both inline labels and the legend.
    legend_loc : str
        Matplotlib ``loc`` for the legend (e.g. ``'center right'``).
    ax : matplotlib axes, optional
        Existing axes to draw into.

    Returns
    -------
    (ax, fig, plot_objs)
        ``plot_objs`` is the dict expected by :class:`InteractiveConnectome`:
        ``{'wedges': [(Wedge, label), ...], 'labels': ..., 'lines': [...]}``.
    """
    root = build_tree(model)
    collapse_passthroughs(root)

    leaf_fn = _resolve_size_fn(size_by)
    compute_sizes(root, leaf_size=leaf_fn)
    if equalize_at is not None:
        equalize(root, equalize_at)
    assign_angles(root, total=360.0, start=0.0)

    leaf_index = build_leaf_index(root)
    edges = extract_connections(model.all_connections, leaf_index)

    return _plot_connectome(
        root, edges,
        ax=ax,
        total_width=total_width,
        max_depth=max_depth,
        label_depth=label_depth,
        labels=labels,
        connection_at=connection_at,
        connection_type=connection_type,
        font_size=font_size,
        legend_loc=legend_loc,
        arc_params=arc_params,
    )


def _resolve_size_fn(size_by: Union[str, SizeFn]) -> SizeFn:
    if callable(size_by):
        return size_by
    if size_by == "n_neurons":
        return by_n_neurons
    if size_by == "n_children":
        return by_n_descendants
    raise ValueError(
        f"size_by must be 'n_neurons', 'n_children', or a callable; got {size_by!r}"
    )
