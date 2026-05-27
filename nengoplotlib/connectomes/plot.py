"""Tree-driven plot orchestrator.

Replaces the legacy ``plot_connectome``. Every quantity that the legacy code
recovered via cumulative-sum lookups -- top-level ancestor, sibling
position, lowest common ancestor for bundled edges, legend parent chains --
is now read directly off the ``Node`` tree.

Public entry (internal to the package):

    _plot_connectome(root, edges, **opts) -> (ax, fig, plot_objs)

``root`` must already have been collapsed, sized, optionally equalized, and
have angles assigned (see :mod:`tree`, :mod:`sizing`, :mod:`layout`).
``edges`` are the un-aggregated :class:`Edge` list; aggregation to
``connection_at`` happens inside.

``plot_objs`` matches the legacy dict shape so ``InteractiveConnectome``
continues to work unchanged: ``{'wedges': [(Wedge, str)...],
'labels': [text_or_legend], 'lines': [(line_obj, str)...]}``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from .connections import Edge, aggregate_to_level
from .draw import draw_arc, draw_path
from .keys import display_label as _display_label
from .keys import get_key as _legacy_key_for_obj
from .legend import build_hierarchical_legend
from .tree import Node


# --------------------------------------------------------------------------- #
# Tree-driven helpers
# --------------------------------------------------------------------------- #

def _segments_at_depth(root: Node, depth: int, plot_depths: int) -> List[Node]:
    """All segments to render at ``depth``, in DFS order.

    A segment is either an actual node at this depth, or a leaf whose own
    depth is shallower (which the legacy padded forward into every deeper
    layer). Padded segments only contribute up to ``plot_depths - 1``.
    """
    out = []
    for n in root.walk():
        if n.is_root:
            continue
        if n.depth == depth:
            out.append(n)
        elif n.is_leaf and n.depth < depth and depth < plot_depths:
            out.append(n)
    return out


def _is_padded(node: Node, depth: int) -> bool:
    return node.is_leaf and node.depth < depth


def _legacy_key(node: Node) -> str:
    """Per-wedge label string; matches the format InteractiveConnectome
    expects (and the legacy ``get_key`` produced)."""
    return _legacy_key_for_obj(node.obj)


def _display(node: Node) -> str:
    return _display_label(_legacy_key(node))


# --------------------------------------------------------------------------- #
# Color: top-level from cmap; deeper as a per-depth blend toward white,
# spread across siblings sharing a top-level ancestor.
# --------------------------------------------------------------------------- #

def _make_color_for(root: Node, plot_depths: int, base_cmap):
    """Closure: ``color_for(node, depth) -> rgba``.

    Memoized: each call computes its segment-group lazily and caches the
    sibling lookup table per top-level ancestor.
    """
    sibling_cache: dict = {}

    def siblings_under(top: Node, depth: int) -> List[Node]:
        key = (id(top), depth)
        cached = sibling_cache.get(key)
        if cached is not None:
            return cached
        out: List[Node] = []
        for n in top.walk():
            if n.depth == depth:
                out.append(n)
            elif n.is_leaf and n.depth < depth and depth < plot_depths:
                out.append(n)
        sibling_cache[key] = out
        return out

    def color_for(node: Node, depth: int):
        top = node.ancestor_at(0)
        top_idx = root.children.index(top)
        if depth == 0:
            return tuple(base_cmap(top_idx))
        base_rgb = np.array(base_cmap(top_idx)[:3])
        blend1 = (plot_depths - depth + 0.01) / plot_depths
        blend2 = (plot_depths - depth + 0.9) / plot_depths
        sibs = siblings_under(top, depth)
        # Identity-based index (Node has eq=False, so ``index`` matches by id).
        try:
            i = sibs.index(node)
        except ValueError:
            i = 0
        n_sibs = max(len(sibs), 1)
        # Matches the LinearSegmentedColormap(N=n_sibs) sampling the legacy
        # used: discrete steps from blend1 (lightest) to blend2 (most saturated).
        pos = i / (n_sibs - 1) if n_sibs > 1 else 0.0
        blend = blend1 + pos * (blend2 - blend1)
        c = tuple(base_rgb * blend + (1.0 - blend)) + (1.0,)
        return c

    return color_for


# --------------------------------------------------------------------------- #
# Bundled-edge control points: walk up from each endpoint, optionally
# trimming both paths at their LCA so the curve dips through the common
# ancestor.
# --------------------------------------------------------------------------- #

def _ancestor_walk(node: Node, depth_target: int) -> List[Tuple[Node, int]]:
    """``(node, depth)`` knots from the leaf up to (but excluding) the root,
    padding a leaf at ``depth < depth_target`` with copies of itself at the
    deeper layers. Mirrors the legacy bundle path."""
    pts: List[Tuple[Node, int]] = []
    if node.depth < depth_target:
        for d in range(depth_target, node.depth, -1):
            pts.append((node, d))
    pts.append((node, node.depth))
    a = node.parent
    while a is not None and not a.is_root:
        pts.append((a, a.depth))
        a = a.parent
    return pts


def _bundled_knots(src: Node, dst: Node, depth_target: int):
    """``(thetas, depths)`` for ``draw_path``."""
    src_pts = _ancestor_walk(src, depth_target)
    dst_pts = _ancestor_walk(dst, depth_target)

    # LCA at the highest common depth in both walks (going outer -> inner).
    src_by_depth = {d: n for n, d in src_pts}
    dst_by_depth = {d: n for n, d in dst_pts}
    lca_depth = -1
    for d in range(depth_target, -1, -1):
        a = src_by_depth.get(d)
        b = dst_by_depth.get(d)
        if a is not None and a is b:
            lca_depth = d
            break

    if lca_depth >= 0:
        src_pts = [(n, d) for n, d in src_pts if d >= lca_depth]
        dst_pts = [(n, d) for n, d in dst_pts if d >= lca_depth]

    path = src_pts + list(reversed(dst_pts))
    thetas = [(n.theta_start + n.theta_end) / 2.0 for n, _ in path]
    depths = [d for _, d in path]
    return thetas, depths


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def _plot_connectome(
    root: Node,
    edges: Iterable[Edge],
    *,
    ax=None,
    total_width: float = 0.4,
    max_depth: Optional[int] = None,
    label_depth: int = 0,
    labels: str = "legend",
    connection_at: Optional[int] = None,
    connection_type: str = "arc",
    font_size: float = 12,
    legend_loc: str = "center right",
    arc_params: Optional[dict] = None,
):
    arc_params = dict(arc_params or {})
    if labels not in ("legend", "inline", "none"):
        raise ValueError(f"labels must be 'legend', 'inline', or 'none'; got {labels!r}")

    tree_levels = root.max_depth() + 1
    plot_depths = tree_levels if max_depth is None else min(max_depth, tree_levels)
    assert plot_depths > 0, "max_depth must be > 0"
    label_depth = min(label_depth, plot_depths - 1)
    if connection_at is None:
        connection_at = plot_depths - 1
    connection_at = min(connection_at, plot_depths - 1)
    if connection_type == "bundled":
        assert connection_at > 0, "bundled connections require connection_at > 0"

    group_edges = aggregate_to_level(edges, connection_at)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = None

    outer_radius = 1.0
    layer_width = total_width / plot_depths

    ax.axis("equal")
    ax.set_xlim(-outer_radius - 0.2, outer_radius + 0.2 + 1)
    ax.set_ylim(-outer_radius - 0.2, outer_radius + 0.2)
    ax.set_axis_off()

    base_cmap = mpl.colormaps["tab10"]
    color_for = _make_color_for(root, plot_depths, base_cmap)

    wedge_objs: List[Tuple[Wedge, str]] = []
    label_objs: list = []

    for d in range(plot_depths):
        current_radius = outer_radius - d * layer_width
        for node in _segments_at_depth(root, d, plot_depths):
            padded = _is_padded(node, d)
            if padded:
                outer_r = current_radius + 2 * layer_width
                width = 2 * layer_width
            else:
                outer_r = current_radius + layer_width
                width = layer_width
            color = color_for(node, d)
            wedge = Wedge(
                center=(0, 0), r=outer_r,
                theta1=node.theta_start, theta2=node.theta_end,
                facecolor=color, edgecolor="w", linewidth=0.2,
                width=width, zorder=d,
            )
            ax.add_patch(wedge)
            wedge_objs.append((wedge, _legacy_key(node)))

            if labels == "inline" and d <= label_depth:
                mid = np.deg2rad((node.theta_start + node.theta_end) / 2.0)
                lr = current_radius + layer_width / 2.0
                txt = ax.text(
                    lr * np.cos(mid), lr * np.sin(mid),
                    _display(node),
                    ha="center", va="center", fontsize=font_size,
                )
                label_objs.append(txt)

    if labels == "legend" and label_depth >= 0:
        legend = build_hierarchical_legend(
            ax, root, label_depth, color_for,
            font_size=font_size, loc=legend_loc,
        )
        if legend is not None:
            label_objs = legend

    # Connections
    line_objs: list = []
    weight_total_by_src: dict = {}
    for e in group_edges:
        weight_total_by_src[id(e.src)] = weight_total_by_src.get(id(e.src), 0.0) + e.weight

    fixed_lw = arc_params.get("lw")
    fixed_alpha = arc_params.get("alpha")
    fixed_color = arc_params.get("color")
    r_conn = outer_radius - (plot_depths - 1) * layer_width

    for e in group_edges:
        src_mid = (e.src.theta_start + e.src.theta_end) / 2.0
        dst_mid = (e.dst.theta_start + e.dst.theta_end) / 2.0
        total = weight_total_by_src.get(id(e.src), 1.0) or 1.0
        share = e.weight / total

        lw = fixed_lw if fixed_lw is not None else max(share, 0.1)
        alpha = fixed_alpha if fixed_alpha is not None else max(share, 0.05)

        if fixed_color is not None:
            color = fixed_color
        else:
            top = e.src.ancestor_at(0)
            top_idx = root.children.index(top)
            color = tuple(np.array(base_cmap(top_idx))[:-1] * 0.8)

        line_label = f"{_display(e.src)} -> {_display(e.dst)}"

        if connection_type == "arc":
            line = draw_arc(ax, src_mid, dst_mid,
                            r=r_conn, lw=lw, alpha=alpha, color=color)
            line_objs.append((line, line_label))
        elif connection_type == "bundled":
            thetas, depths = _bundled_knots(e.src, e.dst, connection_at)
            line = draw_path(ax, thetas, depths,
                             r=r_conn, lw=lw, alpha=alpha, color=color)
            line_objs.append((line, line_label))

    plt.tight_layout()
    return ax, fig, {"wedges": wedge_objs, "labels": label_objs, "lines": line_objs}
