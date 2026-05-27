"""Hierarchical legend builder.

Walks the ``Node`` tree directly to assemble an indented legend, sorting
siblings at each level alphabetically (matching the legacy
``sorted(hierarchy_dict.items())`` traversal). Padded leaves -- a leaf at
depth ``d`` rendered also at every deeper layer up to ``label_depth``
(inclusive) -- produce one legend entry per padded depth, with their own
indentation.

The caller supplies a ``color_for(node, depth)`` callback so the legend's
patch colors track whatever the plot actually drew at that (node, depth)
slot. This keeps the gradient-driven inner-ring colors in sync between the
wedges and the legend.
"""

from __future__ import annotations

from typing import Callable, Iterator, Tuple

from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from .keys import display_label as _display_label
from .keys import get_key as _get_key
from .tree import Node


_EMPTY_PATCH = Patch(facecolor="none", edgecolor="none", alpha=0)


def _entries(node: Node, label_depth: int) -> Iterator[Tuple[int, Node]]:
    """Yield ``(depth, node)`` legend entries in sorted-DFS order.

    ``label_depth`` is the 0-based deepest ring to include (inclusive). A
    leaf at depth ``d`` yields entries at every depth in ``[d, label_depth]``
    -- matching the legacy "padded leaf appears on every deeper layer"
    behavior.
    """
    if node.is_root:
        for c in sorted(node.children, key=lambda n: n.group_label):
            yield from _entries(c, label_depth)
        return

    if node.depth > label_depth:
        return

    yield (node.depth, node)

    if node.is_leaf:
        for d in range(node.depth + 1, label_depth + 1):
            yield (d, node)
    else:
        for c in sorted(node.children, key=lambda n: n.group_label):
            yield from _entries(c, label_depth)


def build_hierarchical_legend(
    ax,
    root: Node,
    label_depth: int,
    color_for: Callable[[Node, int], Tuple[float, float, float, float]],
    *,
    font_size: float = 12,
    loc: str = "center right",
):
    """Attach an indented hierarchical legend to ``ax``.

    ``label_depth`` is the 0-based deepest ring to include (inclusive).
    ``color_for(node, depth)`` returns the rgba color of the wedge drawn at
    that slot, so the legend's patches match the plot.
    """
    if label_depth < 0:
        return None

    handles = []
    labels = []
    entry_keys = []  # parallel list -- one wedge-key per legend row, in order
    for depth, node in _entries(root, label_depth):
        color = color_for(node, depth)
        key = _get_key(node.obj)  # matches the wedge's label string exactly
        patch = Patch(facecolor=color, edgecolor="w",
                      label=_display_label(key))
        if depth == 0:
            handles.append(patch)
        else:
            handles.append(tuple([_EMPTY_PATCH] * depth) + (patch,))
        labels.append(_display_label(key))
        entry_keys.append(key)

    legend = ax.legend(
        handles, labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=1.0)},
        loc=loc, fontsize=font_size, frameon=False,
    )
    ax.add_artist(legend)

    # Tag each legend label with the wedge-key it corresponds to.
    # InteractiveConnectome does its own hit-testing against these text
    # bboxes on click -- ``picker=True`` is unreliable for text artists
    # across matplotlib versions / backends (notably ipympl), so we don't
    # use it.
    for text, key in zip(legend.get_texts(), entry_keys):
        text._connectome_key = key

    return legend
