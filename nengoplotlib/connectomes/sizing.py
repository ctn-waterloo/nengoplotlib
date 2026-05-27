"""Wedge-size assignment.

Each ``Node`` has a ``size`` field that downstream code (angles, equalize,
draw) interprets uniformly. The strategy that fills it is decided here.

The two built-in strategies map directly onto the legacy ``value`` strings:

============================  ==============================================
Legacy value                  New leaf-size function
============================  ==============================================
``'n_neurons'``               :func:`by_n_neurons`
``'n_children'``              :func:`by_n_descendants`
callable ``f(obj) -> float``  wrap as ``lambda n: f(n.obj)``
============================  ==============================================

For internal nodes the default is ``sum(c.size for c in children)``, which
reproduces both legacy semantics: nengo Networks' ``n_neurons`` attribute is
already a recursive sum, and ``n_children`` was the count of leaf-paths
passing through the node, again a sum. For a callable ``value`` the legacy
applied ``value(obj)`` directly at every depth -- pass ``internal_size`` to
match that behavior.
"""

from __future__ import annotations

from typing import Callable, Optional

from .tree import Node


SizeFn = Callable[[Node], float]


def by_n_neurons(node: Node) -> float:
    """Leaf size = the object's ``n_neurons`` attribute (0 if absent)."""
    return float(getattr(node.obj, "n_neurons", 0) or 0)


def by_n_descendants(node: Node) -> float:
    """Leaf size = 1.

    Internal sizes (computed as sum-of-children) then equal the number of
    leaves in the subtree, matching the legacy ``n_children`` increment.
    """
    return 1.0


def compute_sizes(
    root: Node,
    leaf_size: SizeFn = by_n_neurons,
    internal_size: Optional[SizeFn] = None,
) -> None:
    """Populate ``node.size`` for every node in ``root``'s subtree.

    Parameters
    ----------
    root
        Tree root (typically the synthetic root from :func:`build_tree`).
    leaf_size
        Applied to every leaf.
    internal_size
        Applied to every internal node. If ``None`` (default), internal
        sizes are computed as the sum of their children's sizes.
    """
    for n in root.walk_postorder():
        if n.is_leaf:
            n.size = leaf_size(n)
        elif internal_size is not None:
            n.size = internal_size(n)
        else:
            n.size = sum(c.size for c in n.children)
