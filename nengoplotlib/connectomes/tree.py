"""Network hierarchy as a typed tree.

The single source of truth for every downstream pass (sizing, layout, color,
connection aggregation, drawing). Parent pointers make ancestor / LCA lookups
O(depth) instead of the cumulative-sum reconstructions the legacy code did.

A ``build_tree(model)`` call returns a synthetic root whose ``children`` are
the model's top-level ensembles followed by its top-level networks. Iteration
order matches the legacy ``build_network_hierarchy`` so existing visuals are
preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

import nengo


_GROUP_SUFFIX = re.compile(r"^(.*?)_(\d+)$")


@dataclass(eq=False)
class Node:
    """One entry in the network hierarchy tree.

    Leaves are either ``nengo.Ensemble`` objects or ``nengo.Network`` objects
    that contain no ensembles and no sub-networks. The synthetic root has
    ``obj=None`` and ``depth=-1``; its children are at ``depth=0`` (matching
    ``layers[0]`` in the legacy representation).

    ``size``, ``theta_start``, ``theta_end`` are populated by later passes
    (``compute_sizes``, ``equalize``, ``assign_angles``). They are zero until
    those passes have run.
    """

    obj: Optional[Any]
    raw_label: str
    depth: int
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    size: float = 0.0
    theta_start: float = 0.0
    theta_end: float = 0.0

    @property
    def is_root(self) -> bool:
        return self.obj is None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def key(self) -> str:
        """Stable unique identifier for this node, even across unlabeled objects."""
        if self.obj is None:
            return "<root>"
        return f"{type(self.obj).__name__}#{id(self.obj)}"

    @property
    def label(self) -> str:
        """Human-readable label, falling back to the type name when unlabeled.

        ``EnsembleArray`` member numbering (``foo_3``) is preserved here; use
        ``group_label`` to collapse siblings for legend de-duplication.
        """
        if self.obj is None:
            return ""
        if self.raw_label and self.raw_label.strip():
            return self.raw_label
        return type(self.obj).__name__

    @property
    def group_label(self) -> str:
        """``label`` with any trailing ``_N`` stripped (groups EnsembleArray siblings)."""
        m = _GROUP_SUFFIX.match(self.label)
        return m.group(1) if m else self.label

    @property
    def n_neurons(self) -> int:
        """Direct attribute for leaves; sum of descendants for internal nodes."""
        if self.is_leaf:
            return int(getattr(self.obj, "n_neurons", 0) or 0)
        return sum(c.n_neurons for c in self.children)

    @property
    def n_descendants(self) -> int:
        return sum(1 + c.n_descendants for c in self.children)

    def walk(self) -> Iterator["Node"]:
        """Yield self then all descendants in DFS pre-order."""
        yield self
        for c in self.children:
            yield from c.walk()

    def walk_postorder(self) -> Iterator["Node"]:
        """Yield all descendants then self in DFS post-order."""
        for c in self.children:
            yield from c.walk_postorder()
        yield self

    def leaves(self) -> Iterator["Node"]:
        for n in self.walk():
            if n.is_leaf and not n.is_root:
                yield n

    def ancestor_at(self, depth: int) -> "Node":
        """Return the ancestor (or self) at ``depth``. Errors if ``depth`` exceeds self's depth."""
        if depth > self.depth:
            raise ValueError(
                f"depth={depth} is deeper than this node (depth={self.depth})"
            )
        node = self
        while node.depth > depth:
            assert node.parent is not None
            node = node.parent
        return node

    def max_depth(self) -> int:
        """Max depth reached by any leaf descendant (inclusive)."""
        if self.is_leaf:
            return self.depth
        return max(c.max_depth() for c in self.children)


def build_tree(model: nengo.Network) -> Node:
    """Build the hierarchy tree from a Nengo model.

    Mirrors the legacy ``build_network_hierarchy``: top-level ensembles are
    enumerated before top-level sub-networks, and within each network the
    ensembles come before sub-networks. Plain ``nengo.Node`` objects are
    skipped, matching the legacy behavior.
    """
    root = Node(obj=None, raw_label="", depth=-1, parent=None)

    for ens in model.ensembles:
        root.children.append(_make_ensemble_node(ens, parent=root))
    for sub in model.networks:
        root.children.append(_make_network_node(sub, parent=root))

    return root


def _make_ensemble_node(ens: nengo.Ensemble, parent: Node) -> Node:
    return Node(
        obj=ens,
        raw_label=ens.label or "",
        depth=parent.depth + 1,
        parent=parent,
    )


def _make_network_node(net: nengo.Network, parent: Node) -> Node:
    node = Node(
        obj=net,
        raw_label=net.label or "",
        depth=parent.depth + 1,
        parent=parent,
    )
    for ens in net.ensembles:
        node.children.append(_make_ensemble_node(ens, parent=node))
    for sub in net.networks:
        node.children.append(_make_network_node(sub, parent=node))
    return node


# --------------------------------------------------------------------------- #
# Collapsing single-Network passthrough chains.
# --------------------------------------------------------------------------- #

def collapse_passthroughs(root: Node) -> Node:
    """In-place: fold single-Network-child chains into their parent.

    Mirrors the legacy ``collapse_node`` loop. While a node has exactly one
    child and that child is a Network (not an Ensemble), the child's children
    are lifted into the parent (re-parented; depths shifted up by one). The
    outer node's label is preserved. The descent stops as soon as the single
    child is an Ensemble, or the node has zero / multiple children.

    The synthetic root itself is never collapsed -- its children correspond
    to the model's top-level entities, which the legacy code also keeps.

    Returns the same ``root`` instance, mutated.
    """
    for c in root.children:
        _collapse(c)
    return root


def _collapse(node: Node) -> None:
    while _is_passthrough(node):
        inner = node.children[0]
        node.children = inner.children
        for c in node.children:
            c.parent = node
            _shift_depth(c, -1)
    for c in node.children:
        _collapse(c)


def _is_passthrough(node: Node) -> bool:
    """True iff ``node`` has exactly one child and that child is a Network.

    Equivalent to the legacy "no ensembles and exactly one subnetwork" check:
    in the new model an Ensemble child means ``children == 1`` with an
    Ensemble obj, which is exactly the case we do NOT collapse.
    """
    if len(node.children) != 1:
        return False
    return isinstance(node.children[0].obj, nengo.Network)


def _shift_depth(node: Node, delta: int) -> None:
    node.depth += delta
    for c in node.children:
        _shift_depth(c, delta)
