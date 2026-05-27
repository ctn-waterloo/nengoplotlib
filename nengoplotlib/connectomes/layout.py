"""Angular layout: ``assign_angles`` + ``equalize``.

Both operate on the ``size`` field set by :mod:`sizing`. ``assign_angles``
performs hierarchical proportional subdivision: each node's
``[theta_start, theta_end]`` is a sub-range of its parent's, weighted by
``size``. ``equalize`` rescales sizes so that one chosen depth ends up with
all slots equal -- the same effect the legacy code got by warping cumulative
sums through ``scipy.interpolate.interp1d``, but expressible directly on the
tree as "rescale each slot's subtree by ``per_slot / old_size``".
"""

from __future__ import annotations

from .tree import Node


def assign_angles(root: Node, total: float = 360.0, start: float = 0.0) -> None:
    """Recursively divide ``[start, start+total]`` among ``root.children``,
    proportional to ``node.size``. Sets ``theta_start`` / ``theta_end`` on
    every non-root node. Run after :func:`compute_sizes` (and
    :func:`equalize` if you want one).
    """
    root.theta_start = start
    root.theta_end = start + total
    _subdivide(root)


def _subdivide(node: Node) -> None:
    if not node.children:
        return
    span = node.theta_end - node.theta_start
    total_size = sum(c.size for c in node.children)
    cursor = node.theta_start
    if total_size <= 0:
        # Degenerate -- divide the range equally so we still get a layout.
        step = span / len(node.children)
        for c in node.children:
            c.theta_start = cursor
            c.theta_end = cursor + step
            cursor = c.theta_end
            _subdivide(c)
        return
    for c in node.children:
        width = span * (c.size / total_size)
        c.theta_start = cursor
        c.theta_end = cursor + width
        cursor = c.theta_end
        _subdivide(c)


def equalize(root: Node, d_eq: int) -> None:
    """In-place rescale of ``node.size`` so all *effective slots* at ``d_eq``
    have equal value.

    Effective slots at depth ``d_eq`` are:

    * interior nodes at depth exactly ``d_eq`` (each covers a real subtree)
    * leaves at depth ``< d_eq`` (the legacy padding gives them slot status)

    The total ``size`` across the tree is preserved; each slot ends up at
    ``total / N``. Descendants below ``d_eq`` are scaled proportionally inside
    their owning slot (preserving relative ratios). Ancestors above ``d_eq``
    are recomputed bottom-up as the sum of their children, which also updates
    the synthetic root.

    Equivalent to the legacy interp1d cumsum warp -- see step-2 derivation
    in ``test_plots/validate_layers.py``.
    """
    def is_slot(n: Node) -> bool:
        if n.depth < 0:
            return False
        return n.depth == d_eq or (n.depth < d_eq and n.is_leaf)

    slots = [n for n in root.walk() if is_slot(n)]
    if not slots:
        return

    total = sum(n.size for n in slots)
    if total <= 0:
        return
    per_slot = total / len(slots)

    for slot in slots:
        if slot.size <= 0:
            slot.size = per_slot
            continue
        scale = per_slot / slot.size
        for n in slot.walk():
            n.size *= scale

    # Bottom-up rebuild for ancestors strictly above d_eq (including the
    # synthetic root, depth=-1). Slots and their descendants are already
    # consistent; only nodes with depth < d_eq AND children need refreshing.
    for n in root.walk_postorder():
        if n.depth < d_eq and n.children:
            n.size = sum(c.size for c in n.children)
