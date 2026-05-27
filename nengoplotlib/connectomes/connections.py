"""Typed connection extraction and ancestor-level aggregation.

Replaces the legacy ``extract_ensemble_connections`` (dict-of-tuples,
``set``-iteration nondeterministic) and ``aggregate_connection_map`` (string-key
lookup through padded ``obj_to_path``). With parent pointers on every
:class:`~connectomes.tree.Node`, edge endpoints are real objects and ancestor
lookup is ``Node.ancestor_at(d)`` -- O(depth), deterministic, no cumsum tricks.

An :class:`Edge` carries a synapse weight between two leaf ``Node`` s. After
``aggregate_to_level`` the endpoints are the leaves' ancestors at the requested
depth (or the leaf itself, padded forward, when it's shallower than that depth
-- mirroring the legacy padding semantics).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import nengo

from .tree import Node


@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node
    weight: float


def synapse_count(pre, post, **kwargs) -> float:
    """Default weight: pre.n_neurons * post.n_neurons. Accepts and ignores
    extra kwargs (so callers can pass ``conn=`` without the count fn caring)."""
    return pre.n_neurons * post.n_neurons


def build_leaf_index(root: Node) -> Dict[Any, Node]:
    """Map each leaf's underlying nengo object to its ``Node``.

    The map covers ensembles AND empty-network leaves. Internal nodes are
    skipped: edges only ever land on leaves at this stage; aggregation to
    higher depths is :func:`aggregate_to_level`'s job.
    """
    out: Dict[Any, Node] = {}
    for n in root.leaves():
        if n.obj is not None:
            out[n.obj] = n
    return out


def extract_connections(
    model_connections: Iterable[Any],
    leaf_index: Dict[Any, Node],
    count_fn: Callable = synapse_count,
    ignore_passthrough: bool = True,
) -> List[Edge]:
    """Build an Edge list from ``model.all_connections``.

    Passthrough ``nengo.Node`` objects are followed transitively when
    ``ignore_passthrough`` is True, so a chain ``Ens -> Node -> ... -> Ens2``
    becomes a single Edge ``(Ens, Ens2)``.

    Iteration order is fully deterministic: ``model_connections`` is iterated
    once to build an insertion-ordered out-edge map, then a second time to
    drive the BFS. Multiple connections between the same pair of ensembles are
    summed (one Edge per ``(src, dst)`` pair).
    """
    # Insertion-ordered out-edges (replaces the legacy ``defaultdict(set)``).
    out_edges: Dict[Any, List[Any]] = defaultdict(list)
    for conn in model_connections:
        out_edges[conn.pre_obj].append(conn.post_obj)

    accumulated: Dict[tuple, float] = {}

    def add(pre_obj, post_obj, **kwargs):
        pre_node = leaf_index.get(pre_obj)
        post_node = leaf_index.get(post_obj)
        if pre_node is None or post_node is None:
            return
        w = float(count_fn(pre_obj, post_obj, **kwargs))
        key = (pre_node, post_node)
        accumulated[key] = accumulated.get(key, 0.0) + w

    for conn in model_connections:
        pre = conn.pre_obj
        post = conn.post_obj
        if not isinstance(pre, nengo.Ensemble):
            continue

        if isinstance(post, nengo.Ensemble):
            add(pre, post, conn=conn)
            continue

        if isinstance(post, nengo.Node) and ignore_passthrough:
            visited = {post}
            queue = deque([post])
            while queue:
                curr = queue.popleft()
                for nxt in out_edges.get(curr, ()):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    if isinstance(nxt, nengo.Ensemble):
                        add(pre, nxt)
                    elif isinstance(nxt, nengo.Node):
                        queue.append(nxt)

    return [Edge(src=s, dst=d, weight=w) for (s, d), w in accumulated.items()]


def aggregate_to_level(edges: Iterable[Edge], level: int) -> List[Edge]:
    """Replace each endpoint with its ancestor at ``level`` (or itself if it
    is shallower than ``level``, matching the legacy padded path).

    Self-loops (where ``src`` and ``dst`` collapse to the same ancestor) are
    dropped. Weights are summed within each unique ``(src_anc, dst_anc)`` pair.
    """
    def at(node: Node) -> Node:
        return node if level > node.depth else node.ancestor_at(level)

    accumulated: Dict[tuple, float] = {}
    for e in edges:
        s = at(e.src)
        d = at(e.dst)
        if s is d:
            continue
        key = (s, d)
        accumulated[key] = accumulated.get(key, 0.0) + e.weight

    return [Edge(src=s, dst=d, weight=w) for (s, d), w in accumulated.items()]


def lca(a: Node, b: Node) -> Node:
    """Lowest common ancestor of ``a`` and ``b`` (parent pointers, O(depth))."""
    ancestors = set()
    n: Optional[Node] = a
    while n is not None:
        ancestors.add(id(n))
        n = n.parent
    n = b
    while n is not None:
        if id(n) in ancestors:
            return n
        n = n.parent
    raise ValueError("nodes do not share a common ancestor")
