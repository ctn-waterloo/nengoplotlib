"""Correlation matrix between ensembles, with axis-labels grouped by
hierarchy.

Two pieces here:

- :func:`get_correlation_matrix` -- probe every (or a chosen subset of)
  ensemble in a Nengo model, run a simulation, and return the pairwise
  correlation matrix of their filtered spike counts. Drop-in workflow
  helper to feed straight into :func:`plot_correlation`.

- :func:`plot_correlation` -- display the matrix with row / column groups
  inferred from the model's hierarchy. Each leaf's ancestor at the chosen
  depth becomes a group; consecutive leaves in the same group get a shared
  axis label and colored sidebar.

If a subset of ensembles was probed, pass that subset through both
functions and the plot will only show those rows / columns.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import nengo
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .keys import display_label, get_key
from .tree import Node, build_tree, collapse_passthroughs


# --------------------------------------------------------------------------- #
# Probe + simulate helper
# --------------------------------------------------------------------------- #

def get_correlation_matrix(
    model: nengo.Network,
    T: float,
    probe_ensembles: Optional[List[nengo.Ensemble]] = None,
    *,
    max_neurons: int = 100,
    sample_every: float = 0.1,
    synapse: float = 0.1,
):
    """Probe each ensemble's spike output, simulate for ``T`` seconds, and
    return a pairwise correlation matrix.

    Each probed ensemble is sampled by averaging its filtered spike output
    across (up to) ``max_neurons`` neurons. Useful as a quick "what's
    co-active with what?" matrix to feed into :func:`plot_correlation`.

    Parameters
    ----------
    model : nengo.Network
        Model to probe and run.
    T : float
        Simulation duration in seconds.
    probe_ensembles : list of nengo.Ensemble, optional
        Which ensembles to probe. Defaults to ``model.all_ensembles``. Pass a
        subset to skip slow / uninteresting ensembles; the returned list
        tells :func:`plot_correlation` which rows / cols correspond to
        which leaves of the hierarchy.
    max_neurons : int, default 100
        Cap on the number of neurons probed per ensemble (sub-sample of the
        first N neurons). Keeps memory bounded for large ensembles.
    sample_every : float, default 0.1
        Probe sample period in seconds. Sub-sampling keeps the matrix size
        manageable for long simulations.
    synapse : float, default 0.1
        Filter time constant applied at the probe.

    Returns
    -------
    corr_matrix : np.ndarray
        Square ``(N, N)`` correlation matrix; row ``i`` / col ``i`` is
        ``ensembles[i]``.
    ensembles : list of nengo.Ensemble
        Probed ensembles in matrix row / column order. Pass straight to
        :func:`plot_correlation` as ``ensembles=...``.
    signals_dict : dict[str, np.ndarray]
        ``{get_key(ens): per-timestep mean signal}`` for each probed
        ensemble -- convenient as the ``signals=`` kwarg of
        :class:`InteractiveConnectome`.
    """
    if probe_ensembles is None:
        probe_ensembles = list(model.all_ensembles)
    else:
        probe_ensembles = list(probe_ensembles)

    probes: dict = {}
    with model:
        for ens in probe_ensembles:
            n = min(ens.n_neurons, max_neurons)
            probes[ens] = nengo.Probe(
                ens.neurons[:n],
                sample_every=sample_every,
                synapse=synapse,
            )

    sim = nengo.Simulator(model)
    with sim:
        sim.run(T)

    signals = np.array(
        [np.mean(sim.data[probes[ens]], axis=1) for ens in probe_ensembles]
    )
    corr_matrix = np.corrcoef(signals)
    signals_dict = {get_key(ens): signals[i] for i, ens in enumerate(probe_ensembles)}

    return corr_matrix, probe_ensembles, signals_dict


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_correlation(
    corr_matrix,
    model: nengo.Network,
    ensembles: Optional[List[nengo.Ensemble]] = None,
    label_depth: int = 1,
    ax=None,
    cmap: str = "viridis",
):
    """Plot a correlation matrix with groups inferred from the network's
    hierarchy.

    Parameters
    ----------
    corr_matrix
        Square ``(N, N)`` matrix. Row ``i`` / col ``i`` must correspond to
        ``ensembles[i]`` (or the i-th leaf of ``build_tree(model)`` when
        ``ensembles`` is None).
    model
        The Nengo model. Used purely for its hierarchy.
    ensembles : list of nengo.Ensemble, optional
        Which ensembles the rows / cols correspond to. Defaults to all leaf
        ensembles in DFS order. Any entry that isn't a leaf of the model's
        hierarchy is silently skipped -- so passing the ``ensembles`` list
        returned by :func:`get_correlation_matrix` (potentially a subset of
        the model) just works.
    label_depth : int, default 1
        Tree depth used to group rows / columns. ``0`` groups by top-level
        container, ``1`` by the next level, etc. Leaves shallower than
        ``label_depth`` group by themselves.
    ax : matplotlib axes, optional
        Existing axes to draw into.
    cmap : str
        Matplotlib colormap name for the matrix.
    """
    root = build_tree(model)
    collapse_passthroughs(root)
    corr = np.asarray(corr_matrix)

    if ensembles is None:
        leaves: List[Node] = list(root.leaves())
        if corr.shape != (len(leaves), len(leaves)):
            raise ValueError(
                f"corr_matrix shape {corr.shape} doesn't match "
                f"{len(leaves)} leaf ensembles in the model"
            )
    else:
        if corr.shape != (len(ensembles), len(ensembles)):
            raise ValueError(
                f"corr_matrix shape {corr.shape} doesn't match "
                f"len(ensembles)={len(ensembles)}"
            )
        obj_to_leaf = {leaf.obj: leaf for leaf in root.leaves()}
        leaves = []
        keep_idx = []
        for i, ens in enumerate(ensembles):
            leaf = obj_to_leaf.get(ens)
            if leaf is not None:
                leaves.append(leaf)
                keep_idx.append(i)
        if keep_idx != list(range(len(ensembles))):
            corr = corr[np.ix_(keep_idx, keep_idx)]

    if not leaves:
        raise ValueError("no ensembles in `ensembles` were found in the model's tree")

    n_ens = len(leaves)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    def group_for(leaf: Node) -> Node:
        return leaf if leaf.depth < label_depth else leaf.ancestor_at(label_depth)

    # Walk in given order; whenever the ancestor-group changes, close out the run.
    boundaries: List[Tuple[int, int, Node]] = []
    cur_group: Optional[Node] = None
    start = 0
    for i, leaf in enumerate(leaves):
        g = group_for(leaf)
        if g is not cur_group:
            if cur_group is not None:
                boundaries.append((start, i - 1, cur_group))
            cur_group = g
            start = i
    if cur_group is not None:
        boundaries.append((start, n_ens - 1, cur_group))

    tick_positions = [(s + e) / 2.0 for s, e, _ in boundaries]
    tick_labels = [display_label(get_key(g.obj)) for _, _, g in boundaries]

    for s, _, _ in boundaries:
        if s > 0:
            ax.axhline(y=s - 0.5, color="white", linewidth=2)
            ax.axvline(x=s - 0.5, color="white", linewidth=2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries)))
    for i, (s, e, _) in enumerate(boundaries):
        ax.add_patch(Rectangle((-1.2, s - 0.5), 0.3, e - s + 1,
                               facecolor=colors[i], clip_on=False))
        ax.add_patch(Rectangle((s - 0.5, n_ens + 0.2), e - s + 1, 0.3,
                               facecolor=colors[i], clip_on=False))

    ax.add_patch(Rectangle((-0.5, -0.5), n_ens, n_ens,
                           linewidth=2, edgecolor="black",
                           facecolor="none", zorder=5))

    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    ax.set_xlim(-1.5, n_ens)
    ax.set_ylim(n_ens + 0.8, -1)

    plt.tight_layout()
    return fig, ax
