"""Connectome visualization for Nengo networks.

Public API (the only names re-exported here):

    plot_connectome        circular ring-plot of a nengo.Network
    plot_correlation       correlation matrix with hierarchical group labels
    get_correlation_matrix probe + simulate + correlate helper
    InteractiveConnectome  click-driven selection/inspection for either plot
    ConnectomePlot         nengo.Node subclass for Nengo GUI display

Internal modules are reachable via their dotted paths:

    tree.py         Node, build_tree, collapse_passthroughs
    sizing.py       compute_sizes, by_n_neurons, by_n_descendants
    layout.py       assign_angles, equalize
    connections.py  Edge, extract_connections, aggregate_to_level, lca, ...
    draw.py         polar_to_cartesian, bezier_curve, draw_arc, draw_path
    legend.py       build_hierarchical_legend
    plot.py         _plot_connectome (tree-driven orchestrator, internal)
    api.py          plot_connectome  (model -> plot, top-level entry)
    correlation.py  plot_correlation, get_correlation_matrix
    interactive.py  InteractiveConnectome
    gui.py          ConnectomePlot (Nengo GUI Node)
    keys.py         get_key, display_label
"""

from .api import plot_connectome
from .correlation import get_correlation_matrix, plot_correlation
from .gui import ConnectomePlot
from .interactive import InteractiveConnectome

__all__ = [
    "plot_connectome",
    "plot_correlation",
    "get_correlation_matrix",
    "InteractiveConnectome",
    "ConnectomePlot",
]
