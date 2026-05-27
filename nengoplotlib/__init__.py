"""nengoplotlib -- Plotting utilities for Nengo networks.

Subpackages and modules
-----------------------

    connectomes   Circular ring-plots of nengo.Network hierarchies.
    sorting       Neuron / spike sorting (1D and 2D, cluster or SOM) and merging.
    raster        plot_spikes (alpha-channel grayscale raster).
    heatmap       plot_heatmap (pcolormesh activity heatmap).
    traces        plot_traces (stacked offset traces, calcium-style).
    psth          plot_psth (per-neuron multi-trial raster + firing rate).
    animation     plot_scrolling_raster, plot_grid_animation.

Public re-exports (the top-level names) are listed below.
"""

from .animation import plot_grid_animation, plot_scrolling_raster

try:
    from .connectomes import (
        ConnectomePlot,
        InteractiveConnectome,
        get_correlation_matrix,
        plot_connectome,
        plot_correlation,
    )
except ImportError:  # nengo not installed -- connectome plots unavailable
    ConnectomePlot = None
    InteractiveConnectome = None
    get_correlation_matrix = None
    plot_connectome = None
    plot_correlation = None

from .heatmap import plot_heatmap
from .isi import plot_isi
from .phase_portrait import plot_phase_portrait
from .psth import plot_psth
from .raster import plot_spikes
from .weights import plot_weight_matrix
from .sorting import (
    NeuronSorter,
    Parcellation,
    SOM,
    kmeans_voronoi_parcellation,
    merge_1d,
    merge_2d,
    sample_by_activity,
    sample_by_variance,
    sample_random,
    smooth,
    sort_cluster,
    sort_neurons,
    sort_som,
    voronoi_parcellation,
)
from .traces import plot_traces
from . import style

__all__ = [
    # plots
    "plot_spikes",
    "plot_heatmap",
    "plot_traces",
    "plot_psth",
    "plot_scrolling_raster",
    "plot_grid_animation",
    "plot_phase_portrait",
    "plot_isi",
    "plot_weight_matrix",
    # connectomes
    "plot_connectome",
    "plot_correlation",
    "get_correlation_matrix",
    "InteractiveConnectome",
    "ConnectomePlot",
    # sorting
    "NeuronSorter",
    "Parcellation",
    "sort_neurons",
    "sort_cluster",
    "sort_som",
    "voronoi_parcellation",
    "kmeans_voronoi_parcellation",
    "SOM",
    "smooth",
    "merge_1d",
    "merge_2d",
    "sample_by_variance",
    "sample_by_activity",
    "sample_random",
    # style helpers
    "style",
]
