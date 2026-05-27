"""Neuron / spike sorting and merging.

The original spike-train ordering routines (``cluster``, ``merge``,
``sample_by_variance``, ``sample_by_activity``) come from
``nengo_extras.plot_spikes`` -- credit the original authors. They've been
generalized here to also cover 2D layouts (SOM on a rect or hex grid) and to
be reusable across trials via :class:`NeuronSorter`.

Quick start
-----------

    from nengoplotlib.sorting import NeuronSorter, sort_neurons

    sorted_spikes, sorter = sort_neurons(spikes, t=t, method='cluster', n_out=50)

    # 2D SOM, hex grid, then merge identical-cell neurons:
    sorter = NeuronSorter(method='som', ndim=2, grid='hex',
                          grid_shape=(20, 20), metric='cosine')
    merged = sorter.fit_transform(spikes, features=ensemble.encoders)
    positions = sorter.merged_positions   # for plot_grid_animation
"""

from .api import sort_neurons
from .base import NeuronSorter
from .cluster import sort_cluster
from .features import smooth, auto_features
from .merge import merge_1d, merge_2d
from .sample import sample_by_activity, sample_by_variance, sample_random
from .som import SOM, sort_som
from .voronoi import (
    Parcellation,
    kmeans_voronoi_parcellation,
    voronoi_parcellation,
)

__all__ = [
    "NeuronSorter",
    "Parcellation",
    "sort_neurons",
    "sort_cluster",
    "sort_som",
    "voronoi_parcellation",
    "kmeans_voronoi_parcellation",
    "SOM",
    "smooth",
    "auto_features",
    "merge_1d",
    "merge_2d",
    "sample_by_variance",
    "sample_by_activity",
    "sample_random",
]
