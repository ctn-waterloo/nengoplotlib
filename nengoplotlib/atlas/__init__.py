"""Plot per-region data on a brain atlas.

Public API (the names re-exported here):

    plot_on_atlas        fill Allen atlas regions with scalar or per-neuron
                         data, on a single section or the Swanson flat projection.
    plot_atlas_animation animate time-varying per-region data, returning a
                         matplotlib FuncAnimation.

Internal modules (reachable via their dotted paths):

    cache.py     on-disk caching + the single network entry point
    api.py       Allen RMA client (atlases, section images, SVG, ontology)
                 + the IBL Swanson polygon download
    svg.py       SVG ``d`` strings -> matplotlib Paths keyed by structure_id
    ontology.py  region name / acronym / id -> structure_id resolver
    cmap.py      one shared colormap + Normalize across mixed scalar/array data
    geom.py      Path -> shapely polygon + shape-masked layouts for the sorters
    fills.py     pluggable fill strategies (grid, SOM hex/rect, Voronoi)
    swanson.py   vectorized Swanson flat-projection geometry
    plot.py      plot_on_atlas (orchestrator, top-level entry)
    animation.py plot_atlas_animation (time-varying fills -> FuncAnimation)

Array fills reuse ``nengoplotlib.sorting`` (SOM, Voronoi), constraining the
neuron layout to each region's outline. The ``fills.py`` registry is the
extension point for further layouts.
"""

from .animation import plot_atlas_animation
from .fills import available_fills, register_fill
from .plot import plot_on_atlas

__all__ = [
    "plot_on_atlas",
    "plot_atlas_animation",
    "available_fills",
    "register_fill",
]
