"""``plot_on_atlas`` -- paint per-region data onto a brain atlas.

Pipeline::

    resolve_atlas -> Ontology (structure graph)
                  -> geometry  ({structure_id: Path})
                       sliced : download + parse one section's SVG
                       swanson: vectorized flat projection
                  -> draw every outline (black, unfilled)
                  -> shared colormap from all data values
                  -> per region: resolve name -> id(s) -> scalar or array fill
                  -> optional colorbar
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection

from . import api, cmap as cmap_mod, fills, geom, swanson
from .ontology import Ontology
from .svg import parse_svg


def _is_scalar(v) -> bool:
    arr = np.asarray(v, dtype=float)
    return arr.ndim == 0 or arr.size == 1


def _safe_polygon(path):
    """Region outline as a shapely polygon, or ``None`` if it can't be built.

    ``None`` lets the SOM/Voronoi fills fall back to a grid fill rather than
    error on a degenerate outline.
    """
    try:
        return geom.path_to_polygon(path)
    except Exception:
        return None


def load_atlas(atlas, *, section=None, swanson=False, refresh=False):
    """Resolve the ontology and region geometry for an atlas view.

    Shared by :func:`plot_on_atlas` and
    :func:`nengoplotlib.atlas.animation.plot_atlas_animation`.

    Returns
    -------
    (ontology, geometry, invert_y)
        ``geometry`` maps ``structure_id`` to a matplotlib ``Path``;
        ``invert_y`` is ``True`` for SVG sections (y increases downward).
    """
    record = api.resolve_atlas(atlas, refresh=refresh)
    graph_id = record.get("structure_graph_id")
    if graph_id is None:
        raise ValueError(
            f"atlas {record['name']!r} has no structure graph (not supported)"
        )
    ont = Ontology.for_graph(int(graph_id), refresh=refresh)

    if swanson:
        return ont, _swanson_geometry(refresh), False
    images = api.list_atlas_images(record["id"], refresh=refresh)
    image = _pick_section(images, section)
    geometry = parse_svg(api.download_svg(image["id"], refresh=refresh))
    return ont, geometry, True  # SVG y increases downward


def _pick_section(images: list, section: Optional[Union[int, float]]) -> dict:
    """Choose one ``AtlasImage`` record from the annotated list.

    ``section`` may be an ``AtlasImage`` id, a ``section_number`` (the
    anatomical coordinate), or ``None`` for the middle section. Anything else
    falls back to the nearest section number.
    """
    if not images:
        raise RuntimeError("atlas has no annotated section images")
    if section is None:
        return images[len(images) // 2]
    ids = {im["id"]: im for im in images}
    if section in ids:
        return ids[section]
    nums = np.array([im["section_number"] for im in images])
    exact = np.where(nums == section)[0]
    if exact.size:
        return images[int(exact[0])]
    return images[int(np.argmin(np.abs(nums - section)))]


def _match_geometry(ont: Ontology, geom_keys, structure_id: int) -> list:
    """Geometry ids to fill for a requested ``structure_id``.

    Returns the region itself plus any drawn descendants, so a value given for
    a coarse region (e.g. ``"Isocortex"``) propagates down to every finer
    subregion that is actually drawn. A region absent from the current view
    (e.g. not present in this section) matches nothing and is skipped -- we
    deliberately do *not* climb to an enclosing container, which would smear
    one region's data across unrelated territory.
    """
    return sorted(ont.descendants(structure_id) & set(geom_keys))


def plot_on_atlas(
    atlas: Union[int, str],
    data: Mapping,
    *,
    section: Optional[Union[int, float]] = None,
    swanson: bool = False,
    array_fill_type: str = "pcolormesh",
    features: Optional[Mapping] = None,
    positions: Optional[Mapping] = None,
    cmap="viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edgecolor: str = "k",
    outline_lw: float = 0.5,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    ax=None,
    refresh: bool = False,
    **fill_kwargs,
):
    """Fill brain regions on an Allen atlas with scalar or per-neuron data.

    Parameters
    ----------
    atlas : int or str
        Allen atlas id (e.g. ``1``) or name (e.g. ``"Mouse, P56, Coronal"``,
        matched case-insensitively, substring allowed).
    data : dict
        Maps region names -- acronyms (``"VISp"``) or full names (``"Primary
        visual area"``) or integer structure ids -- to either a scalar (solid
        fill) or a ``(n_neurons,)`` array (laid out and clipped to the region
        via ``array_fill_type``). A value given for a coarse region propagates
        to its drawn subregions.
    section : int or float, optional
        Which slice to draw (ignored when ``swanson=True``): an ``AtlasImage``
        id, a ``section_number``, or ``None`` for the middle section.
    swanson : bool
        Plot the Swanson flat projection instead of a single slice. Scalar
        fills only; arrays raise ``NotImplementedError``.
    array_fill_type : str
        Strategy for array values. Grid fills: ``"pcolormesh"`` (default),
        ``"pcolor"``. Neuron-layout fills constrained to the region shape:
        ``"som_hex"``, ``"som_rect"``, ``"voronoi"``, ``"voronoi_kmeans"``.
        See :func:`nengoplotlib.atlas.fills.available_fills`.
    features : dict, optional
        ``{region: (n_neurons, k)}`` feature matrices that drive the SOM /
        Voronoi layout. When omitted, the plotted values are used as a single
        feature (so neighbouring cells form a smooth gradient).
    positions : dict, optional
        ``{region: (n_neurons, 2)}`` explicit 2-D layouts for the Voronoi
        fills, affine-mapped into the region's bounding box. When omitted, a
        shape-masked layout ordered by feature is generated.
    cmap : str or Colormap
    vmin, vmax : float, optional
        Override the data-derived colour limits.
    edgecolor : str
        Outline colour for region boundaries.
    colorbar : bool
        Draw a colorbar for the shared colour scale.
    ax : matplotlib axes, optional
    refresh : bool
        Bypass the on-disk cache and re-download atlas data.
    **fill_kwargs
        Forwarded to the fill strategy, e.g. ``n_clusters`` (voronoi_kmeans),
        ``random_state``/``n_iter`` (SOM), or ``alpha`` (mesh fills).

    Returns
    -------
    (fig, ax, artists)
        ``artists`` maps each filled ``structure_id`` to its matplotlib artist.
    """
    ont, geometry, invert_y = load_atlas(
        atlas, section=section, swanson=swanson, refresh=refresh)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    fig = ax.figure

    # 1) every region outline, black, no fill.
    ax.add_collection(
        PathCollection(list(geometry.values()), facecolors="none",
                       edgecolors=edgecolor, linewidths=outline_lw)
    )

    # 2) one shared colour scale across all scalars + arrays.
    colormap, norm = cmap_mod.build_norm(data, cmap=cmap, vmin=vmin, vmax=vmax)

    # 3) fill each requested region.
    artists = {}
    geom_keys = set(geometry)
    for region, value in data.items():
        try:
            sid = ont.resolve(region)
        except KeyError as exc:
            warnings.warn(str(exc))
            continue
        targets = _match_geometry(ont, geom_keys, sid)
        if not targets:
            warnings.warn(f"region {region!r} (id {sid}) not drawn in this view")
            continue
        scalar = _is_scalar(value)
        if not scalar and swanson:
            raise NotImplementedError(
                "array fills are not supported on the Swanson view yet; "
                "pass a scalar or use a sliced atlas"
            )
        region_feats = (features or {}).get(region)
        region_pos = (positions or {}).get(region)
        for gid in targets:
            path = geometry[gid]
            if scalar:
                artists[gid] = fills.scalar_fill(
                    ax, path, float(np.asarray(value).ravel()[0]),
                    colormap, norm, edgecolor=edgecolor, **fill_kwargs)
            else:
                artists[gid] = fills.array_fill(
                    ax, path, value, colormap, norm,
                    array_fill_type=array_fill_type, edgecolor=edgecolor,
                    polygon=_safe_polygon(path), features=region_feats,
                    positions=region_pos, **fill_kwargs)

    # 4) frame the plot.
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()

    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)

    return fig, ax, artists


def _swanson_geometry(refresh: bool):
    # Small indirection so the name ``swanson`` (the module) is not shadowed
    # by the ``swanson`` boolean parameter inside ``plot_on_atlas``.
    return swanson.region_paths(refresh=refresh)
