"""Geometry bridges between a region outline and the neuron-layout sorters.

The fill strategies in :mod:`nengoplotlib.atlas.fills` that reuse
``nengoplotlib.sorting`` (SOM grids, Voronoi mosaics) need three things the
sorters don't speak natively:

* a shapely polygon of the region, to confine the layout to its shape;
* a way to seed a grid of points *inside* that polygon (for shape-constrained
  SOM cells); and
* an affine map of an arbitrary 2-D layout into the region's bounding box
  (the post-hoc fallback, and for placing externally-supplied positions).

These are small and atlas-specific, so they live here rather than in the
generic sorting package.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def path_to_polygon(path: Path):
    """Convert a (compound) matplotlib ``Path`` to a shapely (Multi)Polygon.

    Bezier segments are flattened by ``Path.to_polygons``. Each closed ring
    becomes a polygon; the largest is treated as the exterior and any ring it
    contains as a hole, while disjoint rings (e.g. left/right hemisphere) are
    unioned into a MultiPolygon. ``buffer(0)`` repairs self-intersections.
    """
    rings = [np.asarray(r) for r in path.to_polygons() if len(r) >= 3]
    if not rings:
        raise ValueError("path has no fillable rings")
    polys = []
    for r in rings:
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)
    if not polys:
        raise ValueError("path produced no valid polygons")
    # unary_union turns overlapping exterior/hole rings into proper polygons
    # with holes and merges adjacent pieces; disjoint pieces stay separate.
    merged = unary_union(polys)
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged
    # GeometryCollection -> keep only the polygonal parts.
    geoms = [g for g in getattr(merged, "geoms", []) if g.area > 0]
    return unary_union(geoms) if geoms else polys[0]


def grid_in_polygon(polygon, topology="hex", n_target=64, min_spacing_frac=0.01):
    """Return lattice centres that fall inside ``polygon``.

    A regular ``hex`` or ``rect`` lattice is laid over the polygon's bounding
    box at a spacing chosen so that roughly ``n_target`` centres land inside,
    then masked to those actually contained. Returns an ``(m, 2)`` array
    (``m`` near ``n_target``, exact count depends on the shape).
    """
    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny
    area = polygon.area
    if area <= 0 or n_target < 1:
        return np.empty((0, 2))
    # Spacing so that (area / cell_area) ~= n_target. Hex cell area = s^2*sqrt(3)/2,
    # square cell area = s^2; use the square estimate -- close enough, then mask.
    spacing = max(np.sqrt(area / n_target), min_spacing_frac * max(w, h))

    xs = np.arange(minx + spacing / 2, maxx, spacing)
    rows = np.arange(miny + spacing / 2, maxy, spacing * (np.sqrt(3) / 2
                                                          if topology == "hex" else 1.0))
    pts = []
    for i, y in enumerate(rows):
        offset = spacing / 2 if (topology == "hex" and i % 2) else 0.0
        for x in xs:
            pts.append((x + offset, y))
    if not pts:
        return np.empty((0, 2))
    pts = np.asarray(pts)
    from shapely.geometry import Point
    inside = np.array([polygon.contains(Point(p)) for p in pts])
    return pts[inside]


def map_into_bbox(points, bbox, pad_frac=0.0):
    """Affine-normalise ``points`` into ``bbox`` (``xmin, xmax, ymin, ymax``).

    Each axis is independently scaled so the points' extent fills the box
    (optionally inset by ``pad_frac``). Degenerate axes are centred.
    """
    points = np.asarray(points, dtype=float)
    xmin, xmax, ymin, ymax = bbox
    px, py = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
    xmin, xmax, ymin, ymax = xmin + px, xmax - px, ymin + py, ymax - py
    lo = points.min(axis=0)
    span = points.max(axis=0) - lo
    span[span == 0] = 1.0
    unit = (points - lo) / span
    out = np.empty_like(points)
    out[:, 0] = xmin + unit[:, 0] * (xmax - xmin)
    out[:, 1] = ymin + unit[:, 1] * (ymax - ymin)
    return out
