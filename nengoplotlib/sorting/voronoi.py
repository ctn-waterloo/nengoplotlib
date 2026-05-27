"""Voronoi-based parcellation of a 2D neuron layout.

Builds organic, variable-shape regions (one per neuron, or one per cluster
centroid) and returns them both as point coordinates and as a list of
``matplotlib.patches.Patch`` objects ready for a ``PatchCollection``.

The outer border is the alpha-shape (concave hull) of the points so the
parcellation hugs the cloud rather than the bounding box.

Two entry points:

* :func:`voronoi_parcellation` — one cell per input neuron.
* :func:`kmeans_voronoi_parcellation` — k-means in 2D, Voronoi over the
  centroids. Each cluster aggregates its members.

Both return a :class:`Parcellation` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi


@dataclass
class Parcellation:
    """Result of a Voronoi parcellation.

    Attributes
    ----------
    positions : (n_cells, 2) float array
        Anchor point for each cell (neuron position or cluster centroid).
    patches : list of matplotlib.patches.Polygon
        One filled-polygon patch per cell, in the same order as *positions*.
    cell_assignments : (n_neurons,) int array, optional
        For cluster-based parcellations, the cell index for every input
        neuron. ``None`` for per-neuron parcellations (each neuron is its
        own cell).
    bbox : (xmin, xmax, ymin, ymax)
        Tight bounding box used during construction; useful for axis limits.
    """

    positions: np.ndarray
    patches: List[mpatches.Polygon]
    cell_assignments: Optional[np.ndarray]
    bbox: tuple


# ---------------------------------------------------------------- alpha shape


def alpha_shape_loops(points, alpha):
    """Return list of boundary loops of the alpha-shape of *points*.

    For each Delaunay triangle, keep it if its circumradius < 1/alpha. The
    boundary of the union of kept triangles is the alpha shape.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return []
    if len(points) == 3:
        return [points]
    tri = Delaunay(points)
    keep = []
    for simplex in tri.simplices:
        pa, pb, pc = points[simplex]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2
        area = max(s * (s - a) * (s - b) * (s - c), 1e-18) ** 0.5
        circ_r = a * b * c / (4 * area)
        if circ_r < 1.0 / alpha:
            keep.append(simplex)
    if not keep:
        hull = ConvexHull(points)
        return [points[hull.vertices]]

    edge_count = {}
    for simplex in keep:
        for i in range(3):
            e = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary = [e for e, c in edge_count.items() if c == 1]

    adj = {}
    for a_, b_ in boundary:
        adj.setdefault(a_, []).append(b_)
        adj.setdefault(b_, []).append(a_)

    loops = []
    visited_edges = set()
    for start in list(adj):
        if not adj[start]:
            continue
        loop = [start]
        prev = None
        cur = start
        while True:
            nexts = [n for n in adj[cur] if n != prev]
            if not nexts:
                break
            nxt = nexts[0]
            edge = tuple(sorted((cur, nxt)))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            adj[cur].remove(nxt)
            adj[nxt].remove(cur)
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(points[loop])
    return loops


def alpha_shape_polygon(points, alpha, smooth_eps=None):
    """Return the alpha-shape of *points* as a shapely Polygon, holes dropped.

    Largest boundary loop becomes the exterior; interior holes are discarded
    so the clipping polygon is a single solid outer shape.

    When *smooth_eps* is given, the boundary is morphologically smoothed:
    first a closing (dilate-erode, fills inward notches), then an opening
    (erode-dilate, trims outward spikes). The result is a noticeably
    cleaner edge than either operation alone.
    """
    from shapely.geometry import Polygon as ShPoly

    loops = alpha_shape_loops(points, alpha=alpha)
    if not loops:
        return None
    loops_sorted = sorted(loops, key=lambda l: ShPoly(l).area, reverse=True)
    exterior = loops_sorted[0]
    poly = ShPoly(exterior)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if smooth_eps:
        try:
            # Closing pass uses 2x eps: deep inward notches between adjacent
            # generators span ~2 d_nn, so a generous dilation+erosion is what
            # actually bridges them. Round joins keep the result organic.
            close_eps = 2.0 * smooth_eps
            closed = poly.buffer(close_eps, join_style=1).buffer(
                -close_eps, join_style=1
            )
            if not closed.is_empty:
                poly = closed
            # Opening uses the smaller eps — just trims outward spikes and
            # softens the convex corners introduced by the closing pass.
            opened = poly.buffer(-smooth_eps, join_style=1).buffer(
                smooth_eps, join_style=1
            )
            if not opened.is_empty:
                poly = opened
        except Exception:
            pass
    if hasattr(poly, "exterior") and poly.exterior is not None:
        return ShPoly(poly.exterior)
    if hasattr(poly, "geoms"):
        biggest = max(poly.geoms, key=lambda g: g.area)
        return ShPoly(biggest.exterior)
    return poly


# ------------------------------------------------------------ bounded Voronoi


def _sutherland_hodgman(poly, bbox):
    """Clip a polygon to an axis-aligned bounding box."""
    xmin, xmax, ymin, ymax = bbox
    edges = [("xmin", xmin), ("xmax", xmax), ("ymin", ymin), ("ymax", ymax)]

    def inside(p, edge, val):
        if edge == "xmin":
            return p[0] >= val
        if edge == "xmax":
            return p[0] <= val
        if edge == "ymin":
            return p[1] >= val
        return p[1] <= val

    def intersect(a, b, edge, val):
        if edge in ("xmin", "xmax"):
            t = (val - a[0]) / (b[0] - a[0] + 1e-12)
            return np.array([val, a[1] + t * (b[1] - a[1])])
        t = (val - a[1]) / (b[1] - a[1] + 1e-12)
        return np.array([a[0] + t * (b[0] - a[0]), val])

    out = list(poly)
    for edge, val in edges:
        if not out:
            break
        new = []
        for i, p in enumerate(out):
            prev = out[i - 1]
            if inside(p, edge, val):
                if not inside(prev, edge, val):
                    new.append(intersect(prev, p, edge, val))
                new.append(p)
            elif inside(prev, edge, val):
                new.append(intersect(prev, p, edge, val))
        out = new
    return np.array(out) if out else None


def _shapely_intersect(poly_verts, clip_shape, generator, disk_radius):
    from shapely.geometry import MultiPolygon, Point, Polygon as ShPoly

    p = ShPoly(poly_verts)
    if not p.is_valid:
        p = p.buffer(0)
    if clip_shape is not None:
        p = p.intersection(clip_shape)
    if disk_radius is not None:
        disk = Point(generator).buffer(disk_radius, quad_segs=24)
        p = p.intersection(disk)
    if p.is_empty:
        return None
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    if not hasattr(p, "exterior"):
        return None
    return np.array(p.exterior.coords)


def _clip_halfplane(poly, origin, inward_normal):
    """Sutherland-Hodgman against a single half-plane.

    Keeps the side of the plane where ``(p - origin) . inward_normal >= 0``.
    """
    if poly is None or len(poly) < 3:
        return None

    def inside(p):
        return np.dot(p - origin, inward_normal) >= -1e-12

    def intersect(a, b):
        da = np.dot(a - origin, inward_normal)
        db = np.dot(b - origin, inward_normal)
        t = da / (da - db + 1e-12)
        return a + t * (b - a)

    out = list(poly)
    new = []
    for i, p in enumerate(out):
        prev = out[i - 1]
        if inside(p):
            if not inside(prev):
                new.append(intersect(prev, p))
            new.append(p)
        elif inside(prev):
            new.append(intersect(prev, p))
    return np.array(new) if len(new) >= 3 else None


def _hull_outward_normals(points):
    """For each convex-hull vertex, return its outward unit normal.

    The normal at vertex v is the (normalized) sum of the outward normals of
    the two hull edges incident to v. ``ConvexHull(...).vertices`` returns
    hull indices in counterclockwise order, so the outward normal of edge
    ``(a, b)`` is ``rotate(b - a, -90deg)`` normalized.
    """
    hull = ConvexHull(points)
    hv = hull.vertices  # CCW order
    edge_normals = []
    for i in range(len(hv)):
        a = points[hv[i]]
        b = points[hv[(i + 1) % len(hv)]]
        tangent = b - a
        # rotate -90 to get the outward normal of a CCW polygon
        n = np.array([tangent[1], -tangent[0]])
        n /= max(np.linalg.norm(n), 1e-12)
        edge_normals.append(n)
    out = {}
    for i, vidx in enumerate(hv):
        n = edge_normals[(i - 1) % len(hv)] + edge_normals[i]
        n /= max(np.linalg.norm(n), 1e-12)
        out[int(vidx)] = n
    return out


def _flat_cap_frame(points, d):
    """Build the polygonal frame whose edges are the per-hull-vertex flat
    caps. Adjacent cap lines are intersected to form frame vertices.

    Returns a shapely Polygon (CCW) ready to be used as a clip shape, plus
    the per-vertex (origin, outward_normal) cap data.
    """
    from shapely.geometry import Polygon as ShPoly

    hull = ConvexHull(points)
    hv = hull.vertices  # CCW
    n = len(hv)
    normals = _hull_outward_normals(points)

    caps = []  # (origin_on_cap_line, outward_unit_normal)
    for vidx in hv:
        v = points[vidx]
        n_vec = normals[int(vidx)]
        caps.append((v + d * n_vec, n_vec))

    frame_pts = []
    for i in range(n):
        o1, n1 = caps[i]
        o2, n2 = caps[(i + 1) % n]
        A = np.array([n1, n2])
        rhs = np.array([np.dot(n1, o1), np.dot(n2, o2)])
        try:
            p = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        frame_pts.append(p)
    if len(frame_pts) < 3:
        return None
    poly = ShPoly(frame_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def bounded_voronoi(points, bbox, clip_shape=None, disk_radius=None,
                    flat_caps=False, flat_cap_distance=None):
    """Return one finite polygon (as a vertex array) per input point.

    Each raw Voronoi region is intersected with *bbox*, with the optional
    shapely *clip_shape*, and (optionally) with a disk of *disk_radius*
    around its generator.

    When *flat_caps* is True, each convex-hull generator's cell is also
    clipped by a half-plane perpendicular to the hull's local outward
    normal at distance *flat_cap_distance* from the generator. Interior
    cells are unaffected. This replaces curvy alpha-clipping with one
    clean flat edge per border cell.
    """
    points = np.asarray(points, dtype=float)
    xmin, xmax, ymin, ymax = bbox
    far = max(xmax - xmin, ymax - ymin) * 10
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    extra = np.array(
        [[cx - far, cy], [cx + far, cy], [cx, cy - far], [cx, cy + far]]
    )
    vor = Voronoi(np.vstack([points, extra]))

    # When flat_caps is True, build a polygonal frame whose edges are the
    # per-hull-vertex cap lines. Every border cell is then clipped against
    # this frame, gaining one flat outer edge — or two, when the cell sits
    # at a corner (its Voronoi region crosses the join between two cap
    # lines). Replaces the alpha-shape clip in this mode.
    if flat_caps and flat_cap_distance is not None and len(points) >= 3:
        frame = _flat_cap_frame(points, flat_cap_distance)
        if frame is not None:
            clip_shape = (frame if clip_shape is None
                          else clip_shape.intersection(frame))

    polys = []
    for i, gen in enumerate(points):
        region_idx = vor.point_region[i]
        verts = vor.regions[region_idx]
        if not verts or -1 in verts:
            polys.append(None)
            continue
        poly = _sutherland_hodgman(vor.vertices[verts], bbox)
        if poly is None or len(poly) < 3:
            polys.append(None)
            continue
        if clip_shape is not None or disk_radius is not None:
            poly = _shapely_intersect(poly, clip_shape, gen, disk_radius)
            if poly is None or len(poly) < 3:
                polys.append(None)
                continue
        polys.append(poly)
    return polys


def median_nn_distance(points):
    """Median nearest-neighbor distance — a natural length-scale."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 1.0
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    return float(np.median(d[:, 1]))


def bbox_of(points, pad_frac=0.05):
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = (hi - lo) * pad_frac
    return (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])


# ------------------------------------------------------------------- entry pts


def voronoi_parcellation(
    positions,
    *,
    outer="alpha",
    alpha_factor=3.0,
    flat_cap_factor=0.7,
    disk_factor=None,
    smooth_eps_factor=0.8,
):
    """Per-neuron Voronoi parcellation of points in a 2D layout.

    Parameters
    ----------
    positions : (n, 2) array
        2D coordinates of every neuron.
    outer : {'alpha', 'flat'}
        How to handle border cells whose natural Voronoi region is
        unbounded:

        - ``'alpha'`` (default) clips every cell to a smoothed alpha-shape
          of the cloud — organic curvy outer boundary.
        - ``'flat'`` gives each convex-hull generator a single flat outer
          edge perpendicular to the local outward normal at distance
          ``flat_cap_factor * d_nn`` from the generator.
    alpha_factor : float
        Alpha-shape parameter as a multiplier of the median nearest-neighbor
        distance: ``alpha = 1 / (alpha_factor * d_nn)``. Larger value =
        looser hull. Only used when ``outer='alpha'``.
    flat_cap_factor : float
        Distance from each hull generator at which to place the flat outer
        edge, in units of d_nn. Only used when ``outer='flat'``.
    disk_factor : float or None
        If set, additionally cap each Voronoi cell to a disk of radius
        ``disk_factor * d_nn`` around its generator. ``None`` lets cells
        fill the outer boundary continuously.
    smooth_eps_factor : float
        Corner-smoothing for the alpha hull, in units of d_nn. Set to 0 to
        disable. Only applied when ``outer='alpha'``.

    Returns
    -------
    Parcellation
    """
    if outer not in ("alpha", "flat"):
        raise ValueError(f"outer must be 'alpha' or 'flat', got {outer!r}")
    positions = np.asarray(positions, dtype=float)
    bbox = bbox_of(positions)
    d_nn = median_nn_distance(positions)
    clip_shape = None
    flat_caps = False
    flat_cap_distance = None
    if outer == "alpha":
        alpha = 1.0 / (alpha_factor * d_nn)
        smooth_eps = smooth_eps_factor * d_nn if smooth_eps_factor else None
        clip_shape = alpha_shape_polygon(
            positions, alpha=alpha, smooth_eps=smooth_eps
        )
    else:  # flat
        flat_caps = True
        flat_cap_distance = flat_cap_factor * d_nn
    disk_radius = disk_factor * d_nn if disk_factor is not None else None

    polys = bounded_voronoi(
        positions, bbox, clip_shape=clip_shape, disk_radius=disk_radius,
        flat_caps=flat_caps, flat_cap_distance=flat_cap_distance,
    )
    patches = [
        mpatches.Polygon(p, closed=True) if p is not None and len(p) >= 3
        else None
        for p in polys
    ]
    return Parcellation(
        positions=positions,
        patches=patches,
        cell_assignments=None,
        bbox=bbox,
    )


def kmeans_voronoi_parcellation(
    positions,
    n_clusters,
    *,
    outer="alpha",
    alpha_factor=3.0,
    flat_cap_factor=0.7,
    disk_factor=None,
    smooth_eps_factor=0.8,
    random_state=0,
):
    """Cluster points in 2D with k-means, then Voronoi over centroids.

    Parameters
    ----------
    positions : (n, 2) array
    n_clusters : int
    outer, alpha_factor, flat_cap_factor, disk_factor, smooth_eps_factor
        See :func:`voronoi_parcellation`. The alpha hull uses the per-point
        d_nn (so it tracks the underlying cloud, not the centroid lattice);
        ``flat_cap_factor`` and ``disk_factor`` use the inter-centroid d_nn.
    random_state : int

    Returns
    -------
    Parcellation
        ``positions`` is the (n_clusters, 2) centroids; ``patches`` is one
        Polygon per cluster; ``cell_assignments`` is the per-neuron cluster
        index of length n.
    """
    from sklearn.cluster import KMeans

    if outer not in ("alpha", "flat"):
        raise ValueError(f"outer must be 'alpha' or 'flat', got {outer!r}")

    positions = np.asarray(positions, dtype=float)
    km = KMeans(n_clusters=n_clusters, n_init=10,
                random_state=random_state).fit(positions)
    labels = km.labels_
    centroids = km.cluster_centers_

    bbox = bbox_of(positions)
    d_nn_points = median_nn_distance(positions)
    d_nn_centroids = median_nn_distance(centroids)
    clip_shape = None
    flat_caps = False
    flat_cap_distance = None
    if outer == "alpha":
        alpha = 1.0 / (alpha_factor * d_nn_points)
        smooth_eps = (smooth_eps_factor * d_nn_points
                      if smooth_eps_factor else None)
        clip_shape = alpha_shape_polygon(
            positions, alpha=alpha, smooth_eps=smooth_eps
        )
    else:
        flat_caps = True
        flat_cap_distance = flat_cap_factor * d_nn_centroids
    disk_radius = (disk_factor * d_nn_centroids
                   if disk_factor is not None else None)

    polys = bounded_voronoi(
        centroids, bbox, clip_shape=clip_shape, disk_radius=disk_radius,
        flat_caps=flat_caps, flat_cap_distance=flat_cap_distance,
    )
    patches = [
        mpatches.Polygon(p, closed=True) if p is not None and len(p) >= 3
        else None
        for p in polys
    ]
    return Parcellation(
        positions=centroids,
        patches=patches,
        cell_assignments=labels,
        bbox=bbox,
    )
