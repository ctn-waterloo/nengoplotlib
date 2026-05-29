"""How a region's value is painted onto its outline.

Scalars are easy: one flat colour. A ``(n_neurons,)`` array is the interesting
case -- those values have to be *laid out* somehow inside an irregular polygon
and clipped to it. There are many reasonable layouts (a grid, a hex SOM,
Voronoi parcels, ...), so the array case goes through a small **registry** of
named strategies. ``array_fill_type`` picks one.

Strategies that ship:

* ``"pcolormesh"`` / ``"pcolor"`` -- reshape the vector into a near-square grid
  over the region's bounding box and clip it to the outline.
* ``"som_hex"`` / ``"som_rect"`` -- train a self-organizing map whose cells are
  a hex/rect lattice *inside the region polygon*, assign each neuron to a cell,
  and colour each cell (a clipped Voronoi tile of its centre) by the mean of
  its neurons' values.
* ``"voronoi"`` / ``"voronoi_kmeans"`` -- an organic Voronoi mosaic confined to
  the region: one cell per neuron, or one per k-means cluster.

The SOM/Voronoi strategies reuse ``nengoplotlib.sorting`` (``SOM``,
``voronoi_parcellation``, ``kmeans_voronoi_parcellation``, ``bounded_voronoi``).
Adding another layout means registering one more function -- the orchestrator
never changes.

A strategy has signature::

    fn(ax, path, values, cmap, norm, **opts) -> matplotlib artist

and is responsible for clipping its own artist to ``path`` (or to the shapely
``polygon`` passed through ``opts``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path

_STRATEGIES: Dict[str, Callable] = {}


def register_fill(name: str) -> Callable:
    """Decorator registering an array-fill strategy under ``name``."""
    def deco(fn: Callable) -> Callable:
        _STRATEGIES[name] = fn
        return fn
    return deco


def available_fills() -> list:
    return sorted(_STRATEGIES)


# kwargs meaningful only to the SOM/Voronoi strategies. A single call site
# (or the example) may pass these alongside any ``array_fill_type``, so the
# scalar and grid fills drop them rather than forwarding to a matplotlib artist.
_SORT_ONLY = ("polygon", "features", "positions", "random_state", "n_iter",
              "n_clusters", "metric", "n_cells")


def _drop_sort_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in _SORT_ONLY}


def scalar_fill(ax, path: Path, value: float, cmap, norm, *,
                edgecolor="k", lw=0.5, **kwargs):
    """Fill a region with a single solid colour from the shared colormap."""
    patch = PathPatch(path, facecolor=cmap(norm(value)), edgecolor=edgecolor,
                      lw=lw, **_drop_sort_kwargs(kwargs))
    ax.add_patch(patch)
    return patch


def _grid_geometry(n: int, path: Path):
    """Fixed near-square grid over ``path``'s bbox for ``n`` values.

    Returns ``(rows, cols, x_edges, y_edges, inside)`` where ``inside`` is the
    ``(rows, cols)`` boolean mask of cells whose centre is within the outline.
    Geometry only -- computed once, reused for every animation frame.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    (x0, y0), (x1, y1) = path.get_extents().get_points()
    x_edges = np.linspace(x0, x1, cols + 1)
    y_edges = np.linspace(y0, y1, rows + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    XC, YC = np.meshgrid(xc, yc)
    inside = path.contains_points(
        np.column_stack([XC.ravel(), YC.ravel()])).reshape(rows, cols)
    return rows, cols, x_edges, y_edges, inside


def _grid_values(values, rows, cols, inside):
    """Reshape ``values`` into the grid, masking outside / non-finite cells."""
    values = np.asarray(values, dtype=float).ravel()
    grid = np.full(rows * cols, np.nan)
    grid[:values.size] = values
    grid = grid.reshape(rows, cols)
    return np.ma.array(grid, mask=~inside | ~np.isfinite(grid))


def _grid_layout(values: np.ndarray, path: Path):
    """``(x_edges, y_edges, C)`` ready for ``pcolormesh``/``pcolor``."""
    values = np.asarray(values, dtype=float).ravel()
    rows, cols, x_edges, y_edges, inside = _grid_geometry(values.size, path)
    return x_edges, y_edges, _grid_values(values, rows, cols, inside)


def _mesh_fill(ax, path, values, cmap, norm, *, kind, edgecolor=None, **kwargs):
    x_edges, y_edges, C = _grid_layout(values, path)
    fn = ax.pcolormesh if kind == "pcolormesh" else ax.pcolor
    mesh = fn(x_edges, y_edges, C, cmap=cmap, norm=norm, **kwargs)
    # Clip the rectangular mesh to the (irregular) region outline.
    mesh.set_clip_path(path, ax.transData)
    if edgecolor is not None:
        # Trace the outline on top so clipped meshes still show a border.
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=edgecolor, lw=0.5))
    return mesh


@register_fill("pcolormesh")
def _fill_pcolormesh(ax, path, values, cmap, norm, **kwargs):
    return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                      **_drop_sort_kwargs(kwargs))


@register_fill("pcolor")
def _fill_pcolor(ax, path, values, cmap, norm, **kwargs):
    return _mesh_fill(ax, path, values, cmap, norm, kind="pcolor",
                      **_drop_sort_kwargs(kwargs))


# ----------------------------------------------------- SOM / Voronoi layouts


def _resolve_features(values, features):
    """Per-neuron feature matrix; default to the plotted values as 1 feature."""
    if features is not None:
        f = np.asarray(features, dtype=float)
        return f[:, None] if f.ndim == 1 else f
    return np.asarray(values, dtype=float).reshape(-1, 1)


def _cell_means(values, assign, n_cells):
    """Mean value per cell index; NaN for cells with no assigned neurons."""
    sums = np.zeros(n_cells)
    counts = np.zeros(n_cells)
    np.add.at(sums, assign, values)
    np.add.at(counts, assign, 1.0)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)


def _feature_order(feats):
    """1-D ordering of neurons (by first principal component of features)."""
    if feats.shape[1] == 1:
        key = feats[:, 0]
    else:
        from sklearn.decomposition import PCA
        key = PCA(n_components=1, random_state=0).fit_transform(feats)[:, 0]
    return np.argsort(key, kind="stable")


def _rejection_sample(polygon, k, rng, max_tries=20000):
    from shapely.geometry import Point
    minx, miny, maxx, maxy = polygon.bounds
    out = []
    tries = 0
    while len(out) < k and tries < max_tries:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            out.append((x, y))
        tries += 1
    return np.asarray(out) if out else np.empty((0, 2))


def _points_in_polygon(polygon, n, rng):
    """Exactly ``n`` points inside ``polygon`` (lattice, topped up at random)."""
    from .geom import grid_in_polygon
    pts = grid_in_polygon(polygon, topology="hex", n_target=int(n * 1.3) + 8)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    elif len(pts) < n:
        extra = _rejection_sample(polygon, n - len(pts), rng)
        if len(extra):
            pts = np.vstack([pts, extra]) if len(pts) else extra
    return pts


def _layout_positions(polygon, n, feats, positions, rng):
    """Resolve one 2-D generator per neuron, inside the region.

    Uses supplied ``positions`` (mapped into the region's bbox) when given;
    otherwise lays neurons on a shape-masked point set ordered by feature so
    neighbouring generators carry similar values (a smooth gradient).
    """
    from .geom import map_into_bbox
    minx, miny, maxx, maxy = polygon.bounds
    bbox = (minx, maxx, miny, maxy)
    if positions is not None:
        return map_into_bbox(np.asarray(positions, dtype=float), bbox)
    pts = _points_in_polygon(polygon, n, rng)
    m = min(n, len(pts))
    if m < 3:
        return None
    out = np.empty((n, 2))
    order_neuron = _feature_order(feats)            # neurons sorted by feature
    order_point = np.lexsort((pts[:, 0], pts[:, 1]))  # points row-major
    out[order_neuron[:m]] = pts[order_point[:m]]
    if m < n:  # leftover neurons (couldn't sample enough): reuse interior pts
        out[order_neuron[m:]] = pts[rng.integers(0, len(pts), size=n - m)]
    return out


def _render_cells(ax, cells, values, cmap, norm, edgecolor, lw=0.0):
    """Add a PatchCollection of polygon ``cells`` coloured by ``values``.

    ``cells`` items may be matplotlib Polygons or ``(k, 2)`` vertex arrays;
    ``None`` cells and non-finite values are dropped. Faces are coloured via
    the shared ``cmap``/``norm`` so the figure colorbar stays meaningful.
    """
    patches, face_vals = [], []
    for cell, v in zip(cells, values):
        if cell is None or not np.isfinite(v):
            continue
        if isinstance(cell, MplPolygon):
            patches.append(cell)
        else:
            arr = np.asarray(cell)
            if len(arr) < 3:
                continue
            patches.append(MplPolygon(arr, closed=True))
        face_vals.append(v)
    if not patches:
        return None
    coll = PatchCollection(patches, edgecolors=(edgecolor or "none"), linewidths=lw)
    coll.set_cmap(cmap)
    coll.set_norm(norm)
    coll.set_array(np.asarray(face_vals))
    ax.add_collection(coll)
    return coll


def _outline(ax, path, edgecolor):
    if edgecolor is not None:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=edgecolor, lw=0.5))


def _som_fill(ax, path, values, cmap, norm, *, topology, polygon=None,
              features=None, edgecolor="k", random_state=0, n_iter=1000,
              n_cells=None, metric="euclidean", **kwargs):
    """SOM whose cells tile the region; each cell coloured by its neuron mean."""
    from ..sorting.som import SOM
    from ..sorting.voronoi import bounded_voronoi
    from .geom import grid_in_polygon

    values = np.asarray(values, dtype=float).ravel()
    if polygon is None:
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)
    feats = _resolve_features(values, features)
    target = n_cells or min(len(values), 400)
    centers = grid_in_polygon(polygon, topology=topology, n_target=target)
    if len(centers) < 4:
        warnings.warn(
            "region too small for a SOM grid; falling back to pcolormesh fill")
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)

    som = SOM(grid_shape=(1, len(centers)), topology=topology,
              cell_centers=centers, metric=metric, n_iter=n_iter,
              random_state=random_state).fit(feats)
    assign = som.cell_assignments(feats)
    cell_vals = _cell_means(values, assign, len(centers))

    minx, miny, maxx, maxy = polygon.bounds
    polys = bounded_voronoi(centers, bbox=(minx, maxx, miny, maxy),
                            clip_shape=polygon)
    coll = _render_cells(ax, polys, cell_vals, cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)
    return coll


def _voronoi_fill(ax, path, values, cmap, norm, *, polygon=None, features=None,
                  positions=None, edgecolor="k", random_state=0, **kwargs):
    """One Voronoi cell per neuron, clipped to the region."""
    from ..sorting.voronoi import voronoi_parcellation

    values = np.asarray(values, dtype=float).ravel()
    if polygon is None:
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)
    rng = np.random.default_rng(random_state)
    pos = _layout_positions(polygon, len(values),
                            _resolve_features(values, features), positions, rng)
    if pos is None or len(pos) < 3:
        warnings.warn(
            "region too small for a Voronoi mosaic; falling back to pcolormesh")
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)

    minx, miny, maxx, maxy = polygon.bounds
    parc = voronoi_parcellation(pos, clip_shape=polygon,
                                bbox=(minx, maxx, miny, maxy))
    coll = _render_cells(ax, parc.patches, values, cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)
    return coll


def _voronoi_kmeans_fill(ax, path, values, cmap, norm, *, polygon=None,
                         features=None, positions=None, n_clusters=None,
                         edgecolor="k", random_state=0, **kwargs):
    """k-means clusters of the layout, one Voronoi cell per cluster."""
    from ..sorting.voronoi import kmeans_voronoi_parcellation

    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if polygon is None:
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)
    if n_clusters is None:
        n_clusters = max(2, int(round(np.sqrt(n))))
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(random_state)
    pos = _layout_positions(polygon, n,
                            _resolve_features(values, features), positions, rng)
    if pos is None or len(pos) < n_clusters or n_clusters < 2:
        warnings.warn(
            "region too small for a k-means Voronoi mosaic; "
            "falling back to pcolormesh")
        return _mesh_fill(ax, path, values, cmap, norm, kind="pcolormesh",
                          edgecolor=edgecolor)

    minx, miny, maxx, maxy = polygon.bounds
    parc = kmeans_voronoi_parcellation(
        pos, n_clusters=n_clusters, clip_shape=polygon,
        bbox=(minx, maxx, miny, maxy), random_state=random_state)
    cell_vals = _cell_means(values, parc.cell_assignments, n_clusters)
    coll = _render_cells(ax, parc.patches, cell_vals, cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)
    return coll


@register_fill("som_hex")
def _fill_som_hex(ax, path, values, cmap, norm, **kwargs):
    return _som_fill(ax, path, values, cmap, norm, topology="hex", **kwargs)


@register_fill("som_rect")
def _fill_som_rect(ax, path, values, cmap, norm, **kwargs):
    return _som_fill(ax, path, values, cmap, norm, topology="rect", **kwargs)


@register_fill("voronoi")
def _fill_voronoi(ax, path, values, cmap, norm, **kwargs):
    return _voronoi_fill(ax, path, values, cmap, norm, **kwargs)


@register_fill("voronoi_kmeans")
def _fill_voronoi_kmeans(ax, path, values, cmap, norm, **kwargs):
    return _voronoi_kmeans_fill(ax, path, values, cmap, norm, **kwargs)


def array_fill(ax, path: Path, values, cmap, norm, *,
               array_fill_type="pcolormesh", **kwargs):
    """Dispatch a ``(n_neurons,)`` array to the chosen fill strategy."""
    try:
        fn = _STRATEGIES[array_fill_type]
    except KeyError:
        raise ValueError(
            f"unknown array_fill_type {array_fill_type!r}; "
            f"available: {available_fills()}"
        )
    return fn(ax, path, values, cmap, norm, **kwargs)


# --------------------------------------------------------------- animatable fills
#
# The static fills above bake one timestep's colours into their artists. For an
# animation we instead build the *geometry* once and recolour every frame. A
# builder returns a :class:`Fill` whose ``set_values(per_neuron_or_scalar)``
# updates the colours in place. The layout helpers (``_grid_geometry``,
# ``_layout_positions``, ``grid_in_polygon``, ``bounded_voronoi``, ...) are the
# same ones the static path uses, so a frame looks identical to the equivalent
# ``plot_on_atlas`` call.


@dataclass
class Fill:
    """An updatable region fill: an artist plus a recolour closure."""
    primary: object                 # the matplotlib artist added to the axes
    apply: Callable                 # apply(values) -> recolours ``primary``

    def set_values(self, values):
        self.apply(values)


def build_scalar_fill(ax, path, cmap, norm, *, edgecolor="k", lw=0.5):
    """A solid-fill patch whose colour tracks a scalar value over frames."""
    patch = PathPatch(path, facecolor=cmap(norm(0.0)), edgecolor=edgecolor, lw=lw)
    ax.add_patch(patch)

    def apply(value):
        patch.set_facecolor(cmap(norm(float(np.ravel(value)[0]))))

    return Fill(patch, apply)


def _build_cells(ax, patches, cmap, norm, edgecolor, lw=0.0):
    """PatchCollection over fixed ``patches``; ``apply`` sets per-cell values."""
    coll = PatchCollection(list(patches), edgecolors=(edgecolor or "none"),
                           linewidths=lw)
    coll.set_cmap(cmap)
    coll.set_norm(norm)
    coll.set_array(np.zeros(len(patches)))
    ax.add_collection(coll)
    return coll


def _build_mesh(ax, path, n, cmap, norm, *, kind, edgecolor):
    rows, cols, x_edges, y_edges, inside = _grid_geometry(n, path)
    fn = ax.pcolormesh if kind == "pcolor" else ax.pcolormesh  # QuadMesh either way
    mesh = fn(x_edges, y_edges, _grid_values(np.zeros(n), rows, cols, inside),
              cmap=cmap, norm=norm)
    mesh.set_clip_path(path, ax.transData)
    _outline(ax, path, edgecolor)

    def apply(values):
        mesh.set_array(_grid_values(values, rows, cols, inside))

    return Fill(mesh, apply)


def _build_som(ax, path, n, cmap, norm, *, topology, polygon, features,
               edgecolor, random_state, n_iter, n_cells, metric):
    from ..sorting.som import SOM
    from ..sorting.voronoi import bounded_voronoi
    from .geom import grid_in_polygon

    if polygon is None:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    target = n_cells or min(n, 400)
    centers = grid_in_polygon(polygon, topology=topology, n_target=target)
    if len(centers) < 4:
        warnings.warn("region too small for a SOM grid; falling back to grid fill")
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)

    som = SOM(grid_shape=(1, len(centers)), topology=topology,
              cell_centers=centers, metric=metric, n_iter=n_iter,
              random_state=random_state).fit(features)
    assign = som.cell_assignments(features)
    counts = np.bincount(assign, minlength=len(centers))
    minx, miny, maxx, maxy = polygon.bounds
    polys = bounded_voronoi(centers, bbox=(minx, maxx, miny, maxy),
                            clip_shape=polygon)
    # Cells that are both drawn and own >=1 neuron -- stable across frames since
    # the SOM assignment is fixed.
    drawn = np.array([c for c in range(len(centers))
                      if polys[c] is not None and counts[c] > 0])
    if drawn.size == 0:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    patches = [MplPolygon(np.asarray(polys[c]), closed=True) for c in drawn]
    coll = _build_cells(ax, patches, cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)

    def apply(values):
        cv = _cell_means(np.asarray(values, dtype=float).ravel(), assign,
                         len(centers))
        coll.set_array(cv[drawn])

    return Fill(coll, apply)


def _build_voronoi(ax, path, n, cmap, norm, *, polygon, features, positions,
                   edgecolor, random_state):
    from ..sorting.voronoi import voronoi_parcellation

    if polygon is None:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    rng = np.random.default_rng(random_state)
    pos = _layout_positions(polygon, n, features, positions, rng)
    if pos is None or len(pos) < 3:
        warnings.warn("region too small for a Voronoi mosaic; falling back to grid")
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    minx, miny, maxx, maxy = polygon.bounds
    parc = voronoi_parcellation(pos, clip_shape=polygon,
                                bbox=(minx, maxx, miny, maxy))
    drawn = np.array([i for i, p in enumerate(parc.patches) if p is not None])
    if drawn.size == 0:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    coll = _build_cells(ax, [parc.patches[i] for i in drawn], cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)

    def apply(values):
        coll.set_array(np.asarray(values, dtype=float).ravel()[drawn])

    return Fill(coll, apply)


def _build_voronoi_kmeans(ax, path, n, cmap, norm, *, polygon, features,
                          positions, n_clusters, edgecolor, random_state):
    from ..sorting.voronoi import kmeans_voronoi_parcellation

    if polygon is None:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    if n_clusters is None:
        n_clusters = max(2, int(round(np.sqrt(n))))
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(random_state)
    pos = _layout_positions(polygon, n, features, positions, rng)
    if pos is None or len(pos) < n_clusters or n_clusters < 2:
        warnings.warn("region too small for a k-means Voronoi mosaic; "
                      "falling back to grid")
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    minx, miny, maxx, maxy = polygon.bounds
    parc = kmeans_voronoi_parcellation(
        pos, n_clusters=n_clusters, clip_shape=polygon,
        bbox=(minx, maxx, miny, maxy), random_state=random_state)
    labels = parc.cell_assignments
    drawn = np.array([c for c, p in enumerate(parc.patches) if p is not None])
    if drawn.size == 0:
        return _build_mesh(ax, path, n, cmap, norm, kind="pcolormesh",
                           edgecolor=edgecolor)
    coll = _build_cells(ax, [parc.patches[c] for c in drawn], cmap, norm, edgecolor)
    _outline(ax, path, edgecolor)

    def apply(values):
        cv = _cell_means(np.asarray(values, dtype=float).ravel(), labels,
                         n_clusters)
        coll.set_array(cv[drawn])

    return Fill(coll, apply)


def build_array_fill(ax, path, n_neurons, cmap, norm, *,
                     array_fill_type="pcolormesh", polygon=None, features=None,
                     positions=None, edgecolor="k", random_state=0, n_iter=1000,
                     n_cells=None, metric="euclidean", n_clusters=None):
    """Build an updatable :class:`Fill` for a ``(n_neurons,)`` region array.

    ``features`` is the static ``(n_neurons, k)`` layout driver (the caller
    supplies a sensible default, e.g. mean activity); ``positions`` an optional
    static 2-D layout for the Voronoi fills. The returned ``Fill`` is recoloured
    each frame via ``set_values(activity_vector)``.
    """
    if array_fill_type in ("pcolormesh", "pcolor"):
        return _build_mesh(ax, path, n_neurons, cmap, norm,
                           kind=array_fill_type, edgecolor=edgecolor)
    if array_fill_type in ("som_hex", "som_rect"):
        topology = "hex" if array_fill_type == "som_hex" else "rect"
        return _build_som(ax, path, n_neurons, cmap, norm, topology=topology,
                          polygon=polygon, features=features, edgecolor=edgecolor,
                          random_state=random_state, n_iter=n_iter,
                          n_cells=n_cells, metric=metric)
    if array_fill_type == "voronoi":
        return _build_voronoi(ax, path, n_neurons, cmap, norm, polygon=polygon,
                             features=features, positions=positions,
                             edgecolor=edgecolor, random_state=random_state)
    if array_fill_type == "voronoi_kmeans":
        return _build_voronoi_kmeans(ax, path, n_neurons, cmap, norm,
                                    polygon=polygon, features=features,
                                    positions=positions, n_clusters=n_clusters,
                                    edgecolor=edgecolor, random_state=random_state)
    raise ValueError(
        f"unknown array_fill_type {array_fill_type!r}; available: {available_fills()}"
    )
