"""NeuronSorter -- stateful sorter that can be re-applied across trials."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .cluster import sort_cluster
from .features import auto_features
from .merge import merge_1d, merge_2d
from .som import SOM
from .voronoi import (
    Parcellation,
    kmeans_voronoi_parcellation,
    voronoi_parcellation,
)


_VORONOI_METHODS = ("voronoi", "voronoi_kmeans")
_VALID_METHODS = ("cluster", "som") + _VORONOI_METHODS


class NeuronSorter:
    """Fit a 1D or 2D neuron ordering once; apply it to many activity arrays.

    Parameters
    ----------
    method : {'cluster', 'som', 'voronoi', 'voronoi_kmeans'}
        - ``cluster``: 1D hierarchical-clustering order.
        - ``som``: 1D or 2D self-organizing map on a fixed grid.
        - ``voronoi``: organic per-neuron Voronoi parcellation in 2D.
        - ``voronoi_kmeans``: k-means in 2D, Voronoi over centroids.
    ndim : {1, 2}
    grid : {'line', 'rect', 'hex'}
        SOM grid layout. Ignored for non-SOM methods.
    grid_shape : int or (int, int), optional
        SOM grid. Auto-sized from the number of input neurons if None.
    n_clusters : int, optional
        Required when ``method='voronoi_kmeans'``.
    outer : {'alpha', 'flat'}
        Voronoi only: outer-boundary style. ``'alpha'`` (default) clips
        every cell to a smoothed alpha-shape of the cloud; ``'flat'``
        gives each border (convex-hull) cell a single flat outer edge.
    alpha_factor : float, optional
        Voronoi-with-``outer='alpha'`` only: hull looseness in units of
        d_nn (``alpha = 1/(alpha_factor * d_nn)``). Default 3.0.
    flat_cap_factor : float, optional
        Voronoi-with-``outer='flat'`` only: flat-cap distance from each
        border generator, in units of d_nn. Default 0.7.
    disk_factor : float, optional
        Voronoi only: per-cell radius cap in units of d_nn. ``None`` lets
        cells fill the outer boundary continuously.
    smoothing : float, optional
        Gaussian filter width (seconds) applied to spike-like inputs before
        building features. ``None`` -> use input as-is.
    metric : str
        Distance for the underlying sorter.
    n_out : int, optional
        Target number of output columns after merging. ``None`` skips merging.
    random_state : int
    **som_kwargs
        Forwarded to :class:`SOM` (sigma, n_iter, learning_rate, pbc, ...).
    """

    def __init__(
        self,
        method: str = "cluster",
        ndim: int = 1,
        grid: str = "hex",
        grid_shape=None,
        n_clusters: Optional[int] = None,
        outer: str = "alpha",
        alpha_factor: float = 3.0,
        flat_cap_factor: float = 0.7,
        disk_factor: Optional[float] = None,
        smoothing: Optional[float] = 0.02,
        metric: str = "euclidean",
        n_out: Optional[int] = None,
        random_state: int = 0,
        standardize: bool = True,
        **som_kwargs,
    ):
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {method!r}"
            )
        if method == "cluster" and ndim != 1:
            raise ValueError("method='cluster' only supports ndim=1")
        if method in _VORONOI_METHODS:
            ndim = 2
        if method == "voronoi_kmeans" and n_clusters is None:
            raise ValueError(
                "method='voronoi_kmeans' requires n_clusters=<int>"
            )
        if ndim not in (1, 2):
            raise ValueError(f"ndim must be 1 or 2, got {ndim}")

        self.method = method
        self.ndim = ndim
        self.grid = grid if ndim == 2 else "line"
        self.grid_shape = grid_shape
        self.n_clusters = n_clusters
        self.outer = outer
        self.alpha_factor = alpha_factor
        self.flat_cap_factor = flat_cap_factor
        self.disk_factor = disk_factor
        self.smoothing = smoothing
        self.metric = metric
        self.n_out = n_out
        self.random_state = random_state
        self.standardize = standardize
        self.som_kwargs = som_kwargs

        # populated by fit()
        self.order: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.cell_assignments: Optional[np.ndarray] = None
        self.som: Optional[SOM] = None
        self.parcellation: Optional[Parcellation] = None
        self.n_neurons_in: Optional[int] = None
        self._merge_centers: Optional[np.ndarray] = None
        self._nonempty_cells: Optional[np.ndarray] = None

    # --------------------------------------------------------------- fitting

    def fit(self, X, t=None, features=None, positions_2d=None):
        """Learn the sort from *X* (or directly from *features*).

        Parameters
        ----------
        X : (n_time, n_neurons) array
        t : (n_time,) array, optional
        features : (n_neurons, n_features) array, optional
            Pre-built features; bypasses smoothing.
        positions_2d : (n_neurons, 2) array, optional
            Required for Voronoi methods. The 2D layout used to build the
            parcellation (e.g. encoders for a 2D ensemble, or a UMAP/PCA
            projection of higher-D features).
        """
        X = np.asarray(X)
        self.n_neurons_in = X.shape[1]

        # Voronoi methods don't need a feature matrix — they parcellate the
        # 2D layout passed via positions_2d.
        feats = None
        if self.method not in _VORONOI_METHODS:
            if features is None:
                feats = auto_features(X, t=t, smoothing=self.smoothing)
            else:
                feats = np.asarray(features, dtype=float)
                if feats.shape[0] != self.n_neurons_in:
                    raise ValueError(
                        f"features has {feats.shape[0]} rows but X has "
                        f"{self.n_neurons_in} neurons"
                    )

            if self.standardize and self.method == "som":
                mean = feats.mean(axis=0)
                std = feats.std(axis=0)
                std[std == 0] = 1.0
                feats = (feats - mean) / std

        if self.method == "cluster":
            self.order, _ = sort_cluster(
                X, t=t, smoothing=self.smoothing, metric=self.metric
            )
            self.positions = np.arange(self.n_neurons_in, dtype=float).reshape(-1, 1)
            return self

        if self.method in _VORONOI_METHODS:
            if positions_2d is None:
                raise ValueError(
                    f"method={self.method!r} requires positions_2d "
                    "(an (n_neurons, 2) layout). For high-D features, "
                    "project to 2D yourself (e.g. via umap-learn or "
                    "sklearn.decomposition.PCA) and pass the result."
                )
            positions_2d = np.asarray(positions_2d, dtype=float)
            if positions_2d.shape != (self.n_neurons_in, 2):
                raise ValueError(
                    f"positions_2d must be ({self.n_neurons_in}, 2), "
                    f"got {positions_2d.shape}"
                )
            if self.method == "voronoi":
                self.parcellation = voronoi_parcellation(
                    positions_2d,
                    outer=self.outer,
                    alpha_factor=self.alpha_factor,
                    flat_cap_factor=self.flat_cap_factor,
                    disk_factor=self.disk_factor,
                )
                self.cell_assignments = np.arange(self.n_neurons_in)
            else:  # voronoi_kmeans
                self.parcellation = kmeans_voronoi_parcellation(
                    positions_2d,
                    n_clusters=self.n_clusters,
                    outer=self.outer,
                    alpha_factor=self.alpha_factor,
                    flat_cap_factor=self.flat_cap_factor,
                    disk_factor=self.disk_factor,
                    random_state=self.random_state,
                )
                self.cell_assignments = self.parcellation.cell_assignments
            self.positions = self.parcellation.positions
            return self

        # SOM
        grid_shape = self.grid_shape
        if grid_shape is None:
            if self.ndim == 1:
                grid_shape = (1, self.n_neurons_in)
            else:
                side = int(np.ceil(np.sqrt(self.n_neurons_in)))
                grid_shape = (side, side)

        topology = "line" if self.ndim == 1 else self.grid
        self.som = SOM(
            grid_shape=grid_shape,
            topology=topology,
            metric=self.metric,
            random_state=self.random_state,
            **self.som_kwargs,
        ).fit(feats)
        self.cell_assignments = self.som.cell_assignments(feats)
        self.positions = self.som.cell_centers[self.cell_assignments]

        if self.ndim == 1:
            # derive a 1D order from cell index
            self.order = np.argsort(self.cell_assignments, kind="stable")
        return self

    # ------------------------------------------------------------- applying

    def transform(self, X, merge: Optional[bool] = None):
        """Apply the learned sort (and optionally merge) to *X*.

        With ``merge=None`` (default), merging happens iff ``n_out`` was set.
        Returns the sorted/merged array. After this call,
        :attr:`merged_positions` and :attr:`nonempty_cells` reflect the merge.
        """
        if self.positions is None:
            raise RuntimeError("call fit() before transform()")

        X = np.asarray(X)
        if X.shape[1] != self.n_neurons_in:
            raise ValueError(
                f"X has {X.shape[1]} columns, sorter was fit on {self.n_neurons_in}"
            )

        do_merge = self.n_out is not None if merge is None else merge

        if self.ndim == 1:
            X_sorted = X[:, self.order]
            if do_merge and self.n_out is not None and self.n_out < self.n_neurons_in:
                merged, centers = merge_1d(X_sorted, self.n_out)
                self._merge_centers = centers
                return merged
            self._merge_centers = np.arange(self.n_neurons_in, dtype=float)
            return X_sorted

        # 2D, voronoi_kmeans: always merge by cluster (each cluster becomes
        # one column of output, matching its patch).
        if self.method == "voronoi_kmeans":
            n_cells = int(self.cell_assignments.max()) + 1
            merged, centers, nonempty = merge_2d(
                X, self.cell_assignments, n_cells=n_cells,
                cell_centers=self.positions,
            )
            self._merge_centers = centers
            self._nonempty_cells = nonempty
            return merged

        # 2D, voronoi (per-neuron): no merging — one column per neuron.
        if self.method == "voronoi":
            self._merge_centers = self.positions
            return X

        # 2D, SOM: optional cell-merge
        if do_merge and self.som is not None:
            merged, centers, nonempty = merge_2d(
                X,
                self.cell_assignments,
                n_cells=self.som.n_cells,
                cell_centers=self.som.cell_centers,
            )
            self._merge_centers = centers
            self._nonempty_cells = nonempty
            return merged

        self._merge_centers = self.positions
        return X

    def fit_transform(self, X, t=None, features=None, positions_2d=None,
                      merge: Optional[bool] = None):
        return self.fit(
            X, t=t, features=features, positions_2d=positions_2d
        ).transform(X, merge=merge)

    # ------------------------------------------------------------ accessors

    @property
    def patches(self) -> Optional[List]:
        """Per-cell matplotlib patches, if the sort produced a parcellation.

        Returns the list of ``matplotlib.patches.Polygon`` objects from the
        underlying :class:`Parcellation` (Voronoi methods), or ``None`` for
        cluster/SOM sorts.
        """
        if self.parcellation is None:
            return None
        return self.parcellation.patches

    @property
    def merged_positions(self):
        """Coordinates of each column produced by the most recent ``transform``.

        Shape ``(n_out, 1)`` for 1D, ``(n_out, 2)`` for 2D. Equal to
        ``positions`` when no merge happened.
        """
        if self._merge_centers is None:
            return self.positions
        c = np.asarray(self._merge_centers)
        if c.ndim == 1:
            return c.reshape(-1, 1)
        return c

    @property
    def nonempty_cells(self):
        return self._nonempty_cells
