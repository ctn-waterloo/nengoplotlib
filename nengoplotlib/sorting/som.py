"""Self-organizing map for sorting neurons onto a 1D or 2D grid.

Pure NumPy. Supports rectangular and hexagonal 2D grids (and a degenerate
1xN line for 1D ordering). Replaces the MiniSom dependency used in the
original notebook.
"""

from __future__ import annotations

import numpy as np


def _hex_offset_coords(rows, cols):
    """Offset (x, y) coordinates for a *rows* x *cols* hex grid.

    Odd rows are shifted by 0.5 in x; y is scaled by sqrt(3)/2 so the cells
    are equilateral. Returns ``coords`` of shape ``(rows*cols, 2)`` indexed
    row-major.
    """
    i, j = np.indices((rows, cols))
    x = j + 0.5 * (i % 2)
    y = i * (np.sqrt(3) / 2)
    return np.stack([x, y], axis=-1).reshape(-1, 2).astype(float)


def _rect_coords(rows, cols):
    i, j = np.indices((rows, cols))
    return np.stack([j, i], axis=-1).reshape(-1, 2).astype(float)


def _normalize_rows(M, eps=1e-12):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, eps)


class SOM:
    """A small self-organizing map.

    Parameters
    ----------
    grid_shape : int or (int, int)
        ``(rows, cols)`` for a 2D map, or an ``int`` for a 1xN line. Pass
        ``(1, N)`` explicitly for the same effect.
    topology : {'rect', 'hex', 'line'}
        Cell layout. ``'line'`` is treated like ``'rect'`` with ``rows=1`` --
        it just yields evenly-spaced 1D coordinates.
    metric : {'euclidean', 'cosine'}
        Distance used when finding the best matching unit. For 'cosine', both
        weights and inputs are L2-normalized before the dot product.
    sigma : float
        Initial Gaussian neighborhood width (in grid-coordinate units).
        Decays exponentially to ``sigma_end`` over training.
    sigma_end : float
        Final neighborhood width.
    learning_rate : float
        Initial learning rate; decays exponentially to ``learning_rate_end``.
    learning_rate_end : float
    n_iter : int
        Number of input samples drawn during training. Each iteration picks
        one input vector (with replacement) and updates the weights.
    pbc : bool
        Periodic boundary conditions on the grid.
    random_state : int or None
    cell_centers : (n_cells, 2) array, optional
        Explicit cell-centre coordinates. When given, the regular grid built
        from ``grid_shape``/``topology`` is replaced by these centres (so the
        map can fill an arbitrary, e.g. polygon-masked, footprint).
        ``n_cells`` becomes ``len(cell_centers)`` and ``pbc`` is forced off,
        since wrapping is undefined for an irregular layout. Neighbourhood
        training is unchanged -- it operates on the centre coordinates.
    """

    def __init__(
        self,
        grid_shape,
        topology="hex",
        metric="euclidean",
        sigma=None,
        sigma_end=0.5,
        learning_rate=0.5,
        learning_rate_end=0.01,
        n_iter=1000,
        pbc=False,
        random_state=0,
        cell_centers=None,
    ):
        if isinstance(grid_shape, int):
            grid_shape = (1, grid_shape)
        self.rows, self.cols = grid_shape
        self.topology = topology
        self.metric = metric
        self.lr_start = learning_rate
        self.lr_end = learning_rate_end
        self.n_iter = n_iter
        self.random_state = random_state

        if cell_centers is not None:
            self.cell_centers = np.asarray(cell_centers, dtype=float)
            self.pbc = False  # wrapping is undefined for an irregular layout
            # Half the larger spatial extent of the custom layout.
            span = self.cell_centers.max(axis=0) - self.cell_centers.min(axis=0)
            default_sigma = max(float(span.max()), 1.0) / 2.0
        else:
            self.pbc = pbc
            if topology == "hex":
                self.cell_centers = _hex_offset_coords(self.rows, self.cols)
            elif topology in ("rect", "line"):
                self.cell_centers = _rect_coords(self.rows, self.cols)
            else:
                raise ValueError(f"unknown topology {topology!r}")
            default_sigma = max(self.rows, self.cols) / 2.0

        self.n_cells = len(self.cell_centers)
        self.sigma_start = sigma if sigma is not None else default_sigma
        self.sigma_end = sigma_end

        self.weights = None  # (n_cells, n_features) after fit

    # ------------------------------------------------------------------ utils

    def _grid_dist2(self, winner_idx):
        """Squared distance from each cell to *winner_idx* on the grid."""
        diff = self.cell_centers - self.cell_centers[winner_idx]
        if self.pbc:
            # Wrap on a rect grid. For hex, treat x extent as cols and y as
            # rows*sqrt(3)/2 -- close enough for the SOM-neighborhood use case.
            wx = float(self.cols)
            wy = self.rows * (np.sqrt(3) / 2 if self.topology == "hex" else 1.0)
            diff[:, 0] -= wx * np.round(diff[:, 0] / wx)
            diff[:, 1] -= wy * np.round(diff[:, 1] / wy)
        return np.sum(diff * diff, axis=1)

    def _bmu(self, x):
        if self.metric == "euclidean":
            d = np.sum((self.weights - x) ** 2, axis=1)
            return int(np.argmin(d))
        if self.metric == "cosine":
            wn = _normalize_rows(self.weights)
            xn = x / max(np.linalg.norm(x), 1e-12)
            return int(np.argmax(wn @ xn))
        raise ValueError(f"unknown metric {self.metric!r}")

    # ------------------------------------------------------------------- fit

    def fit(self, features):
        """Train the SOM on a (n_samples, n_features) array."""
        features = np.asarray(features, dtype=float)
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = features.shape
        # init weights by random samples (jittered slightly so duplicates differ)
        init_idx = rng.integers(0, n_samples, size=self.n_cells)
        self.weights = features[init_idx] + 1e-6 * rng.standard_normal((self.n_cells, n_features))

        log_sigma_ratio = np.log(self.sigma_end / self.sigma_start)
        log_lr_ratio = np.log(self.lr_end / self.lr_start)

        for it in range(self.n_iter):
            frac = it / max(self.n_iter - 1, 1)
            sigma = self.sigma_start * np.exp(log_sigma_ratio * frac)
            lr = self.lr_start * np.exp(log_lr_ratio * frac)

            x = features[rng.integers(0, n_samples)]
            bmu = self._bmu(x)
            d2 = self._grid_dist2(bmu)
            h = np.exp(-d2 / (2.0 * sigma * sigma))[:, None]
            self.weights += lr * h * (x - self.weights)

        return self

    # ----------------------------------------------------------- projection

    def winner(self, x):
        """Return the cell index of the BMU for one input vector."""
        return self._bmu(np.asarray(x, dtype=float))

    def cell_assignments(self, features):
        """Return the BMU cell index for every row of *features*."""
        features = np.asarray(features, dtype=float)
        return np.array([self._bmu(x) for x in features])

    def project(self, features):
        """Return the (n_samples, 2) grid coordinates of each input's BMU."""
        return self.cell_centers[self.cell_assignments(features)]


def sort_som(
    X,
    t=None,
    smoothing=None,
    features=None,
    ndim=2,
    grid_shape=None,
    topology="hex",
    metric="cosine",
    sigma=None,
    n_iter=1000,
    random_state=0,
    standardize=True,
):
    """One-shot SOM-based neuron sort.

    Parameters
    ----------
    X : (n_time, n_neurons) array
        Used to build features if *features* is None.
    features : (n_neurons, n_features) array, optional
        Pre-computed features (e.g. ensemble encoders). When given, *X* is
        only used for shape and *smoothing*/*t* are ignored.
    smoothing : float, optional
        Gaussian smoothing width in seconds applied to *X* before transposing
        to features. Ignored when *features* is supplied.
    ndim : {1, 2}
        Output dimensionality. ``1`` uses a 1xN grid.
    grid_shape : int or (int, int), optional
        Defaults to ``ceil(sqrt(n_neurons))`` on each side for 2D, or
        ``n_neurons`` for 1D.
    standardize : bool
        Z-score the features before training.

    Returns
    -------
    positions : (n_neurons, 2) array
        BMU coordinate for each neuron.
    som : SOM
    """
    from .features import auto_features

    if features is None:
        features = auto_features(X, t=t, smoothing=smoothing)
    features = np.asarray(features, dtype=float)

    if standardize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std

    n_neurons = features.shape[0]
    if grid_shape is None:
        if ndim == 1:
            grid_shape = (1, n_neurons)
        else:
            side = int(np.ceil(np.sqrt(n_neurons)))
            grid_shape = (side, side)

    som = SOM(
        grid_shape=grid_shape,
        topology=("line" if ndim == 1 else topology),
        metric=metric,
        sigma=sigma,
        n_iter=n_iter,
        random_state=random_state,
    ).fit(features)

    return som.project(features), som
