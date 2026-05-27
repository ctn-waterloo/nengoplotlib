"""Pytests for nengoplotlib.sorting."""

from __future__ import annotations

import numpy as np
import pytest

from nengoplotlib.sorting import (
    NeuronSorter,
    SOM,
    merge_1d,
    merge_2d,
    sample_by_activity,
    sample_by_variance,
    sample_random,
    smooth,
    sort_cluster,
    sort_neurons,
    sort_som,
)


# ----------------------------------------------------------------- fixtures


def make_two_group_spikes(seed=0, n_time=500, n_per_group=20, dt=0.001):
    """Two groups of neurons with distinct temporal envelopes.

    The first group fires more in the first half, the second more in the
    second half. Used to check that sorting / clustering separates the groups.
    """
    rng = np.random.default_rng(seed)
    t = dt * np.arange(n_time)
    env_a = (np.tanh(5 * (0.5 - t * 5 / t[-1])) + 1) / 2
    env_b = 1 - env_a
    spikes = np.empty((n_time, 2 * n_per_group))
    for j in range(n_per_group):
        spikes[:, j] = rng.random(n_time) < 0.05 * env_a
        spikes[:, n_per_group + j] = rng.random(n_time) < 0.05 * env_b
    rng.shuffle(spikes.T)  # in-place column shuffle so groups are mixed
    return t, spikes.astype(float)


# ------------------------------------------------------------ basic helpers


def test_smooth_uses_dt_when_t_missing():
    X = np.zeros((100, 3))
    X[50, :] = 1.0
    out = smooth(X, dt=0.001, filter_width=0.005)
    assert out.shape == X.shape
    # peak of the gaussian stays at the spike location
    assert np.argmax(out[:, 0]) == 50


def test_smooth_requires_t_or_dt():
    with pytest.raises(ValueError):
        smooth(np.zeros((10, 2)))


# -------------------------------------------------------------- sampling


def test_sample_by_variance_picks_active_columns():
    rng = np.random.default_rng(0)
    n_time = 200
    X = np.zeros((n_time, 10))
    X[:, 0] = np.sin(np.linspace(0, 6, n_time))  # high variance
    X[:, 1] = np.cos(np.linspace(0, 6, n_time))  # high variance
    X[:, 5] = 1e-9 * rng.standard_normal(n_time)  # near-zero variance
    sel, idx = sample_by_variance(X, num=2, dt=0.001, filter_width=0.005)
    assert sel.shape == (n_time, 2)
    assert set(idx.tolist()) == {0, 1}


def test_sample_by_activity_returns_top_columns():
    X = np.zeros((50, 5))
    X[:, 2] = 1.0  # all-on
    X[:10, 4] = 1.0
    sel, idx = sample_by_activity(X, num=2)
    assert idx[0] == 2  # highest activity wins
    assert sel.shape == (50, 2)


def test_sample_random_is_deterministic():
    X = np.arange(200).reshape(10, 20).astype(float)
    a, ia = sample_random(X, num=5, random_state=42)
    b, ib = sample_random(X, num=5, random_state=42)
    np.testing.assert_array_equal(ia, ib)
    np.testing.assert_array_equal(a, b)


def test_sample_returns_input_when_already_small_enough():
    X = np.ones((10, 3))
    sel, idx = sample_by_variance(X, num=5, dt=0.001, filter_width=0.005)
    assert sel.shape == (10, 3)
    np.testing.assert_array_equal(idx, [0, 1, 2])


# ---------------------------------------------------------------- merging


def test_merge_1d_block_average():
    X = np.tile(np.arange(8.0), (3, 1))  # (n_time=3, n_neurons=8)
    merged, centers = merge_1d(X, n_out=4)
    expected = np.array([0.5, 2.5, 4.5, 6.5])
    np.testing.assert_array_equal(merged[0], expected)
    assert centers.shape == (4,)


def test_merge_1d_passthrough_when_n_out_exceeds():
    X = np.arange(6.0).reshape(2, 3)
    merged, centers = merge_1d(X, n_out=10)
    np.testing.assert_array_equal(merged, X)
    np.testing.assert_array_equal(centers, [0, 1, 2])


def test_merge_1d_handles_uneven_split_without_nans():
    """Regression: merge_1d used to produce NaN trailing columns when
    n_neurons / n_out wasn't an integer (ceil(n/n_out) * n_out > n_neurons)."""
    X = np.ones((4, 500))
    merged, centers = merge_1d(X, n_out=80)
    assert merged.shape == (4, 80)
    assert centers.shape == (80,)
    assert np.all(np.isfinite(merged))
    assert np.all(np.isfinite(centers))


def test_merge_1d_uses_order():
    X = np.tile(np.array([10.0, 20, 30, 40]), (2, 1))
    merged, _ = merge_1d(X, n_out=2, order=[3, 2, 1, 0])
    np.testing.assert_array_equal(merged[0], [35.0, 15.0])


def test_merge_2d_groups_by_cell():
    X = np.array([[1.0, 2, 3, 4, 5]])  # 1 timestep, 5 neurons
    cells = np.array([0, 0, 1, 1, 2])
    centers = np.array([[0, 0], [1, 0], [0, 1], [9, 9]])  # cell 3 empty
    merged, pos, nonempty = merge_2d(X, cells, n_cells=4, cell_centers=centers)
    np.testing.assert_array_equal(nonempty, [0, 1, 2])
    np.testing.assert_array_equal(merged[0], [1.5, 3.5, 5.0])
    np.testing.assert_array_equal(pos, centers[:3])


# --------------------------------------------------------- sort_cluster


def test_sort_cluster_returns_permutation():
    t, spikes = make_two_group_spikes()
    order, sorted_spikes = sort_cluster(spikes, t=t, smoothing=0.01)
    assert order.shape == (spikes.shape[1],)
    np.testing.assert_array_equal(sorted(order), np.arange(spikes.shape[1]))
    np.testing.assert_array_equal(sorted_spikes, spikes[:, order])


def test_sort_cluster_groups_similar_neurons():
    """Neurons in the same envelope group should end up close in the order."""
    t, spikes = make_two_group_spikes(seed=1)
    n = spikes.shape[1] // 2
    # ground-truth group: half of cols have envelope A, half B. After shuffle
    # we lost the labels, so reconstruct them from the smoothed correlation.
    filtered = smooth(spikes, t=t, filter_width=0.05)
    template = filtered[:, np.argmax(filtered.std(axis=0))]
    group = (np.array(
        [np.corrcoef(filtered[:, j], template)[0, 1] for j in range(spikes.shape[1])]
    ) > 0).astype(int)

    order, _ = sort_cluster(spikes, t=t, smoothing=0.01)
    # measure: how many adjacent pairs in `order` cross the group boundary?
    crossings = np.sum(group[order[:-1]] != group[order[1:]])
    # Random ordering would average ~half the pairs as crossings; clustered
    # ordering should put nearly all same-group neurons together.
    assert crossings < 0.25 * (len(order) - 1)


# ---------------------------------------------------------------- SOM


def test_som_fits_and_projects():
    rng = np.random.default_rng(0)
    # two well-separated clusters in feature space
    cluster_a = rng.normal(loc=[-3, -3], scale=0.2, size=(30, 2))
    cluster_b = rng.normal(loc=[3, 3], scale=0.2, size=(30, 2))
    features = np.vstack([cluster_a, cluster_b])

    som = SOM(grid_shape=(5, 5), topology="rect", n_iter=400, random_state=0).fit(features)
    cells = som.cell_assignments(features)
    # the two clusters should land in disjoint cells
    cells_a, cells_b = set(cells[:30]), set(cells[30:])
    assert cells_a.isdisjoint(cells_b)
    assert som.cell_centers.shape == (25, 2)


def test_som_hex_coords_are_offset():
    som = SOM(grid_shape=(3, 4), topology="hex", n_iter=10, random_state=0)
    coords = som.cell_centers.reshape(3, 4, 2)
    # row 1 (odd) is shifted in x by 0.5 relative to row 0
    assert np.allclose(coords[1, :, 0] - coords[0, :, 0], 0.5)
    # y spacing is sqrt(3)/2
    assert np.allclose(coords[1, 0, 1] - coords[0, 0, 1], np.sqrt(3) / 2)


def test_sort_som_returns_positions_per_neuron():
    rng = np.random.default_rng(0)
    n_time, n_neurons = 100, 16
    X = rng.standard_normal((n_time, n_neurons))
    positions, som = sort_som(X, smoothing=None, grid_shape=(4, 4), n_iter=100, random_state=0)
    assert positions.shape == (n_neurons, 2)
    assert isinstance(som, SOM)


# -------------------------------------------------------- NeuronSorter


def test_neuronsorter_cluster_round_trip():
    t, spikes = make_two_group_spikes()
    sorter = NeuronSorter(method="cluster", ndim=1, smoothing=0.01).fit(spikes, t=t)
    out = sorter.transform(spikes)
    assert out.shape == spikes.shape
    np.testing.assert_array_equal(out, spikes[:, sorter.order])


def test_neuronsorter_cluster_merges_when_n_out_set():
    t, spikes = make_two_group_spikes()
    sorter = NeuronSorter(method="cluster", n_out=10, smoothing=0.01).fit(spikes, t=t)
    out = sorter.transform(spikes)
    assert out.shape == (spikes.shape[0], 10)
    assert sorter.merged_positions.shape == (10, 1)


def test_neuronsorter_reuse_across_trials():
    """The same fitted sorter must reorder unseen trials consistently."""
    t, spikes_a = make_two_group_spikes(seed=1)
    _, spikes_b = make_two_group_spikes(seed=2)
    sorter = NeuronSorter(method="cluster", smoothing=0.01).fit(spikes_a, t=t)
    out_b = sorter.transform(spikes_b)
    np.testing.assert_array_equal(out_b, spikes_b[:, sorter.order])


def test_neuronsorter_som_2d():
    rng = np.random.default_rng(0)
    n_time, n_neurons = 200, 25
    X = rng.standard_normal((n_time, n_neurons))
    sorter = NeuronSorter(
        method="som", ndim=2, grid="hex",
        grid_shape=(5, 5), n_iter=100, smoothing=None, metric="cosine",
    ).fit(X)
    assert sorter.positions.shape == (n_neurons, 2)
    out = sorter.transform(X, merge=True)
    # merging on SOM never produces more columns than cells
    assert out.shape[1] <= 25
    assert sorter.merged_positions.shape == (out.shape[1], 2)


def test_neuronsorter_rejects_bad_ndim_for_cluster():
    with pytest.raises(ValueError):
        NeuronSorter(method="cluster", ndim=2)


def test_sort_neurons_one_shot():
    t, spikes = make_two_group_spikes()
    out, sorter = sort_neurons(spikes, t=t, method="cluster", n_out=20, smoothing=0.01)
    assert out.shape == (spikes.shape[0], 20)
    assert isinstance(sorter, NeuronSorter)


def test_neuronsorter_features_kwarg():
    """When the user passes encoder-like features, they should drive the sort."""
    t, spikes = make_two_group_spikes()
    encoders = np.eye(spikes.shape[1])  # dummy features
    sorter = NeuronSorter(method="cluster", smoothing=None).fit(
        spikes, t=t, features=encoders
    )
    assert sorter.positions.shape == (spikes.shape[1], 1)


def test_neuronsorter_voronoi_per_neuron():
    """voronoi method produces a patch per neuron and a 2D position array."""
    rng = np.random.default_rng(0)
    n = 60
    positions = rng.standard_normal((n, 2))
    X = rng.standard_normal((50, n)).cumsum(axis=0)

    sorter = NeuronSorter(method="voronoi").fit(X, positions_2d=positions)
    assert sorter.positions.shape == (n, 2)
    assert len(sorter.patches) == n
    out = sorter.transform(X)
    assert out.shape == X.shape  # per-neuron: no merging


def test_neuronsorter_voronoi_kmeans_merges_clusters():
    rng = np.random.default_rng(0)
    n = 80
    positions = rng.standard_normal((n, 2))
    X = rng.standard_normal((50, n)).cumsum(axis=0)

    sorter = NeuronSorter(method="voronoi_kmeans", n_clusters=8).fit(
        X, positions_2d=positions
    )
    assert sorter.positions.shape == (8, 2)
    assert len(sorter.patches) == 8
    out = sorter.transform(X)
    assert out.shape[1] == 8


def test_neuronsorter_voronoi_requires_positions():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 20))
    with pytest.raises(ValueError, match="positions_2d"):
        NeuronSorter(method="voronoi").fit(X)


def test_neuronsorter_voronoi_kmeans_requires_n_clusters():
    with pytest.raises(ValueError, match="n_clusters"):
        NeuronSorter(method="voronoi_kmeans")
