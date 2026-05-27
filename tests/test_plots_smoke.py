"""Lightweight smoke tests for the plotting functions.

We don't validate pixel output; we just check that each function runs end to
end on small synthetic data without raising and returns the expected type.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nengoplotlib as npl


@pytest.fixture
def spikes_small():
    rng = np.random.default_rng(0)
    n_time, n_neurons = 100, 8
    t = 0.001 * np.arange(n_time)
    X = (rng.random((n_time, n_neurons)) < 0.1).astype(float)
    return t, X


def test_plot_spikes_runs(spikes_small):
    t, X = spikes_small
    fig, ax = plt.subplots()
    img = npl.plot_spikes(t, X, ax=ax)
    assert img is not None
    plt.close(fig)


def test_plot_heatmap_runs(spikes_small):
    t, X = spikes_small
    fig, ax = plt.subplots()
    mesh = npl.plot_heatmap(t, X, ax=ax)
    assert mesh is not None
    plt.close(fig)


def test_plot_traces_runs(spikes_small):
    t, X = spikes_small
    fig, ax = plt.subplots()
    out = npl.plot_traces(t, X, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_psth_runs(spikes_small):
    t, X = spikes_small
    trials = [X, X * 0.5, X * 0.25]  # 3 fake "trials"
    fig = npl.plot_psth(trials, t=t, dt=t[1] - t[0], neuron_idxs=[0, 1])
    assert fig is not None
    plt.close(fig)


def test_plot_scrolling_raster_runs(spikes_small):
    t, X = spikes_small
    ani = npl.plot_scrolling_raster(t, X, plot_step=10, window=0.05, interval=50)
    assert ani is not None
    plt.close("all")


def test_plot_grid_animation_with_positions(spikes_small):
    t, X = spikes_small
    rng = np.random.default_rng(0)
    positions = rng.random((X.shape[1], 2))
    ani = npl.plot_grid_animation(
        t, X, positions=positions, topology="rect", plot_step=10, interval=50,
    )
    assert ani is not None
    plt.close("all")


def test_plot_grid_animation_with_sorter(spikes_small):
    t, X = spikes_small
    sorter = npl.NeuronSorter(
        method="som", ndim=2, grid="hex",
        grid_shape=(3, 3), n_iter=50, smoothing=None, metric="cosine",
    ).fit(X)
    ani = npl.plot_grid_animation(t, X, sorter=sorter, plot_step=10, interval=50)
    assert ani is not None
    plt.close("all")
