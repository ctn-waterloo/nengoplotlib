"""Smoke / sanity tests for plot_phase_portrait, plot_isi, plot_weight_matrix."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nengoplotlib as npl


# ------------------------------------------------------------------- ISI


def test_plot_isi_pooled_runs():
    rng = np.random.default_rng(0)
    spikes = (rng.random((1000, 5)) < 0.05).astype(float)
    fig = npl.plot_isi(spikes, dt=0.001, bins=20)
    assert fig is not None
    plt.close(fig)


def test_plot_isi_per_neuron_grid():
    rng = np.random.default_rng(1)
    spikes = (rng.random((1000, 6)) < 0.05).astype(float)
    fig = npl.plot_isi(spikes, dt=0.001, neuron_idxs=[0, 1, 2, 3], per_neuron=True, bins=15)
    assert fig is not None
    plt.close(fig)


def test_plot_isi_handles_silent_neurons():
    spikes = np.zeros((500, 3))
    spikes[100, 1] = 1.0  # only one spike anywhere — no ISIs
    fig = npl.plot_isi(spikes, dt=0.001)
    assert fig is not None
    plt.close(fig)


def test_plot_isi_max_isi_clips_tail():
    spikes = np.zeros((10000, 1))
    spikes[[100, 200, 5000], 0] = 1.0  # ISIs of 0.1s and 4.9s
    fig = npl.plot_isi(spikes, dt=0.001, max_isi=1.0, bins=5)
    plt.close(fig)


# ------------------------------------------------------------- weights


def test_plot_weight_matrix_array():
    W = np.linspace(-1, 1, 200).reshape(10, 20)
    fig, ax, img = npl.plot_weight_matrix(W, colorbar=False)
    assert img.get_array().shape == W.shape
    # symmetric defaults to True -> clim symmetric about 0
    cmin, cmax = img.get_clim()
    assert np.isclose(cmin, -cmax)
    plt.close(fig)


def test_plot_weight_matrix_with_sorter():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((30, 30))
    # build a fake sorter-like object
    class FakeSorter:
        order = np.arange(30)[::-1]
    fig, ax, img = npl.plot_weight_matrix(
        W, row_order=FakeSorter(), col_order=FakeSorter(), colorbar=False,
    )
    np.testing.assert_array_equal(img.get_array(), W[::-1][:, ::-1])
    plt.close(fig)


def test_plot_weight_matrix_rejects_3d_input():
    with pytest.raises(ValueError):
        npl.plot_weight_matrix(np.zeros((3, 4, 5)))


def test_plot_weight_matrix_real_neuron_sorter():
    """Pass an actual NeuronSorter trained on a fake feature matrix."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((40, 40))
    features = rng.standard_normal((40, 8))
    sorter = npl.NeuronSorter(method="cluster", smoothing=None).fit(W, features=features)
    fig, ax, img = npl.plot_weight_matrix(W, row_order=sorter, col_order=sorter, colorbar=False)
    np.testing.assert_array_equal(img.get_array(), W[sorter.order][:, sorter.order])
    plt.close(fig)


# ---------------------------------------------------------- phase portrait


nengo = pytest.importorskip("nengo")


@pytest.fixture(scope="module")
def recurrent_2d():
    """A simple 2D ensemble with a recurrent identity-ish connection."""
    with nengo.Network(seed=0) as model:
        ens = nengo.Ensemble(80, 2)
        conn = nengo.Connection(ens, ens, function=lambda x: x * 0.9)
    sim = nengo.Simulator(model, progress_bar=False)
    return ens, conn, sim, model


def test_phase_portrait_2d_quiver(recurrent_2d):
    ens, conn, sim, _ = recurrent_2d
    fig, ax, field = npl.plot_phase_portrait(ens, conn, sim=sim, S=11, plot_type="quiver")
    assert field is not None
    plt.close(fig)


def test_phase_portrait_2d_stream(recurrent_2d):
    ens, conn, sim, _ = recurrent_2d
    fig, ax, field = npl.plot_phase_portrait(ens, conn, sim=sim, S=11, plot_type="stream")
    assert field is not None
    plt.close(fig)


def test_phase_portrait_builds_sim_from_network(recurrent_2d):
    ens, conn, _, model = recurrent_2d
    fig, ax, field = npl.plot_phase_portrait(ens, conn, network=model, S=7)
    assert field is not None
    plt.close(fig)


def test_phase_portrait_1d():
    with nengo.Network(seed=0) as model:
        ens = nengo.Ensemble(40, 1)
        conn = nengo.Connection(ens, ens, function=lambda x: -x)
    sim = nengo.Simulator(model, progress_bar=False)
    fig, ax, field = npl.plot_phase_portrait(ens, conn, sim=sim, S=15)
    assert field is not None
    plt.close(fig)


def test_phase_portrait_3d():
    with nengo.Network(seed=0) as model:
        ens = nengo.Ensemble(120, 3)
        conn = nengo.Connection(ens, ens, function=lambda x: 0.5 * x)
    sim = nengo.Simulator(model, progress_bar=False)
    fig, ax, field = npl.plot_phase_portrait(ens, conn, sim=sim, S=5)
    assert field is not None
    plt.close(fig)


def test_phase_portrait_rejects_high_dim():
    with nengo.Network(seed=0) as model:
        ens = nengo.Ensemble(50, 4)
        conn = nengo.Connection(ens, ens)
    sim = nengo.Simulator(model, progress_bar=False)
    with pytest.raises(ValueError):
        npl.plot_phase_portrait(ens, conn, sim=sim)


def test_phase_portrait_rejects_mismatched_dims():
    with nengo.Network(seed=0) as model:
        pre = nengo.Ensemble(40, 2)
        post = nengo.Ensemble(40, 3)
        conn = nengo.Connection(pre, post, function=lambda x: [x[0], x[1], 0])
    sim = nengo.Simulator(model, progress_bar=False)
    with pytest.raises(ValueError):
        npl.plot_phase_portrait(pre, conn, sim=sim)
