"""Run every nengoplotlib plot function and save its output.

PNGs land in ``examples/_images/``, animated GIFs alongside them. Re-run after
API changes to refresh the README assets.

    python examples/make_example_images.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import nengo

import nengoplotlib as npl
from nengoplotlib.style import (
    EDGE_COLOR, FG_COLOR,
    activity_cmap, add_thin_colorbar, apply_style, style_legend,
    title_block,
)

apply_style()
ACTIVITY_CMAP = activity_cmap()


HERE = Path(__file__).resolve().parent
OUT = HERE / "_images"
OUT.mkdir(exist_ok=True)
DPI = 150


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  -> {path.relative_to(HERE.parent)}")


# ---------------------------------------------------------------- data


def simulate_oscillator(seed=1, n_neurons=500, t_run=5.0):
    with nengo.Network(seed=seed) as model:
        inp = nengo.Node(lambda t: [np.sin(t), np.cos(t)])
        ens = nengo.Ensemble(n_neurons, 2)
        nengo.Connection(inp, ens)
        p_spikes = nengo.Probe(ens.neurons, synapse=None)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_run)
    return sim.trange(), sim.data[p_spikes], ens, sim


def simulate_sparse(seed=1, n_neurons=500, t_run=3.0):
    with nengo.Network(seed=seed) as model:
        inp = nengo.Node(lambda t: np.sin(10 * t))
        ens = nengo.Ensemble(
            n_neurons, 1,
            max_rates=nengo.dists.Uniform(10, 20),
            intercepts=nengo.dists.Uniform(0.5, 0.99),
        )
        nengo.Connection(inp, ens, synapse=0.05)
        p_spikes = nengo.Probe(ens.neurons, synapse=None)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_run)
    return sim.trange(), sim.data[p_spikes]


def simulate_multi_trial(n_trials=10, seed=1, n_neurons=200, t_run=2.0):
    trials = []
    for i in range(n_trials):
        proc = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 0.05), seed=i)
        with nengo.Network(seed=seed) as model:
            inp = nengo.Node(lambda t: [np.sin(t), np.cos(t)])
            ens = nengo.Ensemble(
                n_neurons, 2, noise=proc,
                max_rates=nengo.dists.Uniform(20, 50),
            )
            nengo.Connection(inp, ens)
            p_spikes = nengo.Probe(ens.neurons)
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(t_run)
        trials.append(sim.data[p_spikes])
    return sim.trange(), trials


def build_oscillator_recurrent(seed=0):
    """A 2D recurrent network that decodes a rotation -- nice phase portrait."""
    with nengo.Network(seed=seed) as model:
        ens = nengo.Ensemble(200, 2)

        def rotate(x):
            return [x[0] - 1.5 * x[1], 1.5 * x[0] + x[1]]

        conn = nengo.Connection(ens, ens, function=rotate, synapse=0.1)
    sim = nengo.Simulator(model, progress_bar=False)
    return ens, conn, sim, model


def simulate_hierarchy(seed=2):
    """A small two-level network for a connectome example."""
    with nengo.Network(seed=seed, label="model") as model:
        with nengo.Network(label="sensory"):
            v1 = nengo.Ensemble(80, 2, label="v1")
            mt = nengo.Ensemble(60, 2, label="mt")
            nengo.Connection(v1, mt)
        with nengo.Network(label="motor"):
            m1 = nengo.Ensemble(100, 2, label="m1")
            sma = nengo.Ensemble(50, 2, label="sma")
            nengo.Connection(sma, m1)
        nengo.Connection(mt, sma)
        nengo.Connection(m1, v1, transform=0.1)
    return model


# ----------------------------------------------------------------- plots


def make_plot_spikes(t, spikes):
    print("plot_spikes")
    sorted_X, _ = npl.sort_neurons(spikes, t=t, method="cluster", n_out=50, smoothing=0.002)
    fig, ax = plt.subplots(figsize=(5, 2))
    npl.plot_spikes(t, sorted_X, ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron #")
    save(fig, "plot_spikes.png")


def make_plot_heatmap(t, spikes):
    print("plot_heatmap")
    sorted_X, _ = npl.sort_neurons(spikes, t=t, method="cluster", n_out=80, smoothing=0.002)
    filtered = npl.smooth(sorted_X, t=t, filter_width=0.02)
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    mesh = npl.plot_heatmap(t, filtered, cmap=ACTIVITY_CMAP, ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron #")
    add_thin_colorbar(fig, ax, mesh, filtered.ravel(), label="activity")
    save(fig, "plot_heatmap.png")


def make_plot_traces():
    print("plot_traces")
    t, spikes = simulate_sparse()
    sub = spikes[:, ::10]
    filtered = npl.smooth(sub, t=t, filter_width=0.1)
    fig, ax = plt.subplots(figsize=(5, 3))
    npl.plot_traces(t, filtered[:, :40], offset=0.6, cmap="tab10", ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron #")
    save(fig, "plot_traces.png")


def make_plot_psth(t, trials):
    print("plot_psth")
    fig = npl.plot_psth(
        trials, t=t, dt=t[1] - t[0],
        neuron_idxs=[10, 30, 100],
        smoothing_sigma=50,
    )
    save(fig, "plot_psth.png")
    # npl.plot_psth(
    #     trials, t=t, dt=t[1] - t[0],
    #     neuron_idxs=[10, 30, 100],
    #     smoothing_sigma=50,  figsize=(1,20)
    # )
    # save(fig, "plot_psth_v2.png")


def make_scrolling_raster(t, spikes):
    print("plot_scrolling_raster")
    sorted_X, _ = npl.sort_neurons(spikes, t=t, method="cluster", n_out=80, smoothing=0.002)
    ani = npl.plot_scrolling_raster(
        t, sorted_X, plot_step=50, window=1.0, tau=0.01, interval=80,
    )
    path = OUT / "plot_scrolling_raster.gif"
    ani.save(path, writer="pillow", fps=15, dpi=100)
    plt.close("all")
    print(f"  -> {path.relative_to(HERE.parent)}")


def make_grid_animation(t, spikes, ens, sim):
    print("plot_grid_animation")
    sub_idx = np.arange(0, spikes.shape[1], 2)
    sub_spikes = spikes[:, sub_idx]
    features = sim.data[ens].encoders[sub_idx]
    sorter = npl.NeuronSorter(
        method="som", ndim=2, grid="hex",
        grid_shape=(15, 15), metric="cosine",
        n_iter=2000, smoothing=None, random_state=0,
    ).fit(sub_spikes, features=features)
    ani = npl.plot_grid_animation(
        t, sub_spikes, sorter=sorter,
        plot_step=50, tau=0.02, cmap=ACTIVITY_CMAP, interval=80,
    )
    path = OUT / "plot_grid_animation.gif"
    ani.save(path, writer="pillow", fps=15, dpi=100,
             savefig_kwargs={"transparent": True, "facecolor": "none"})
    plt.close("all")
    print(f"  -> {path.relative_to(HERE.parent)}")


def simulate_disk_ensemble(seed=1, n_neurons=600, dims=6, t_run=4.0):
    """A higher-D ensemble whose encoders, projected to 2D via PCA, fill a
    disk (not a ring). Drives the Voronoi parcellation examples."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 1.2, size=dims)
    phases = rng.uniform(0, 2 * np.pi, size=dims)

    def input_fn(t):
        return 0.7 * np.sin(2 * np.pi * freqs * t + phases)

    with nengo.Network(seed=seed) as model:
        inp = nengo.Node(input_fn, size_out=dims)
        ens = nengo.Ensemble(n_neurons, dims)
        nengo.Connection(inp, ens)
        p_spikes = nengo.Probe(ens.neurons, synapse=0.05)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_run)
    encoders = sim.data[ens].encoders
    # PCA-project the (n_neurons, dims) encoders to 2D. Mean-center first.
    centered = encoders - encoders.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    positions = centered @ Vt[:2].T
    return sim.trange(), sim.data[p_spikes], positions


def make_voronoi_parcellation_image():
    """Static colored Voronoi parcellation — README hero image."""
    print("plot_voronoi_parcellation (README image)")
    t, spikes, positions = simulate_disk_ensemble(seed=1, n_neurons=600,
                                                  t_run=4.0)
    # sorter = npl.NeuronSorter(
    #     method="voronoi_kmeans", n_clusters=36, outer="flat",
    # ).fit(spikes, positions_2d=positions)
    sorter = npl.NeuronSorter(method='voronoi').fit(spikes, positions_2d=positions)
    merged = sorter.transform(spikes)
    cluster_activity = merged.mean(axis=0)

    bbox = sorter.parcellation.bbox
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    from matplotlib.collections import PatchCollection
    valid = [(p, a) for p, a in zip(sorter.patches, cluster_activity) if p is not None]
    patches = [p for p, _ in valid]
    colors = np.array([a for _, a in valid])
    pc = PatchCollection(patches, cmap=ACTIVITY_CMAP,
                         edgecolor=EDGE_COLOR, linewidth=1.0)
    pc.set_array(colors)
    ax.add_collection(pc)
    add_thin_colorbar(fig, ax, pc, colors, label="mean firing rate")
    save(fig, "plot_voronoi_parcellation.png")


def make_voronoi_grid_animation_disk():
    """Animated Voronoi parcellation using the disk-filled ensemble."""
    print("plot_voronoi_grid_animation")
    t, spikes, positions = simulate_disk_ensemble(seed=1, n_neurons=600,
                                                  t_run=5.0)
    # sorter = npl.NeuronSorter(
    #     method="voronoi_kmeans", n_clusters=36, outer="flat",
    # ).fit(spikes, positions_2d=positions)
    sorter = npl.NeuronSorter(method='voronoi').fit(spikes, positions_2d=positions)

    ani = npl.plot_grid_animation(
        t, spikes, sorter=sorter,
        plot_step=50, tau=0.05, cmap=ACTIVITY_CMAP, interval=80,
        figsize=(4.5, 4.5),
    )
    path = OUT / "plot_voronoi_grid_animation.gif"
    ani.save(path, writer="pillow", fps=15, dpi=100,
             savefig_kwargs={"transparent": True, "facecolor": "none"})
    plt.close("all")
    print(f"  -> {path.relative_to(HERE.parent)}")


def make_phase_portrait():
    print("plot_phase_portrait")
    ens, conn, sim, _ = build_oscillator_recurrent()
    fig, ax, _ = npl.plot_phase_portrait(
        ens, conn, sim=sim, S=21, plot_type="stream", cmap=ACTIVITY_CMAP,
        figsize=(4, 4),
    )
    save(fig, "plot_phase_portrait.png")


def make_isi(spikes):
    print("plot_isi")
    # Pick four neurons with a spread of firing rates so the histograms differ.
    rates = (spikes > 0).sum(axis=0)
    sorted_by_rate = np.argsort(rates)
    chosen = sorted_by_rate[[len(rates) // 5, len(rates) // 2,
                             3 * len(rates) // 4, -2]]
    fig = npl.plot_isi(
        spikes, dt=0.001, neuron_idxs=chosen,
        bins=30, max_isi=0.25, per_neuron=True,
    )
    save(fig, "plot_isi.png")


def make_weight_matrix():
    print("plot_weight_matrix")
    with nengo.Network(seed=2) as model:
        pre = nengo.Ensemble(60, 2)
        post = nengo.Ensemble(60, 2)
        conn = nengo.Connection(pre, post, function=lambda x: [x[1], -x[0]])
    sim = nengo.Simulator(model, progress_bar=False)
    W = sim.data[conn].weights  # (2, 60) decoder

    encoders = sim.data[pre].encoders
    sorter = npl.NeuronSorter(method="cluster", smoothing=None).fit(W, features=encoders)
    fig, ax, _ = npl.plot_weight_matrix(
        W, col_order=sorter, cmap="RdBu_r",
        xlabel="pre neuron (sorted)", ylabel="output dim",
    )
    save(fig, "plot_weight_matrix.png")


def make_connectome():
    print("plot_connectome")
    model = simulate_hierarchy()
    fig, ax = plt.subplots(figsize=(7, 5))
    npl.plot_connectome(model, label_depth=1, ax=ax)
    save(fig, "plot_connectome.png")


# ---------------------------------------------------------------- main


def main():
    print("Simulating data...")
    t_osc, spikes_osc, ens, sim = simulate_oscillator(t_run=5.0)
    t_long, spikes_long, ens_long, sim_long = simulate_oscillator(seed=1, t_run=8.0)
    t_trials, trials = simulate_multi_trial()

    print("Rendering plots into", OUT)
    make_plot_spikes(t_osc, spikes_osc)
    make_plot_heatmap(t_osc, spikes_osc)
    make_plot_traces()
    make_plot_psth(t_trials, trials)
    make_scrolling_raster(t_long, spikes_long)
    make_grid_animation(t_long, spikes_long, ens_long, sim_long)
    make_voronoi_grid_animation_disk()
    make_voronoi_parcellation_image()
    make_phase_portrait()
    make_isi(spikes_osc)
    make_weight_matrix()
    make_connectome()
    print("done.")


if __name__ == "__main__":
    main()
