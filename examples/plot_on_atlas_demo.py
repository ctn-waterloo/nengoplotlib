"""Paint per-region data onto the Allen Mouse atlas with ``plot_on_atlas``.

Four views of the same idea:

1. a coronal section with **scalar** fills (one colour per area),
2. a **per-neuron array** laid out inside a region three ways -- a grid, a
   shape-constrained **SOM**, and a per-neuron **Voronoi** mosaic,
3. the **Swanson** flat projection (whole brain at once, scalar fills),
4. the **same data across several atlases** -- coronal, sagittal and the
   adult 3D coronal reference -- which all share the adult-mouse ontology, so
   one ``data`` dict works for every one.

Atlas geometry (section SVGs, the structure ontology, and the Swanson
polygons) is downloaded from the Allen / IBL public APIs on first run and
cached under ``~/.cache/nengoplotlib/atlas`` (override with
``NENGOPLOTLIB_CACHE_DIR``), so re-runs are fast and work offline.

Run with::

    python examples/plot_on_atlas_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nengoplotlib as npl


def main():
    rng = np.random.default_rng(0)

    # ---- 1) scalar fills on a posterior coronal section -----------------
    # Keys may be acronyms or full names; values given for an area propagate
    # to its drawn layer subregions.
    # (section 400 is posterior, so we use visual / retrosplenial areas that
    # are actually present there; regions absent from a section are skipped
    # with a warning rather than drawn.)
    scalar_data = {
        "VISp": 0.9,                   # primary visual area
        "VISpm": 0.6,                  # posteromedial visual area
        "VISpl": 0.7,                  # posterolateral visual area
        "RSPv": 0.3,                   # retrosplenial, ventral
        "RSPd": 0.45,                  # retrosplenial, dorsal
    }
    fig1, ax1, _ = npl.plot_on_atlas(
        "Mouse, P56, Coronal", scalar_data,
        section=400, cmap="magma", cbar_label="activity",
    )
    ax1.set_title("Scalar fills (coronal section 400)")

    # ---- 2) per-neuron arrays, three ways of laying them out ------------
    # The (n_neurons,) array on VISp is laid out *inside* the region outline.
    # "pcolormesh" is a plain grid; "som_hex" trains a self-organizing map
    # whose hex cells tile the region; "voronoi" gives one organic cell per
    # neuron. RSPv/RSPd stay solid scalars for context.
    array_data = {
        "VISp": rng.random(160),
        "RSPv": 0.2,
        "RSPd": 0.8,
    }
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, fill in zip(axes2, ["pcolormesh", "som_hex", "voronoi"]):
        npl.plot_on_atlas(1, array_data, section=400, cmap="viridis",
                          array_fill_type=fill, colorbar=False, ax=ax)
        ax.set_title(f"VISp array — {fill}")
    fig2.suptitle("Per-neuron array, shape-constrained layouts")

    # ---- 3) Swanson flat projection (scalar) ----------------------------
    swanson_data = {"Isocortex": 0.8, "DG": 0.2, "CA3": 0.35, "MOp": 0.5, "VISp": 1.0}
    fig3, ax3, _ = npl.plot_on_atlas(
        1, swanson_data, swanson=True, cmap="cividis",
    )
    ax3.set_title("Swanson flat projection")

    # ---- 4) the same data on several atlases ----------------------------
    # All three use structure graph 1 (the adult-mouse ontology), so the same
    # region names resolve everywhere. Each (atlas, section) below is chosen to
    # contain these areas; Isocortex propagates to whatever cortical
    # subregions that slice happens to draw.
    multi_data = {"Isocortex": 0.45, "RSPv": 0.95, "RSPd": 0.7}
    views = [
        ("Mouse, P56, Coronal", 402, "Coronal"),
        ("Mouse, P56, Sagittal", 130, "Sagittal"),
        ("Mouse, Adult, 3D Coronal", 903, "3D coronal"),
    ]
    fig4, axes = plt.subplots(1, len(views), figsize=(15, 5))
    for (atlas, section, title), ax in zip(views, axes):
        # Fix the colour scale across panels so the same value reads the same
        # colour in every atlas (and keep it off the colormap's black end).
        npl.plot_on_atlas(atlas, multi_data, section=section, cmap="viridis",
                          vmin=0.0, vmax=1.0, colorbar=False, ax=ax)
        ax.set_title(title)
    fig4.suptitle("Same data, several atlases")

    plt.show()


if __name__ == "__main__":
    main()
