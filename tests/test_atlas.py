"""Smoke / unit tests for nengoplotlib.atlas.plot_on_atlas.

The network layer (Allen RMA + IBL Swanson download) is monkeypatched to
return tiny hand-built fixtures, so these run fully offline.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nengoplotlib as npl
from nengoplotlib.atlas import api, fills, geom, svg
from matplotlib.collections import PatchCollection


# --------------------------------------------------------------- fixtures

# A 2x2-square SVG: structure 385 (left) and 985 (right).
FAKE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
  <g>
    <path structure_id="385" d="M0,0 L8,0 L8,8 L0,8 Z"/>
    <path structure_id="985" d="M10,0 L18,0 L18,8 L10,8 Z"/>
  </g>
</svg>"""

FAKE_ONTOLOGY = [
    {"id": "997", "acronym": "root", "name": "root",
     "structure_id_path": "/997/", "color_hex_triplet": "FFFFFF"},
    {"id": "315", "acronym": "Isocortex", "name": "Isocortex",
     "structure_id_path": "/997/315/", "color_hex_triplet": "70FF71"},
    {"id": "385", "acronym": "VISp", "name": "Primary visual area",
     "structure_id_path": "/997/315/385/", "color_hex_triplet": "08858C"},
    {"id": "985", "acronym": "MOp", "name": "Primary motor area",
     "structure_id_path": "/997/315/985/", "color_hex_triplet": "1F9D5A"},
]

FAKE_SWANSON = [
    {"thisID": 385, "hole": False,
     "coordsReg": {"x": [0, 8, 8, 0], "y": [0, 0, 8, 8]}},
    {"thisID": 985, "hole": False,
     "coordsReg": {"x": [10, 18, 18, 10], "y": [0, 0, 8, 8]}},
]


@pytest.fixture
def mock_allen(monkeypatch):
    monkeypatch.setattr(api, "resolve_atlas",
                        lambda atlas, **k: {"id": 1, "name": "Mouse, P56, Coronal",
                                            "structure_graph_id": 1})
    monkeypatch.setattr(api, "structure_ontology",
                        lambda graph_id, **k: FAKE_ONTOLOGY)
    monkeypatch.setattr(api, "list_atlas_images",
                        lambda atlas_id, **k: [
                            {"id": 100, "section_number": 100, "annotated": True},
                            {"id": 200, "section_number": 200, "annotated": True},
                            {"id": 300, "section_number": 300, "annotated": True},
                        ])
    monkeypatch.setattr(api, "download_svg", lambda image_id, **k: FAKE_SVG)
    monkeypatch.setattr(api, "download_swanson_paths", lambda **k: FAKE_SWANSON)


# --------------------------------------------------------------- svg parsing


def test_parse_svg_keys_by_structure_id():
    paths = svg.parse_svg(FAKE_SVG)
    assert set(paths) == {385, 985}
    # The left square spans x in [0, 8].
    (x0, _), (x1, _) = paths[385].get_extents().get_points()
    assert np.isclose(x0, 0) and np.isclose(x1, 8)


def test_parse_path_d_relative_and_curves():
    # absolute moveto then relative line + relative cubic
    verts, codes = svg.parse_path_d("M10,10 l5,0 c1,1 2,2 3,3 z")
    assert codes[0] == svg.Path.MOVETO
    assert codes[-1] == svg.Path.CLOSEPOLY
    # relative line endpoint is (15, 10)
    assert np.allclose(verts[1], [15, 10])


# --------------------------------------------------------------- scalar fills


def test_scalar_fill_runs(mock_allen):
    fig, ax, artists = npl.plot_on_atlas(
        "Mouse, P56, Coronal", {"VISp": 0.2, "MOp": 0.9}, colorbar=True)
    assert set(artists) == {385, 985}
    plt.close(fig)


def test_full_name_resolution(mock_allen):
    fig, ax, artists = npl.plot_on_atlas(1, {"Primary visual area": 1.0})
    assert 385 in artists
    plt.close(fig)


def test_parent_propagates_to_descendants(mock_allen):
    # A value on Isocortex (315) should fill both drawn children.
    fig, ax, artists = npl.plot_on_atlas(1, {"Isocortex": 0.5})
    assert set(artists) == {385, 985}
    plt.close(fig)


def test_unknown_region_warns_not_raises(mock_allen):
    with pytest.warns(UserWarning):
        fig, ax, artists = npl.plot_on_atlas(1, {"NotARegion": 0.5})
    assert artists == {}
    plt.close(fig)


def test_section_selection_by_number(mock_allen, monkeypatch):
    captured = {}

    def fake_svg(image_id, **k):
        captured["id"] = image_id
        return FAKE_SVG

    monkeypatch.setattr(api, "download_svg", fake_svg)
    fig, ax, _ = npl.plot_on_atlas(1, {"VISp": 0.1}, section=200)
    assert captured["id"] == 200
    plt.close(fig)


# --------------------------------------------------------------- array fills


@pytest.mark.parametrize("fill_type", ["pcolormesh", "pcolor"])
def test_array_fill_runs(mock_allen, fill_type):
    rng = np.random.default_rng(0)
    fig, ax, artists = npl.plot_on_atlas(
        1, {"VISp": rng.random(16)}, array_fill_type=fill_type)
    assert 385 in artists
    plt.close(fig)


def test_unknown_array_fill_type_raises(mock_allen):
    with pytest.raises(ValueError):
        npl.plot_on_atlas(1, {"VISp": np.arange(9)}, array_fill_type="nope")


def test_available_fills_lists_builtins():
    assert {"pcolormesh", "pcolor"} <= set(fills.available_fills())


# --------------------------------------------------------------- swanson


def test_swanson_scalar_runs(mock_allen):
    fig, ax, artists = npl.plot_on_atlas(1, {"VISp": 0.3, "MOp": 0.7}, swanson=True)
    assert set(artists) == {385, 985}
    plt.close(fig)


def test_swanson_array_raises(mock_allen):
    with pytest.raises(NotImplementedError):
        npl.plot_on_atlas(1, {"VISp": np.arange(16)}, swanson=True)


# ----------------------------------------------- SOM / Voronoi shape fills


def test_path_to_polygon_roundtrips_square():
    paths = svg.parse_svg(FAKE_SVG)
    poly = geom.path_to_polygon(paths[385])
    assert np.isclose(poly.area, 64.0, atol=1e-6)  # 8x8 fixture square


def test_grid_in_polygon_inside_only():
    paths = svg.parse_svg(FAKE_SVG)
    poly = geom.path_to_polygon(paths[385])
    pts = geom.grid_in_polygon(poly, topology="hex", n_target=40)
    assert len(pts) > 4
    from shapely.geometry import Point
    assert all(poly.contains(Point(p)) for p in pts)


@pytest.mark.parametrize("fill_type", ["som_hex", "som_rect", "voronoi",
                                       "voronoi_kmeans"])
def test_shape_fills_run(mock_allen, fill_type):
    rng = np.random.default_rng(0)
    fig, ax, artists = npl.plot_on_atlas(
        1, {"VISp": rng.random(60)}, array_fill_type=fill_type,
        random_state=0)
    assert 385 in artists
    assert isinstance(artists[385], PatchCollection)
    plt.close(fig)


def test_available_fills_lists_new_strategies():
    assert {"som_hex", "som_rect", "voronoi", "voronoi_kmeans"} <= set(
        fills.available_fills())


def test_features_dict_plumbs_through(mock_allen):
    rng = np.random.default_rng(1)
    feats = {"VISp": rng.standard_normal((60, 5))}
    fig, ax, artists = npl.plot_on_atlas(
        1, {"VISp": rng.random(60)}, array_fill_type="som_hex",
        features=feats, random_state=0)
    assert isinstance(artists[385], PatchCollection)
    plt.close(fig)


def test_positions_dict_plumbs_through_voronoi(mock_allen):
    rng = np.random.default_rng(2)
    pos = {"VISp": rng.random((60, 2))}
    fig, ax, artists = npl.plot_on_atlas(
        1, {"VISp": rng.random(60)}, array_fill_type="voronoi",
        positions=pos, random_state=0)
    assert isinstance(artists[385], PatchCollection)
    plt.close(fig)


def test_voronoi_kmeans_honors_n_clusters(mock_allen):
    rng = np.random.default_rng(3)
    fig, ax, artists = npl.plot_on_atlas(
        1, {"VISp": rng.random(80)}, array_fill_type="voronoi_kmeans",
        n_clusters=6, random_state=0)
    coll = artists[385]
    # at most n_clusters patches (empty clusters dropped)
    assert 0 < len(coll.get_paths()) <= 6
    plt.close(fig)


# ------------------------------------------------------- atlas animation

from matplotlib.animation import FuncAnimation


def test_fill_set_values_updates_scalar_patch():
    from nengoplotlib.atlas import cmap as cmap_mod
    paths = svg.parse_svg(FAKE_SVG)
    colormap, norm = cmap_mod.build_norm({"x": [0.0, 1.0]})
    fig, ax = plt.subplots()
    fill = fills.build_scalar_fill(ax, paths[385], colormap, norm)
    fill.set_values(1.0)
    np.testing.assert_allclose(fill.primary.get_facecolor(), colormap(norm(1.0)))
    plt.close(fig)


def test_build_array_fill_set_values_recolors():
    from nengoplotlib.atlas import cmap as cmap_mod, geom
    paths = svg.parse_svg(FAKE_SVG)
    poly = geom.path_to_polygon(paths[385])
    colormap, norm = cmap_mod.build_norm({"x": np.linspace(0, 1, 40)})
    fig, ax = plt.subplots()
    feats = np.arange(40, dtype=float)[:, None]
    fill = fills.build_array_fill(ax, paths[385], 40, colormap, norm,
                                  array_fill_type="voronoi", polygon=poly,
                                  features=feats, random_state=0)
    fill.set_values(np.random.default_rng(0).random(40))
    assert fill.primary.get_array() is not None
    plt.close(fig)


def test_atlas_animation_scalar_series(mock_allen):
    rng = np.random.default_rng(0)
    data = {"VISp": rng.random(15), "MOp": rng.random(15)}
    ani = npl.plot_atlas_animation(1, data, interval=50)
    assert isinstance(ani, FuncAnimation)
    assert len(list(ani.new_frame_seq())) == 15
    plt.close("all")


@pytest.mark.parametrize("fill", ["pcolormesh", "som_hex", "som_rect",
                                  "voronoi", "voronoi_kmeans"])
def test_atlas_animation_array_fill_types(mock_allen, fill):
    rng = np.random.default_rng(0)
    ani = npl.plot_atlas_animation(
        1, {"VISp": rng.random((8, 40))}, array_fill_type=fill, random_state=0)
    assert isinstance(ani, FuncAnimation)
    plt.close("all")


def test_atlas_animation_plot_step_frame_count(mock_allen):
    ani = npl.plot_atlas_animation(1, {"VISp": np.random.rand(20)}, plot_step=4)
    assert len(list(ani.new_frame_seq())) == 5
    plt.close("all")


def test_atlas_animation_mismatched_lengths_raises(mock_allen):
    with pytest.raises(ValueError):
        npl.plot_atlas_animation(1, {"VISp": np.zeros(10), "MOp": np.zeros(12)})


def test_atlas_animation_renders_frames_end_to_end(mock_allen):
    # to_jshtml drives init + every frame through the real FuncAnimation.
    data = {"VISp": np.random.default_rng(0).random((4, 16)), "MOp": np.zeros(4)}
    ani = npl.plot_atlas_animation(
        1, data, array_fill_type="pcolormesh", colorbar=False, figsize=(2, 2))
    html = ani.to_jshtml()
    assert isinstance(html, str) and len(html) > 0
    plt.close("all")


def test_atlas_animation_swanson_array_raises(mock_allen):
    with pytest.raises(NotImplementedError):
        npl.plot_atlas_animation(1, {"VISp": np.zeros((5, 10))}, swanson=True)


def test_atlas_animation_swanson_scalar_runs(mock_allen):
    ani = npl.plot_atlas_animation(1, {"VISp": np.random.rand(6)}, swanson=True)
    assert isinstance(ani, FuncAnimation)
    plt.close("all")
