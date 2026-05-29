"""Thin client for the Allen Brain Atlas RMA API (plus the IBL Swanson file).

Only the handful of endpoints ``plot_on_atlas`` needs are wrapped here, each
returning plain Python data structures and each cached to disk via
:mod:`nengoplotlib.atlas.cache`.

A note on RMA option syntax
---------------------------
Paging / ordering options are passed *inline* in the ``criteria`` parameter
(``model::X,rma::criteria,<filter>,rma::options[...]``) rather than as a
separate ``&rma::options=`` query parameter. Both are valid RMA, but the
inline form survives proxies that strip unknown query parameters, so it is the
more portable choice.

The default Allen page size is 50 rows; :func:`_query_json_all` pages through
``num_rows``/``start_row`` until everything is collected.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Optional, Union

from . import cache

_RMA = "http://api.brain-map.org/api/v2/data/query"
_SVG = "http://api.brain-map.org/api/v2/svg_download"
# Public IBL data bucket; the vectorized Swanson projection lives here.
_SWANSON_URL = "https://ibl-brain-wide-map-public.s3.amazonaws.com/atlas/swansonpaths.json"

# GraphicGroupLabel id 28 == structure boundary drawings.
STRUCTURE_GRAPHIC_GROUP = 28

_PAGE = 200


def _query_json_all(model: str, criteria: str = "", order: Optional[str] = None,
                    *, refresh: bool = False) -> list:
    """Run an RMA ``query.json`` and page through *all* matching rows."""
    crit = f",rma::criteria,{criteria}" if criteria else ""
    order_opt = f"[order$eq'{order}']" if order else ""
    rows: list = []
    start = 0
    while True:
        url = (f"{_RMA}.json?criteria=model::{model}{crit},"
               f"rma::options[num_rows$eq{_PAGE}][start_row$eq{start}]{order_opt}")
        payload = json.loads(cache.cached_text(url, suffix=".json", refresh=refresh))
        if not payload.get("success", False):
            raise RuntimeError(f"Allen RMA query failed: {payload.get('msg')!r}")
        batch = payload.get("msg", [])
        rows.extend(batch)
        if len(batch) < _PAGE:
            return rows
        start += len(batch)


def list_atlases(*, refresh: bool = False) -> list:
    """Return every ``Atlas`` record (id, name, ``structure_graph_id``, ...)."""
    return _query_json_all("Atlas", refresh=refresh)


def resolve_atlas(atlas: Union[int, str], *, refresh: bool = False) -> dict:
    """Resolve an atlas id or (sub)string name to its full ``Atlas`` record.

    Name matching is case-insensitive: an exact (normalized) match wins,
    otherwise a unique substring match is accepted; ambiguous or absent names
    raise ``ValueError`` listing the choices.
    """
    atlases = list_atlases(refresh=refresh)
    if isinstance(atlas, int) or (isinstance(atlas, str) and atlas.isdigit()):
        wanted = int(atlas)
        for rec in atlases:
            if rec["id"] == wanted:
                return rec
        raise ValueError(f"no Allen atlas with id {wanted}")

    def norm(s: str) -> str:
        return "".join(s.lower().split()).replace(",", "")

    target = norm(atlas)
    exact = [r for r in atlases if norm(r["name"]) == target]
    if len(exact) == 1:
        return exact[0]
    subs = [r for r in atlases if target in norm(r["name"])]
    if len(subs) == 1:
        return subs[0]
    names = ", ".join(repr(r["name"]) for r in atlases)
    if not subs:
        raise ValueError(f"no Allen atlas matching {atlas!r}. Available: {names}")
    raise ValueError(
        f"atlas {atlas!r} is ambiguous; matches "
        f"{', '.join(repr(r['name']) for r in subs)}"
    )


def list_atlas_images(atlas_id: int, *, annotated_only: bool = True,
                      refresh: bool = False) -> list:
    """Return ``AtlasImage`` records for an atlas, ordered by section number.

    Only annotated sections carry structure-boundary SVGs, so by default the
    unannotated (raw Nissl) sections are filtered out.
    """
    images = _query_json_all(
        "AtlasImage",
        f"atlas_data_set(atlases[id$eq{atlas_id}])",
        order="sub_images.section_number",
        refresh=refresh,
    )
    if annotated_only:
        images = [im for im in images if im.get("annotated")]
    return images


def download_svg(image_id: int, *, groups: int = STRUCTURE_GRAPHIC_GROUP,
                 refresh: bool = False) -> str:
    """Download the structure-boundary SVG for one ``AtlasImage`` as text."""
    url = f"{_SVG}/{image_id}?groups={groups}"
    return cache.cached_text(url, suffix=".svg", refresh=refresh)


def structure_ontology(graph_id: int, *, refresh: bool = False) -> list:
    """Return ``Structure`` rows for a structure graph as a list of dicts.

    Each row includes at least ``id``, ``acronym``, ``name``,
    ``structure_id_path`` and ``color_hex_triplet``.
    """
    # CSV (rather than JSON) keeps the payload compact for the full ontology.
    # ``num_rows$eqall`` returns the whole graph in one response.
    url = (f"{_RMA}.csv?criteria=model::Structure,rma::criteria,[graph_id$eq{graph_id}],"
           f"rma::options[num_rows$eqall][order$eq'structures.graph_order']")
    text = cache.cached_text(url, suffix=".csv", refresh=refresh)
    return list(csv.DictReader(io.StringIO(text)))


def download_swanson_paths(*, refresh: bool = False) -> list:
    """Download the vectorized Swanson projection polygons (IBL public data).

    Returns a list of ``{"thisID": <allen structure id>, "hole": bool,
    "coordsReg": {"x": [...], "y": [...]}}`` dicts.
    """
    return json.loads(cache.cached_text(_SWANSON_URL, suffix=".json", refresh=refresh))
