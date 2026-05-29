"""Swanson flat-projection geometry.

The Swanson projection flattens the whole mouse brain into a single 2-D
diagram, so it is an alternative to picking one coronal/sagittal slice: every
region is visible at once. We use the vectorized polygons published by the
International Brain Lab (see :func:`api.download_swanson_paths`), whose
``thisID`` field is the Allen ``structure_id`` -- letting us reuse the same
ontology resolver as the sliced-atlas view.

This module only builds geometry (``{structure_id: Path}``); colour fills are
applied by the shared loop in :mod:`nengoplotlib.atlas.plot`. Array fills on
the flatmap are not supported yet -- the orchestrator raises for those.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from matplotlib.path import Path

from . import api
from .svg import polygons_to_path


def region_paths(*, refresh: bool = False) -> Dict[int, Path]:
    """Return ``{structure_id: compound Path}`` for the Swanson projection.

    All polygons sharing a ``thisID`` (a region plus any islands/holes) are
    merged into one compound path. Y is negated so the diagram plots upright
    under matplotlib's default (y-up) axes.
    """
    rings: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for entry in api.download_swanson_paths(refresh=refresh):
        sid = int(entry["thisID"])
        coords = entry["coordsReg"]
        # ``coordsReg`` is either a single {x, y} ring or a list of such rings
        # (regions split across the projection into several pieces).
        for ring in (coords if isinstance(coords, list) else [coords]):
            x = np.asarray(ring["x"], dtype=float)
            y = -np.asarray(ring["y"], dtype=float)
            if x.size:
                rings.setdefault(sid, []).append((x, y))
    return {sid: polygons_to_path(polys) for sid, polys in rings.items()}
