"""Resolve user-supplied region names to Allen ``structure_id`` values.

The ``data`` dict passed to ``plot_on_atlas`` is keyed by whatever names are
natural to the user -- acronyms (``"VISp"``) or full names (``"Primary visual
area"``). :class:`Ontology` wraps the Allen ``Structure`` table to translate
those into the integer ids that appear in the SVG (and Swanson) geometry, and
to walk the structure hierarchy so that a value given for a *parent* region
can fill all of its drawn descendants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from . import api


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


class Ontology:
    """Lookup helper over one Allen structure graph."""

    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.by_id: Dict[int, dict] = {}
        self._acronym: Dict[str, int] = {}
        self._name: Dict[str, int] = {}
        self._color: Dict[int, str] = {}
        for r in rows:
            sid = int(r["id"])
            self.by_id[sid] = r
            self._acronym[_norm(r["acronym"])] = sid
            self._name[_norm(r["name"])] = sid
            self._color[sid] = r.get("color_hex_triplet") or "FFFFFF"

    @classmethod
    def for_graph(cls, graph_id: int, *, refresh: bool = False) -> "Ontology":
        return cls(api.structure_ontology(graph_id, refresh=refresh))

    def resolve(self, region) -> int:
        """Map an acronym, full name, or integer id to a ``structure_id``."""
        if isinstance(region, int) or (isinstance(region, str) and region.isdigit()):
            sid = int(region)
            if sid in self.by_id:
                return sid
            raise KeyError(f"no structure with id {sid} in this atlas")
        key = _norm(region)
        if key in self._acronym:
            return self._acronym[key]
        if key in self._name:
            return self._name[key]
        raise KeyError(
            f"region {region!r} not found (tried acronym and full-name match)"
        )

    def color(self, structure_id: int) -> str:
        """Allen's canonical ``#RRGGBB`` color for a structure."""
        return "#" + self._color.get(structure_id, "FFFFFF")

    def acronym(self, structure_id: int) -> Optional[str]:
        row = self.by_id.get(structure_id)
        return row["acronym"] if row else None

    def descendants(self, structure_id: int) -> Set[int]:
        """All structure ids whose ``structure_id_path`` contains ``structure_id``.

        Includes ``structure_id`` itself. Lets a value assigned to a coarse
        region (e.g. ``"Isocortex"``) propagate to every finer subregion that
        is actually drawn in the geometry.
        """
        token = f"/{structure_id}/"
        out: Set[int] = set()
        for r in self.rows:
            path = r.get("structure_id_path") or ""
            if token in path:
                out.add(int(r["id"]))
        return out
