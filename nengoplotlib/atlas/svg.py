"""Parse Allen structure-boundary SVGs into matplotlib ``Path`` objects.

The Allen ``svg_download`` endpoint returns one ``<path>`` per drawn region,
each carrying a ``structure_id`` attribute and an SVG ``d`` path string that
mixes absolute and relative move/line/cubic commands (``M m L l C c S s
V v H h Z z`` are all observed). A single structure can appear as several
``<path>`` elements (e.g. left/right hemisphere, or islands), so paths are
merged by ``structure_id`` into one compound :class:`~matplotlib.path.Path`.

Coordinates are returned in the SVG's own pixel frame (y increasing
downward); the caller is responsible for inverting the y-axis when plotting.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.path import Path

# Tokenize a ``d`` string into command letters and numbers. The number
# pattern follows the SVG grammar so that packed forms split correctly:
# ``117.785-7.511`` -> two numbers (the ``-`` is a separator), ``.5.5`` ->
# ``.5`` ``.5``, and ``1e-3`` stays one number.
_NUMBER = r"[+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(" + _NUMBER + r")")


def _tokenize(d: str) -> List[str]:
    out: List[str] = []
    for cmd, num in _TOKEN.findall(d):
        out.append(cmd if cmd else num)
    return out


def parse_path_d(d: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one SVG ``d`` string to ``(vertices, codes)`` arrays.

    Cubic beziers (C/S) become ``CURVE4`` segments, quadratics (Q/T) become
    ``CURVE3``, lines/moves map directly, and ``Z`` closes the current
    subpath. Unsupported elliptical arcs (``A``) are approximated by a line
    to their endpoint -- they do not occur in Allen atlas SVGs.
    """
    toks = _tokenize(d)
    verts: List[Tuple[float, float]] = []
    codes: List[int] = []
    i = 0
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd = ""
    prev_ctrl = None  # last cubic/quadratic control point, for S/T reflection

    def num() -> float:
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        tok = toks[i]
        if re.match(r"[A-Za-z]", tok):
            cmd = tok
            i += 1
            if cmd in "Zz":
                if verts:
                    verts.append(start)
                    codes.append(Path.CLOSEPOLY)
                cur = start
                prev_ctrl = None
                continue
        rel = cmd.islower()
        c = cmd.upper()

        if c == "M":
            x, y = num(), num()
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = start = (x, y)
            verts.append(cur)
            codes.append(Path.MOVETO)
            cmd = "l" if rel else "L"  # subsequent coords are implicit lineto
            prev_ctrl = None
        elif c == "L":
            x, y = num(), num()
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = (x, y)
            verts.append(cur)
            codes.append(Path.LINETO)
            prev_ctrl = None
        elif c == "H":
            x = num()
            x = cur[0] + x if rel else x
            cur = (x, cur[1])
            verts.append(cur)
            codes.append(Path.LINETO)
            prev_ctrl = None
        elif c == "V":
            y = num()
            y = cur[1] + y if rel else y
            cur = (cur[0], y)
            verts.append(cur)
            codes.append(Path.LINETO)
            prev_ctrl = None
        elif c in ("C", "S"):
            if c == "C":
                c1 = (num(), num())
                c2 = (num(), num())
                end = (num(), num())
                if rel:
                    c1 = (cur[0] + c1[0], cur[1] + c1[1])
                    c2 = (cur[0] + c2[0], cur[1] + c2[1])
                    end = (cur[0] + end[0], cur[1] + end[1])
            else:  # smooth cubic: first control is reflection of previous
                c2 = (num(), num())
                end = (num(), num())
                if rel:
                    c2 = (cur[0] + c2[0], cur[1] + c2[1])
                    end = (cur[0] + end[0], cur[1] + end[1])
                if prev_ctrl is not None:
                    c1 = (2 * cur[0] - prev_ctrl[0], 2 * cur[1] - prev_ctrl[1])
                else:
                    c1 = cur
            verts.extend([c1, c2, end])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            cur = end
            prev_ctrl = c2
        elif c in ("Q", "T"):
            if c == "Q":
                c1 = (num(), num())
                end = (num(), num())
                if rel:
                    c1 = (cur[0] + c1[0], cur[1] + c1[1])
                    end = (cur[0] + end[0], cur[1] + end[1])
            else:  # smooth quadratic
                end = (num(), num())
                if rel:
                    end = (cur[0] + end[0], cur[1] + end[1])
                if prev_ctrl is not None:
                    c1 = (2 * cur[0] - prev_ctrl[0], 2 * cur[1] - prev_ctrl[1])
                else:
                    c1 = cur
            verts.extend([c1, end])
            codes.extend([Path.CURVE3, Path.CURVE3])
            cur = end
            prev_ctrl = c1
        elif c == "A":  # arc: approximate by a straight line to the endpoint
            num(), num(), num(), num(), num()
            x, y = num(), num()
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = (x, y)
            verts.append(cur)
            codes.append(Path.LINETO)
            prev_ctrl = None
        else:
            i += 1  # skip anything unrecognized

    return np.asarray(verts, dtype=float), np.asarray(codes, dtype=np.uint8)


def parse_svg(svg_text: str) -> Dict[int, Path]:
    """Parse an Allen boundary SVG into ``{structure_id: compound Path}``."""
    root = ET.fromstring(svg_text)
    by_id: Dict[int, Tuple[List, List]] = {}
    for el in root.iter():
        if not el.tag.endswith("path"):
            continue
        sid = el.get("structure_id")
        d = el.get("d")
        if sid is None or not d:
            continue
        verts, codes = parse_path_d(d)
        if len(verts) == 0:
            continue
        v, c = by_id.setdefault(int(sid), ([], []))
        v.append(verts)
        c.append(codes)

    paths: Dict[int, Path] = {}
    for sid, (vlist, clist) in by_id.items():
        paths[sid] = Path(np.vstack(vlist), np.concatenate(clist))
    return paths


def polygons_to_path(polys: List[Tuple[np.ndarray, np.ndarray]]) -> Path:
    """Build one compound ``Path`` from a list of ``(x, y)`` polygon arrays.

    Each polygon is emitted as a closed subpath. Used by the Swanson view,
    whose regions arrive as raw coordinate rings rather than SVG strings.
    """
    verts: List[np.ndarray] = []
    codes: List[np.ndarray] = []
    for x, y in polys:
        n = len(x)
        if n == 0:
            continue
        ring = np.column_stack([x, y])
        ring = np.vstack([ring, ring[0]])  # explicit closing vertex
        cs = np.full(n + 1, Path.LINETO, dtype=np.uint8)
        cs[0] = Path.MOVETO
        cs[-1] = Path.CLOSEPOLY
        verts.append(ring)
        codes.append(cs)
    return Path(np.vstack(verts), np.concatenate(codes))
