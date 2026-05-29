"""Build a single colour scale shared by every region in a plot.

Whether a region's value is a scalar (solid fill) or a ``(n_neurons,)`` array
(per-neuron mesh), all of them must read against the *same* colormap and
normalization so that the colorbar means one thing across the whole figure.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import colors


def _resolve_cmap(cmap):
    """Return a Colormap from a name or pass through an existing Colormap."""
    if isinstance(cmap, str):
        try:
            return mpl.colormaps[cmap]
        except (KeyError, AttributeError):  # very old matplotlib
            from matplotlib import cm
            return cm.get_cmap(cmap)
    return cmap


def collect_values(data: Mapping) -> np.ndarray:
    """Flatten every finite value across the mixed scalar/array ``data`` dict."""
    vals = []
    for v in data.values():
        arr = np.asarray(v, dtype=float).ravel()
        vals.append(arr)
    if not vals:
        return np.asarray([], dtype=float)
    allv = np.concatenate(vals)
    return allv[np.isfinite(allv)]


def build_norm(data: Mapping, cmap="viridis",
               vmin: Optional[float] = None,
               vmax: Optional[float] = None) -> Tuple[colors.Colormap, colors.Normalize]:
    """Return ``(colormap, Normalize)`` spanning all values in ``data``.

    ``vmin``/``vmax`` override the data-derived bounds. A degenerate range
    (all values equal, or no values) is nudged to ``[v-0.5, v+0.5]`` so
    matplotlib does not warn or divide by zero.
    """
    colormap = _resolve_cmap(cmap)
    finite = collect_values(data)
    lo = vmin if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    hi = vmax if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        mid = lo if np.isfinite(lo) else 0.0
        lo, hi = mid - 0.5, mid + 0.5
    return colormap, colors.Normalize(vmin=lo, vmax=hi)
