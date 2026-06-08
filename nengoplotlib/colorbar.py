"""Slim colorbar for nengoplotlib figures.

:func:`colorbar` draws a thin vertical colorbar with min / median / max ticks
only -- the in-house alternative to ``fig.colorbar``'s boxy default.
"""

from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np

_FG = "#2a2422"


def colorbar(fig, ax, mappable, values=None, label="",
             *, cax_w=0.012, gap=0.015, height_frac=0.55,
             y_offset_frac=0.06, label_loc="right"):
    """Slim vertical colorbar with min / median / max ticks only.

    Parameters
    ----------
    fig, ax : Figure and Axes the colorbar relates to.
    mappable : a ScalarMappable / mesh / collection with set_array values.
    values : sequence used to compute min / median / max ticks. If None,
        the mappable's ``get_array()`` is used.
    label : axis label for the colorbar.
    label_loc : {'right', 'top'}
        ``'right'`` (default) renders the label as rotated vertical text on
        the right side of the strip -- the matplotlib-standard placement,
        safe even in short/wide figures. ``'top'`` puts it as a caption
        above the strip; nicer when there's headroom but it can collide
        with the plot above in short axes.
    """
    pos = ax.get_position()
    cax = fig.add_axes(
        [pos.x1 + gap, pos.y0 + y_offset_frac * pos.height,
         cax_w, pos.height * height_frac]
    )
    cb = fig.colorbar(mappable, cax=cax)
    if values is None:
        arr = mappable.get_array()
        values = np.asarray(arr).ravel()
    values = np.asarray(values)
    if values.size:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        vmed = float(np.median(values))
        ticks = sorted({vmin, vmed, vmax})
    else:
        ticks = []
    cb.set_ticks(ticks)
    fmt = (mticker.FormatStrFormatter("%.2f")
           if ticks and abs(ticks[-1]) < 10
           else mticker.FormatStrFormatter("%.0f"))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.tick_params(colors=_FG, labelsize=8, length=0, pad=4)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    cb.outline.set_visible(False)
    if label:
        if label_loc == "top":
            fig.text(
                pos.x1 + gap + cax_w / 2,
                pos.y0 + y_offset_frac * pos.height
                + pos.height * height_frac + 0.012,
                label, ha="center", va="bottom",
                fontsize=8.5, color=_FG, alpha=0.7,
            )
        else:
            cb.set_label(label, color=_FG, fontsize=8.5,
                         rotation=270, labelpad=10, alpha=0.85)
    return cb
