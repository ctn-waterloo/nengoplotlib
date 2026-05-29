"""Visual style helpers shared across nengoplotlib figures.

Use :func:`apply_style` once at the top of a script (or a notebook cell) to
set matplotlib rcParams. Use :func:`add_thin_colorbar` to draw a slim
vertical colorbar with min/median/max ticks only — the in-house alternative
to ``fig.colorbar``'s boxy default.

Constants
---------
``BG_COLOR``
    White background for legend
``FG_COLOR``
    Default text / axis color.
``EDGE_COLOR``
    white
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BG_COLOR = "#ffffff"
FG_COLOR = "#2a2422"
EDGE_COLOR = "#ffffff"
RULE_COLOR = "#ffffff"


def apply_style():
    """Apply the nengoplotlib rcParams (font stack, axes colors)."""
    plt.rcParams.update({
        "font.family": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
        "axes.labelcolor": FG_COLOR,
        "axes.edgecolor": FG_COLOR,
        "xtick.color": FG_COLOR,
        "ytick.color": FG_COLOR,
        "text.color": FG_COLOR,
        "axes.titlecolor": FG_COLOR,
        "savefig.transparent": False,
    })


def activity_cmap(base="inferno", lo=0.02, hi=0.92, n=256):
    cmap = plt.get_cmap(base)
    return cmap
    # colors = cmap(np.linspace(lo, hi, n))
    # return mcolors.LinearSegmentedColormap.from_list(f"{base}_trim", colors)




def title_block(ax, title, subtitle=None, *, pad=22, fontsize=14,
                 subtitle_fontsize=9):
    """Two-line title: bold heading + faded subtitle."""
    ax.set_title(title, color=FG_COLOR, fontsize=fontsize, loc="left",
                 pad=pad, fontweight="semibold")
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=subtitle_fontsize,
                color=FG_COLOR, alpha=0.55)


def add_title_rule(fig, ax, color=RULE_COLOR):
    """Thin horizontal rule below the title block — pure decoration."""
    pos = ax.get_position()
    y = pos.y1 + 0.005
    rule = plt.Line2D(
        [pos.x0, pos.x1], [y, y], transform=fig.transFigure,
        color=color, linewidth=0.6, alpha=0.9, zorder=10,
    )
    fig.add_artist(rule)


def add_thin_colorbar(fig, ax, mappable, values=None, label="",
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
        the right side of the strip — the matplotlib-standard placement,
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
    cb.ax.tick_params(colors=FG_COLOR, labelsize=8, length=0, pad=4)
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
                fontsize=8.5, color=FG_COLOR, alpha=0.7,
            )
        else:
            cb.set_label(label, color=FG_COLOR, fontsize=8.5,
                         rotation=270, labelpad=10, alpha=0.85)
    return cb


def style_legend(legend):
    """Apply the nengoplotlib palette to a matplotlib Legend."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(BG_COLOR)
    frame.set_edgecolor(RULE_COLOR)
    frame.set_linewidth(0.6)
    for text in legend.get_texts():
        text.set_color(FG_COLOR)
        text.set_fontsize(8.5)
