"""Nengo GUI integration.

:class:`ConnectomePlot` is a ``nengo.Node`` subclass that renders its parent
network's connectome as inline SVG and exposes it via ``output._nengo_html_``
-- the attribute Nengo GUI looks for when drawing a node's panel.

The connectome doesn't change during simulation, so the SVG is built once on
the first step and cached on the node's output callable; subsequent steps
serve the same string with no matplotlib re-render.

Usage
-----

>>> import nengo
>>> from connectomes import ConnectomePlot
>>>
>>> with model:
...     ConnectomePlot(label="connectome",
...                    size_by='n_neurons', equalize_at=0)

Pass any :func:`plot_connectome` keyword via ``**plot_kwargs``.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import matplotlib.pyplot as plt
import nengo

from .api import plot_connectome


_LOADING_SVG = (
    '<svg width="100%" height="100%" viewBox="0 0 100 60" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<text x="50" y="30" text-anchor="middle" '
    'style="font-size:5px;fill:gray">rendering connectome...</text>'
    "</svg>"
)


class ConnectomePlot(nengo.Node):
    """Display a connectome plot inside the Nengo GUI.

    Add inside a ``with network:`` block; the surrounding network is captured
    via :attr:`nengo.Network.context` so the node knows what to plot. Override
    by passing ``network=...`` explicitly.

    Parameters
    ----------
    network : nengo.Network, optional
        The network to plot. Defaults to the innermost active
        ``with network:`` context.
    label : str, optional
        Node label shown in the GUI.
    figsize : tuple of (float, float), default (10, 6)
        Matplotlib figure size (inches) used to build the SVG. The resulting
        SVG is then resized to fill its container, so this only controls the
        plot's internal aspect ratio.
    **plot_kwargs
        Forwarded to :func:`plot_connectome` (e.g. ``size_by``,
        ``equalize_at``, ``connection_type``, ``labels``, ``arc_params``).
    """

    def __init__(
        self,
        network: Optional[nengo.Network] = None,
        label: Optional[str] = None,
        figsize=(10, 6),
        **plot_kwargs,
    ):
        if network is None:
            ctx = nengo.Network.context
            if not ctx:
                raise RuntimeError(
                    "ConnectomePlot must be created inside a `with network:` "
                    "block, or have an explicit network= passed in."
                )
            network = ctx[-1]

        self._network = network
        self._figsize = figsize
        self._plot_kwargs = plot_kwargs

        # Capture self into a closure -- the function is what nengo calls each
        # step, and the function object is also where the GUI reads
        # `_nengo_html_` from.
        def update(t):
            if not getattr(update, "_rendered", False):
                update._nengo_html_ = self._build_svg()
                update._rendered = True

        super().__init__(update, size_in=0, size_out=0, label=label)
        # Placeholder until the first sim step fires update().
        self.output._nengo_html_ = _LOADING_SVG

    def _build_svg(self) -> str:
        fig, ax = plt.subplots(figsize=self._figsize)
        try:
            plot_connectome(self._network, ax=ax, **self._plot_kwargs)
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
        finally:
            plt.close(fig)
        return _make_responsive(buf.getvalue().decode("utf-8"))


# --------------------------------------------------------------------------- #
# SVG post-processing
# --------------------------------------------------------------------------- #

def _make_responsive(svg: str) -> str:
    """Strip the XML preamble and force the root ``<svg>`` to fill its parent.

    Matplotlib's SVG output uses fixed pt dimensions; for embedding into a
    Nengo GUI panel we want the SVG to scale to whatever container it's
    placed in (preserving aspect ratio via the inherent ``viewBox``).
    """
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)

    def _rewrite_root(m: re.Match) -> str:
        attrs = m.group(1)
        attrs = re.sub(r'\swidth="[^"]+"', ' width="100%"', attrs, count=1)
        attrs = re.sub(r'\sheight="[^"]+"', ' height="100%"', attrs, count=1)
        return f"<svg{attrs}>"

    return re.sub(r"<svg([^>]*)>", _rewrite_root, svg, count=1)
