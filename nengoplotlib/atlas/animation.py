"""Animate per-region activity on a brain atlas over time.

:func:`plot_atlas_animation` is the temporal counterpart of
:func:`nengoplotlib.atlas.plot.plot_on_atlas`. Each region's data is now a
*time series*:

* ``(n_timesteps,)``            -- one scalar per frame; the region is a solid
  patch whose colour tracks the value.
* ``(n_timesteps, n_neurons)``  -- per-neuron activity; the region is laid out
  once (grid / shape-constrained SOM / Voronoi, via ``array_fill_type``) and the
  cells are recoloured every frame.

The geometry is built a single time through :mod:`nengoplotlib.atlas.fills`
builders, so a frame is identical to the equivalent static ``plot_on_atlas``
call. Returns a :class:`matplotlib.animation.FuncAnimation`; call ``.save(...)``
or ``.to_jshtml()`` on it.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from scipy.signal import lfilter

from . import cmap as cmap_mod, fills
from .plot import _match_geometry, _safe_polygon, load_atlas


def _lowpass(X, dt, tau):
    """One-pole IIR lowpass along axis 0 (matches ``nengo.synapses.Lowpass``)."""
    if tau is None or tau <= 0:
        return X
    alpha = dt / (dt + tau)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], X, axis=0)


def _series_length(data: Mapping) -> int:
    """Validate every value shares one leading time axis; return its length."""
    lengths = set()
    for region, v in data.items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"data[{region!r}] must be (n_timesteps,) or "
                f"(n_timesteps, n_neurons); got shape {arr.shape}"
            )
        lengths.add(arr.shape[0])
    if len(lengths) != 1:
        raise ValueError(
            f"all data series must share the same n_timesteps; got {sorted(lengths)}"
        )
    return lengths.pop()


def plot_atlas_animation(
    atlas: Union[int, str],
    data: Mapping,
    *,
    section: Optional[Union[int, float]] = None,
    swanson: bool = False,
    array_fill_type: str = "pcolormesh",
    features: Optional[Mapping] = None,
    positions: Optional[Mapping] = None,
    t=None,
    dt: Optional[float] = None,
    tau: float = 0.0,
    plot_step: int = 1,
    cmap="viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edgecolor: str = "k",
    outline_lw: float = 0.5,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    interval: int = 100,
    figsize=(8, 6),
    dpi: Optional[int] = None,
    blit: bool = False,
    ax=None,
    refresh: bool = False,
    **fill_kwargs,
) -> FuncAnimation:
    """Animate time-varying per-region data on an Allen atlas.

    Parameters
    ----------
    atlas, section, swanson, array_fill_type, features, positions, cmap, vmin,
    vmax, edgecolor, outline_lw, colorbar, cbar_label, refresh, **fill_kwargs
        As in :func:`nengoplotlib.atlas.plot.plot_on_atlas`. ``features`` /
        ``positions`` are *static* (they fix the layout); when ``features`` is
        omitted, each region's per-neuron **mean over time** drives its layout.
    data : dict
        ``{region: array}`` where each array is ``(n_timesteps,)`` (scalar
        activity, solid fill) or ``(n_timesteps, n_neurons)`` (per-neuron
        activity, laid out inside the region). All series must share
        ``n_timesteps``.
    t : (n_timesteps,) array, optional
        Time stamps; used to derive ``dt`` for the synaptic filter.
    dt : float, optional
        Timestep (seconds). Defaults to ``t`` spacing, else ``1.0``.
    tau : float
        Synapse time constant (seconds) for an optional one-pole lowpass of the
        activity before display. ``0`` (default) shows raw values.
    plot_step : int
        Timesteps advanced per animation frame.
    interval : int
        Milliseconds between frames.
    figsize, dpi, ax, blit
        Standard matplotlib figure controls.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if not data:
        raise ValueError("data is empty")
    n_time = _series_length(data)

    if dt is None:
        if t is not None:
            t = np.asarray(t, dtype=float)
            dt = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0
        else:
            dt = 1.0

    # Filter every series up front; the shared colour scale and the per-frame
    # colours both read from the filtered values.
    filtered = {r: _lowpass(np.asarray(v, dtype=float), dt, tau)
                for r, v in data.items()}

    ont, geometry, invert_y = load_atlas(
        atlas, section=section, swanson=swanson, refresh=refresh)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig = ax.figure

    # Region outlines (static).
    ax.add_collection(
        PathCollection(list(geometry.values()), facecolors="none",
                       edgecolors=edgecolor, linewidths=outline_lw)
    )

    # Shared colour scale over every (filtered) value across all frames.
    colormap, norm = cmap_mod.build_norm(filtered, cmap=cmap, vmin=vmin, vmax=vmax)

    geom_keys = set(geometry)
    updaters = []  # (Fill, series, is_scalar)
    for region, series in filtered.items():
        try:
            sid = ont.resolve(region)
        except KeyError as exc:
            warnings.warn(str(exc))
            continue
        targets = _match_geometry(ont, geom_keys, sid)
        if not targets:
            warnings.warn(f"region {region!r} (id {sid}) not drawn in this view")
            continue
        scalar = series.ndim == 1
        if not scalar and swanson:
            raise NotImplementedError(
                "array fills are not supported on the Swanson view yet; "
                "pass a (n_timesteps,) scalar series or use a sliced atlas"
            )
        region_feats = (features or {}).get(region)
        region_pos = (positions or {}).get(region)
        for gid in targets:
            path = geometry[gid]
            if scalar:
                fill = fills.build_scalar_fill(ax, path, colormap, norm,
                                               edgecolor=edgecolor)
            else:
                n_neurons = series.shape[1]
                # Static layout features: caller's, else per-neuron mean activity.
                feats = (np.asarray(region_feats, dtype=float)
                         if region_feats is not None
                         else series.mean(axis=0)[:, None])
                fill = fills.build_array_fill(
                    ax, path, n_neurons, colormap, norm,
                    array_fill_type=array_fill_type, polygon=_safe_polygon(path),
                    features=feats, positions=region_pos, edgecolor=edgecolor,
                    **fill_kwargs)
            updaters.append((fill, series, scalar))

    # Frame the plot (matches plot_on_atlas).
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()

    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)

    def _draw(idx):
        idx = min(idx, n_time - 1)
        for fill, series, scalar in updaters:
            fill.set_values(series[idx] if scalar else series[idx, :])
        return [f.primary for f, _, _ in updaters]

    def init():
        return _draw(0)

    def update(frame):
        return _draw(frame * plot_step)

    n_frames = int(np.ceil(n_time / plot_step))
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=blit, repeat=False)
    return ani
