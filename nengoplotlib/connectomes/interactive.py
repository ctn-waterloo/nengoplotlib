"""Click-driven interaction for connectome plots.

A thin matplotlib event handler that lets users click wedges or connection
lines on a rendered connectome to highlight them and optionally pop up an
inset time-series plot. Operates on the ``plot_objs`` dict returned by
:func:`plot_connectome`.

Backend requirement
-------------------
This class hooks ``button_press_event`` on the figure's canvas, so it only
works with an interactive matplotlib backend:

* **Jupyter notebook / Lab**: enable widget mode -- ``%matplotlib widget``
  (requires the ``ipympl`` package: ``pip install ipympl``) or
  ``%matplotlib qt`` / ``tk`` for a pop-out window. The default Jupyter
  ``inline`` backend produces a static PNG and will *not* respond to clicks.
* **Plain Python scripts**: any GUI backend (``QtAgg``, ``TkAgg``,
  ``MacOSX``, ``GTK3Agg``, ...). Non-interactive backends like ``agg``,
  ``pdf``, ``svg``, etc. silently ignore the click handler.
"""

from __future__ import annotations

import matplotlib.text as _mtext
import numpy as np

from .keys import display_label


class InteractiveConnectome:
    """Click handler for a rendered connectome plot.

    Wires up a single ``button_press_event`` callback on ``fig.canvas``
    that handles three kinds of clicks:

    * **Wedge** (inside the data axes) -- toggles selection: a white 3-px
      outline and raised z-order. A yellow hover-text popup with the
      wedge's label appears at the click location, and (if ``times`` /
      ``signals`` are supplied) the upper-right corner shows an inset
      time-series of that ensemble's signal.
    * **Connection arc** (inside the data axes) -- toggles selection:
      linewidth bumped 5x and alpha forced to 1.0. Hover text shows the
      ``"src -> dst"`` label.
    * **Legend label** (anywhere in the figure) -- same effect as clicking
      the corresponding wedge. Every wedge sharing that label is toggled
      as a group (a leaf padded across multiple rings has several wedges
      with the same label, all of which select / deselect together) and
      the inset signal plot pops up. No separate hover text since the
      legend label is itself the name.

    Clicking the same wedge / line / label again deselects it and clears
    its popups. Clicking on empty space inside the axes dismisses all
    popups but leaves existing selections intact.

    Legend-label clicks are detected by hit-testing each tagged Text
    artist's bbox; no ``pick_event`` is involved, so this works across
    matplotlib versions and backends (including ipympl).

    Requires an **interactive matplotlib backend** (see module docstring).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure returned by :func:`plot_connectome`.
    ax : matplotlib.axes.Axes
        The axes the connectome is drawn into. Clicks outside this axes are
        ignored.
    wedges : list of (Wedge, str)
        The wedge entries from ``plot_objs['wedges']`` -- pairs of a
        ``matplotlib.patches.Wedge`` and its node label string (as produced
        by :func:`get_key`).
    lines : list of (Line2D | PathPatch, str)
        The connection-arc entries from ``plot_objs['lines']`` -- pairs of
        a matplotlib line / path object and a ``"src -> dst"`` label.
    times : np.ndarray, optional
        Time vector for inset plots. If ``None``, clicking a wedge skips
        the inset and just highlights / shows the hover text.
    signals : dict[str, np.ndarray], optional
        Per-ensemble time-series, keyed by the same label strings used in
        ``wedges``. ``get_correlation_matrix`` returns a ready-made
        ``signals_dict`` in this shape. Missing keys are skipped with a
        print message.

    Attributes
    ----------
    selected_wedges, selected_lines : set[int]
        Currently highlighted indices into ``wedges`` / ``lines``.

    Example
    -------
    >>> %matplotlib widget   # in Jupyter; or use a GUI backend in scripts
    >>> ax, fig, plot_objs = plot_connectome(model)
    >>> corr, ens, signals = get_correlation_matrix(model, T=5)
    >>> handler = InteractiveConnectome(
    ...     fig, ax,
    ...     plot_objs['wedges'], plot_objs['lines'],
    ...     times=np.linspace(0, 5, signals[next(iter(signals))].size),
    ...     signals=signals,
    ... )
    """

    def __init__(self, fig, ax, wedges, lines, times=None, signals=None):
        self.fig = fig
        self.ax = ax
        self.wedges = wedges
        self.lines = lines
        self.times = times
        self.signals = signals

        self.original_wedge_props = []
        self.original_line_props = []
        self.selected_wedges = set()
        self.selected_lines = set()
        self.hover_text = None
        self.inset_ax = None

        # Legend labels carry a ``_connectome_key`` attribute set by
        # :func:`build_hierarchical_legend`; cache them so on_click can
        # bbox-test each one without re-walking the figure on every event.
        # Matplotlib's Legend nests each Text in multiple parent containers
        # so ``findobj`` returns the same artist more than once -- dedup
        # by identity to avoid testing the same bbox repeatedly.
        seen: dict = {}
        for a in fig.findobj(_mtext.Text):
            if hasattr(a, "_connectome_key") and id(a) not in seen:
                seen[id(a)] = a
        self._clickable_labels = list(seen.values())

        self._store_original_properties()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def _store_original_properties(self):
        for wedge_obj, _ in self.wedges:
            self.original_wedge_props.append({
                "linewidth": wedge_obj.get_linewidth(),
                "edgecolor": wedge_obj.get_edgecolor(),
                "zorder": wedge_obj.get_zorder(),
            })
        for line_obj, _ in self.lines:
            kind = "line2d" if hasattr(line_obj, "get_xydata") else "pathpatch"
            self.original_line_props.append({
                "linewidth": line_obj.get_linewidth(),
                "alpha": line_obj.get_alpha() if line_obj.get_alpha() is not None else 1.0,
                "type": kind,
            })

    # --------------------------------------------------------------------- #
    # Click handling
    # --------------------------------------------------------------------- #

    def on_click(self, event):
        # Legend labels live outside the data axes, so hit-test them first
        # using display (pixel) coords. event.x / event.y are set on every
        # button_press_event regardless of which axes the click lands in.
        key = self._find_clicked_legend_label(event)
        if key is not None:
            self._toggle_by_key(key)
            return

        if event.inaxes != self.ax:
            return

        clicked = self._find_clicked_wedge(event.xdata, event.ydata)
        if clicked is not None:
            self._toggle_wedge_selection(clicked)
            if clicked in self.selected_wedges:
                self._show_hover_text(event.xdata, event.ydata,
                                      display_label(self.wedges[clicked][1]))
                self._show_inset_plot(self.wedges[clicked][1])
            else:
                self._clear_hover_text()
                self._clear_inset_plot()
            return

        clicked = self._find_clicked_line(event.xdata, event.ydata)
        if clicked is not None:
            self._toggle_line_selection(clicked)
            if clicked in self.selected_lines:
                self._show_hover_text(event.xdata, event.ydata,
                                      self.lines[clicked][1])
            else:
                self._clear_hover_text()
            return

        self._clear_hover_text()
        self._clear_inset_plot()
        print(f"Clicked at ({event.xdata:.2f}, {event.ydata:.2f}) - no object hit")

    # --------------------------------------------------------------------- #
    # Legend label clicks
    # --------------------------------------------------------------------- #

    def _find_clicked_legend_label(self, event):
        """Return the ``_connectome_key`` of the legend label under the
        click, or ``None`` if the click isn't over a tagged label.

        Walks the cached list of tagged Text artists and asks each one
        whether the click hits its bounding box (``Artist.contains`` uses
        the rendered window extent, so this works whether the legend sits
        outside the axes -- the default -- or floats inside them).
        """
        if not self._clickable_labels or event.x is None or event.y is None:
            return None
        for text in self._clickable_labels:
            try:
                hit, _ = text.contains(event)
            except Exception:
                # If the legend hasn't been drawn yet (no renderer), .contains
                # may raise; skip silently rather than break click handling.
                continue
            if hit:
                return getattr(text, "_connectome_key", None)
        return None

    def _toggle_by_key(self, key):
        """Toggle every wedge in ``self.wedges`` whose label matches ``key``.

        A leaf rendered across several rings (padding) has multiple wedges
        with the same key -- they select / deselect together. Also pops up
        the inset signal plot when selecting (skipping the hover-text
        popup, since the legend label itself is the name).
        """
        matching = [i for i, (_, label) in enumerate(self.wedges) if label == key]
        if not matching:
            return
        any_selected = any(i in self.selected_wedges for i in matching)
        if any_selected:
            for i in matching:
                if i in self.selected_wedges:
                    self._toggle_wedge_selection(i)
            self._clear_inset_plot()
        else:
            for i in matching:
                self._toggle_wedge_selection(i)
            if self.times is not None and self.signals is not None:
                self._show_inset_plot(key)

    # --------------------------------------------------------------------- #
    # Hover text and inset plot
    # --------------------------------------------------------------------- #

    def _show_hover_text(self, x, y, text):
        self._clear_hover_text()
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="black", alpha=0.9)
        self.hover_text = self.ax.text(x, y, text, fontsize=10, bbox=bbox_props,
                                       zorder=200, verticalalignment="bottom",
                                       horizontalalignment="left")
        self.fig.canvas.draw()

    def _clear_hover_text(self):
        if self.hover_text is not None:
            self.hover_text.remove()
            self.hover_text = None
            self.fig.canvas.draw()

    def _show_inset_plot(self, label):
        if self.times is None or self.signals is None:
            print(f"No time series data available for {label}")
            return
        if label not in self.signals:
            print(f"No signal data found for label: {label}")
            return
        self._clear_inset_plot()

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        self.inset_ax = inset_axes(self.ax, width="30%", height="25%", loc="upper right")

        self.inset_ax.plot(self.times, self.signals[label], linewidth=2, color="blue")
        self.inset_ax.set_title(f"{display_label(label)}", fontsize=10, pad=5)
        self.inset_ax.set_xlabel("Time", fontsize=8)
        self.inset_ax.set_ylabel("Signal", fontsize=8)
        self.inset_ax.tick_params(labelsize=7)
        self.inset_ax.grid(True, alpha=0.3)
        self.inset_ax.patch.set_facecolor("lightgray")
        self.inset_ax.patch.set_alpha(0.9)
        self.inset_ax.set_zorder(150)
        self.fig.canvas.draw()

    def _clear_inset_plot(self):
        if self.inset_ax is not None:
            self.inset_ax.remove()
            self.inset_ax = None
            self.fig.canvas.draw()

    # --------------------------------------------------------------------- #
    # Hit-testing
    # --------------------------------------------------------------------- #

    def _find_clicked_wedge(self, x, y):
        if x is None or y is None:
            return None
        candidates = [
            (i, w) for i, (w, _) in enumerate(self.wedges)
            if self._is_point_in_wedge(x, y, w)
        ]
        return self._topmost(candidates)

    def _find_clicked_line(self, x, y, tolerance=0.1):
        if x is None or y is None:
            return None
        candidates = [
            (i, ln) for i, (ln, _) in enumerate(self.lines)
            if self._is_point_near_line(x, y, ln, tolerance)
        ]
        return self._topmost(candidates)

    @staticmethod
    def _topmost(candidates):
        """Pick the one with highest zorder; ties broken by latest draw order."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        best_idx = None
        best_z = float("-inf")
        latest = -1
        for i, obj in candidates:
            z = obj.get_zorder()
            if z > best_z or (z == best_z and i > latest):
                best_z = z
                best_idx = i
                latest = i
        return best_idx

    @staticmethod
    def _is_point_in_wedge(x, y, wedge):
        cx, cy = wedge.center
        r = wedge.r
        theta1, theta2 = wedge.theta1, wedge.theta2
        width = getattr(wedge, "width", r)
        dx, dy = x - cx, y - cy
        point_r = np.sqrt(dx * dx + dy * dy)
        point_theta = np.degrees(np.arctan2(dy, dx))
        if point_theta < 0: # todo: clean this up
            point_theta += 360
        if theta1 < 0:
            theta1 += 360
        if theta2 < 0:
            theta2 += 360
        if theta2 < theta1:
            theta2 += 360
            if point_theta < theta1:
                point_theta += 360
        inner_r = r - width
        return inner_r <= point_r <= r and theta1 <= point_theta <= theta2

    def _is_point_near_line(self, x, y, line, tolerance):
        try:
            if hasattr(line, "get_xydata"):
                pts = line.get_xydata()
                if len(pts) < 2:
                    return False
                seg_iter = zip(pts[:-1], pts[1:])
            elif hasattr(line, "get_path"):
                verts = line.get_path().vertices
                seg_iter = zip(verts[:-1], verts[1:])
            else:
                return False
            for (x1, y1), (x2, y2) in seg_iter:
                if self._point_to_line_distance(x, y, x1, y1, x2, y2) < tolerance:
                    return True
        except Exception as e:
            print(f"Error checking line click: {e}")
        return False

    @staticmethod
    def _point_to_line_distance(px, py, x1, y1, x2, y2):
        dx, dy = px - x1, py - y1
        lx, ly = x2 - x1, y2 - y1
        length_sq = lx * lx + ly * ly
        if length_sq == 0:
            return float(np.sqrt(dx * dx + dy * dy))
        t = max(0.0, min(1.0, (dx * lx + dy * ly) / length_sq))
        cx = x1 + t * lx
        cy = y1 + t * ly
        return float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))

    # --------------------------------------------------------------------- #
    # Selection
    # --------------------------------------------------------------------- #

    def _toggle_wedge_selection(self, idx):
        wedge_obj, label = self.wedges[idx]
        if idx in self.selected_wedges:
            self.selected_wedges.remove(idx)
            orig = self.original_wedge_props[idx]
            wedge_obj.set_linewidth(orig["linewidth"])
            wedge_obj.set_edgecolor(orig["edgecolor"])
            wedge_obj.set_zorder(orig["zorder"])
            print(f"Deselected wedge {idx}: {label}")
        else:
            self.selected_wedges.add(idx)
            wedge_obj.set_linewidth(3.0)
            wedge_obj.set_edgecolor("white")
            wedge_obj.set_zorder(100)
            print(f"Selected wedge {idx}: {label}")
        self.fig.canvas.draw()

    def _toggle_line_selection(self, idx):
        line_obj, label = self.lines[idx]
        orig = self.original_line_props[idx]
        if idx in self.selected_lines:
            self.selected_lines.remove(idx)
            line_obj.set_linewidth(orig["linewidth"])
            line_obj.set_alpha(orig["alpha"])
            print(f"Deselected line {idx}: {label}")
        else:
            self.selected_lines.add(idx)
            line_obj.set_linewidth(orig["linewidth"] * 5)
            line_obj.set_alpha(1.0)
            print(f"Selected line {idx}: {label}")
        self.fig.canvas.draw()

    def clear_selections(self):
        """Deselect every wedge and line, and dismiss any open popups.

        Each previously selected item is restored to the linewidth, edge
        color, and z-order it had when the handler was first constructed.
        """
        self._clear_hover_text()
        self._clear_inset_plot()
        for idx in list(self.selected_wedges):
            self._toggle_wedge_selection(idx)
        for idx in list(self.selected_lines):
            self._toggle_line_selection(idx)

    def get_selected_items(self):
        """Return the currently highlighted wedges and lines.

        Returns
        -------
        dict
            ``{'wedges': [(idx, label), ...], 'lines': [(idx, label), ...]}``.
            ``idx`` is the position in the ``wedges`` / ``lines`` list
            passed to ``__init__``; ``label`` is the same string carried in
            those tuples (i.e. :func:`get_key` form, suitable for indexing
            into ``signals``).
        """
        return {
            "wedges": [(i, self.wedges[i][1]) for i in self.selected_wedges],
            "lines":  [(i, self.lines[i][1])  for i in self.selected_lines],
        }
