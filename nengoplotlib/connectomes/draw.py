"""Low-level drawing primitives.

Pure functions on a matplotlib axis -- no knowledge of trees, edges, or
layouts. Lifted verbatim (with light tidying) from the legacy module so the
plot orchestrator can call them via stable signatures.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
from scipy.special import comb


def polar_to_cartesian(r, theta_deg):
    theta = np.deg2rad(theta_deg)
    return r * np.cos(theta), r * np.sin(theta)


def bezier_curve(points, n_points: int = 100):
    """Evaluate a Bezier curve from control ``points`` (``(N+1, 2)`` array)."""
    points = np.asarray(points, dtype=float)
    n = len(points) - 1
    t = np.linspace(0.0, 1.0, n_points)
    curve = np.zeros((n_points, 2))
    for i, p in enumerate(points):
        bernstein = comb(n, i) * (t**i) * ((1.0 - t) ** (n - i))
        curve += np.outer(bernstein, p)
    return curve[:, 0], curve[:, 1]


def draw_arc(ax, theta1, theta2, *, r=0.9, control_r=0.4,
             color="k", lw=1.0, alpha=0.5, arrow=True, arrow_size=0.005):
    """Quadratic-Bezier arc between two ring-edge points, optional arrowhead.

    Returns the ``PathPatch`` (the arrow head, if drawn, is added separately
    to the axis but not returned -- the caller usually only needs the arc).
    """
    x1, y1 = polar_to_cartesian(r, theta1)
    x2, y2 = polar_to_cartesian(r, theta2)
    cx, cy = polar_to_cartesian(control_r, (theta1 + theta2) / 2.0)

    path = Path([(x1, y1), (cx, cy), (x2, y2)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(patch)

    if arrow:
        dx, dy = x2 - cx, y2 - cy
        norm = (dx * dx + dy * dy) ** 0.5
        if norm:
            dx /= norm
            dy /= norm
        arrow_patch = FancyArrowPatch(
            posA=(x2 - dx * arrow_size, y2 - dy * arrow_size),
            posB=(x2, y2),
            arrowstyle="->",
            color=color, lw=lw, alpha=alpha,
            mutation_scale=10 * arrow_size * 100,
        )
        ax.add_patch(arrow_patch)
    return patch


def draw_path(ax, thetas, depths, *, r=0.9, min_r=0.4,
              color="k", lw=1.0, alpha=0.5, n_points=100):
    """Hierarchical-edge-bundled path through ``(theta, depth)`` knots.

    The deepest control point sits near ``r``; the shallowest sits at
    ``min_r * r``. A Bezier through these points produces the bundled look
    used by ``connection_type='bundled'``.
    """
    thetas = np.asarray(thetas, dtype=float)
    depths = np.asarray(depths, dtype=float)
    if thetas.shape != depths.shape:
        raise ValueError("thetas and depths must have the same length")

    depth_max = depths.max() if depths.size else 0.0
    min_r_abs = min_r * r
    if depth_max == 0:
        rs = np.full_like(depths, r)
    else:
        rs = min_r_abs + (r - min_r_abs) * (depths / depth_max)

    xs, ys = polar_to_cartesian(rs, thetas)
    x_curve, y_curve = bezier_curve(np.column_stack([xs, ys]), n_points=n_points)
    (line,) = ax.plot(x_curve, y_curve, color=color, lw=lw, alpha=alpha)
    return line
