"""A single house style so every figure reads as one consistent system.

Deliberately 2D. For calibration curves, densities, and convergence traces, clean
2D with a shared palette communicates far better than 3D — no occlusion, honest
axes, and a reviewer can read values directly off the plot.
"""
from __future__ import annotations

# One color per model, reused across every figure so a model is always the same
# hue. Muted, colorblind-friendly qualitative set.
PALETTE = {
    "Gaussian":      "#4C72B0",  # blue
    "EWMA-Gaussian": "#DD8452",  # orange
    "Laplace":       "#55A868",  # green
    "Student-t":     "#C44E52",  # red
    "Empirical":     "#8172B3",  # purple
}

# Distribution-family colors for density overlays (mirror the model hues).
DIST_COLORS = {
    "Normal":  "#4C72B0",
    "Laplace": "#55A868",
    "Student": "#C44E52",
}

_GRID = "#e3e3e3"
_INK = "#333333"


def use_house_style() -> None:
    """Apply the shared matplotlib rcParams. Idempotent — safe to call per figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.dpi": 130,
        "figure.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": _INK,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": _GRID,
        "grid.linewidth": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })


def dist_color(label: str) -> str:
    """Pick a palette color for a fitted-distribution label like 'Student-t(nu=2.1)'."""
    for key, color in DIST_COLORS.items():
        if label.lower().startswith(key.lower()):
            return color
    return _INK
