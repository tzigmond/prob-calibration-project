"""Residual-distribution and calibration visuals.

Confirms the distributional assumption actually fits (histograms, Q-Q, tail
probabilities) and communicates the headline calibration result (reliability +
sharpness). Well-calibrated intervals are necessary but not sufficient evidence
for a distribution; the diagnostics close that gap. All plotting functions apply
the shared house style and save to a caller-supplied path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotstyle import PALETTE, dist_color, use_house_style


def _standardize(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    return (residuals - np.mean(residuals)) / np.std(residuals)


def plot_residual_hist(residuals: np.ndarray, fitted_dists: dict, save_path: str) -> None:
    """Histogram of standardized residuals with fitted densities overlaid.

    ``fitted_dists`` maps a label to a frozen scipy distribution (already scaled
    to standardized residuals). The x-range is clipped to a robust limit so a few
    extreme outliers don't squash the bulk of the distribution into a sliver.
    """
    use_house_style()
    import matplotlib.pyplot as plt

    z = _standardize(residuals)
    # Robust symmetric limit: cover ~99.7% of the mass, capped to keep it readable.
    lim = float(np.clip(np.quantile(np.abs(z), 0.997), 4.0, 8.0))
    z_clipped = z[np.abs(z) <= lim]
    n_hidden = len(z) - len(z_clipped)

    fig, ax = plt.subplots()
    ax.hist(z_clipped, bins=70, range=(-lim, lim), density=True,
            color="#c9c9c9", edgecolor="white", linewidth=0.4,
            label="standardized residuals")
    grid = np.linspace(-lim, lim, 500)
    for label, dist in fitted_dists.items():
        ax.plot(grid, dist.pdf(grid), lw=2.2, color=dist_color(label), label=label)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("standardized residual")
    ax.set_ylabel("density")
    ax.set_title("Residual distribution vs. fitted densities")
    if n_hidden:
        ax.text(0.99, 0.97, f"{n_hidden} outlier(s) beyond ±{lim:.0f}σ not shown",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#888")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def qq_plot(residuals: np.ndarray, dist, save_path: str, title: str = "Q-Q plot") -> None:
    """Q-Q plot of standardized residuals against an assumed distribution ``dist``
    (a frozen scipy distribution, e.g. ``stats.t(nu)``)."""
    use_house_style()
    import matplotlib.pyplot as plt

    z = np.sort(_standardize(residuals))
    n = len(z)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = dist.ppf(probs)
    finite = np.isfinite(theo)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theo[finite], z[finite], s=10, alpha=0.5, color="#4C72B0",
               edgecolors="none")
    lo = min(theo[finite].min(), z.min())
    hi = max(theo[finite].max(), z.max())
    ax.plot([lo, hi], [lo, hi], color="#C44E52", lw=1.6, label="y = x")
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("empirical quantile")
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def plot_calibration(table: pd.DataFrame, dataset_name: str, save_path: str) -> None:
    """The headline calibration figure: reliability (coverage vs. nominal, with
    binomial CIs and the perfect-calibration diagonal) alongside sharpness
    (average interval width). Consumes a ``coverage_table`` DataFrame.
    """
    use_house_style()
    import matplotlib.pyplot as plt

    level_tags = list(dict.fromkeys(c[0] for c in table.columns))   # e.g. 90%/95%/99%
    nominal = [int(t.strip("%")) / 100 for t in level_tags]
    models = list(table.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    # --- Reliability ---
    lo_lim = min(nominal) - 0.03
    ax1.plot([lo_lim, 1.0], [lo_lim, 1.0], ls="--", color="#999999", lw=1.3,
             label="perfect", zorder=1)
    for m in models:
        cov = np.array([table.loc[m, (t, "coverage")] for t in level_tags])
        ci_lo = np.array([table.loc[m, (t, "ci_lo")] for t in level_tags])
        ci_hi = np.array([table.loc[m, (t, "ci_hi")] for t in level_tags])
        yerr = np.vstack([cov - ci_lo, ci_hi - cov])
        ax1.errorbar(nominal, cov, yerr=yerr, marker="o", markersize=6, capsize=3,
                     lw=2.0, color=PALETTE.get(m), label=m, zorder=2)
    ax1.set_xticks(nominal)
    ax1.set_xticklabels(level_tags)
    ax1.set_ylim(lo_lim, 1.006)
    ax1.set_xlabel("nominal coverage")
    ax1.set_ylabel("empirical coverage")
    ax1.set_title(f"{dataset_name}: reliability")
    ax1.legend(title="model")

    # --- Sharpness (grouped width bars) ---
    x = np.arange(len(level_tags))
    bw = 0.15
    offset = -(len(models) - 1) / 2 * bw
    for i, m in enumerate(models):
        widths = [table.loc[m, (t, "width")] for t in level_tags]
        ax2.bar(x + offset + i * bw, widths, bw, color=PALETTE.get(m), label=m)
    ax2.set_xticks(x)
    ax2.set_xticklabels(level_tags)
    ax2.set_xlabel("nominal level")
    ax2.set_ylabel("average interval width")
    ax2.set_title(f"{dataset_name}: sharpness (narrower is better)")
    ax2.legend(title="model")

    fig.suptitle(f"Interval calibration - {dataset_name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def tail_probabilities(residuals: np.ndarray, fitted_dists: dict) -> pd.DataFrame:
    """Empirical vs. modeled P(|z| > 2), P(|z| > 3), P(|z| > 4).

    ``fitted_dists`` maps a label to a frozen scipy distribution over standardized
    residuals. Returns a small DataFrame: rows = thresholds, columns = empirical
    plus one per model.
    """
    z = _standardize(residuals)
    thresholds = [2, 3, 4]
    data: dict[str, list[float]] = {
        "empirical": [float(np.mean(np.abs(z) > k)) for k in thresholds]
    }
    for label, dist in fitted_dists.items():
        data[label] = [float(dist.sf(k) + dist.cdf(-k)) for k in thresholds]
    return pd.DataFrame(data, index=[f"P(|z|>{k})" for k in thresholds])
