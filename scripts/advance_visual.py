"""Visualize the V term-structure advance correction.

Three panels on the terminal-V observable:

  1. For one chosen date t: log_V[t], log_V_advanced[t] (today advanced by dt under the
     Bergomi martingale), and log_V[t+1] (tomorrow observed) overlaid at endpoint tenors.
  2. Per-endpoint bars: raw return log_V[t+1] - log_V[t]  vs  advanced residual
     log_V[t+1] - log_V_advanced[t].  The gap is the per-tenor advance correction.
  3. Panel-wide boxplot of log_V_advanced - log_V[:-1] per endpoint -- distribution of the
     daily advance correction across all dates.

Usage:
    .venv\\Scripts\\python advance_visual.py
"""

import datetime as dt
import logging

import matplotlib.pyplot as plt
import numpy as np

from utils.cache_paths import to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, TENOR_DAYS_BENCHMARK, ForwardVariancePanel, assemble_panel,
)

logger = logging.getLogger(__name__)

ROOT = "SPX"
DATE_FROM = dt.date(2025, 1, 2)
DATE_TO = dt.date(2026, 3, 20)

# Typical 1-business-day-gap pair so dt = 1/252 ~ 0.00397 yr.
SAMPLE_DATE = dt.date(2025, 6, 16)


def find_target_pair_index(panel: ForwardVariancePanel, sample_date: dt.date) -> int:
    """Return the panel-pair index whose end-date is closest to sample_date."""
    pair_end_dates = [panel.dates[int(end_index)] for end_index in panel.pair_end_indices]
    distances = [abs((pair_end_date - sample_date).days) for pair_end_date in pair_end_dates]
    return int(np.argmin(np.asarray(distances)))


def panel_log_v_advanced(panel: ForwardVariancePanel) -> np.ndarray:
    """Recover log_V_advanced per pair: log_v_endpoints[pair_end] - log_v_increments."""
    return panel.log_v_endpoints[panel.pair_end_indices] - panel.log_v_increments


def plot_advance_visual_v(panel: ForwardVariancePanel, sample_pair_index: int) -> None:
    """Three-panel figure showing the V advanced-curve construction and its size in the panel."""
    end_index = int(panel.pair_end_indices[sample_pair_index])
    start_index = end_index - 1
    today = panel.dates[start_index]
    tomorrow = panel.dates[end_index]
    dt_years = float(panel.dt_years[sample_pair_index])
    logger.info(
        "Sample pair: %s -> %s, dt_years=%.5f (%.2f BD)", today, tomorrow, dt_years, dt_years * 252,
    )

    endpoint_days = (panel.tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR).round(0).astype(int)
    log_v_today = panel.log_v_endpoints[start_index]
    log_v_tomorrow = panel.log_v_endpoints[end_index]
    log_v_increment_for_pair = panel.log_v_increments[sample_pair_index]
    log_v_advanced_today = log_v_tomorrow - log_v_increment_for_pair

    raw_return = log_v_tomorrow - log_v_today
    advanced_residual = log_v_increment_for_pair

    figure_name = f"04 advance_visual {ROOT}"
    figure, axes = plt.subplots(num=figure_name, nrows=3, ncols=1, figsize=(11, 11))
    colormap = plt.get_cmap("Dark2")

    axes[0].plot(endpoint_days, log_v_today, "-o", color=colormap(0), label=f"log_V[t={today}] (today)")
    axes[0].plot(
        endpoint_days, log_v_advanced_today, "x--", color=colormap(2),
        label=f"log_V_advanced[t]  (today advanced by dt={dt_years * 252:.1f} BD)",
    )
    axes[0].plot(
        endpoint_days, log_v_tomorrow, "-^", color=colormap(1),
        label=f"log_V[t+1={tomorrow}] (tomorrow observed)",
    )
    axes[0].set_xlabel("endpoint tenor (BD)")
    axes[0].set_ylabel("log V (dimensionless)")
    axes[0].set_title(f"Three curves at endpoint tenors -- {today} -> {tomorrow}")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend(fontsize=9)

    n_tenors = len(endpoint_days)
    bar_positions = np.arange(n_tenors)
    width = 0.4
    axes[1].bar(
        bar_positions - width / 2, raw_return, width=width, color=colormap(3),
        label="raw: log_V[t+1] - log_V[t]",
    )
    axes[1].bar(
        bar_positions + width / 2, advanced_residual, width=width, color=colormap(4),
        label="advanced: log_V[t+1] - log_V_advanced[t]",
    )
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(bar_positions)
    axes[1].set_xticklabels([f"{value}d" for value in endpoint_days])
    axes[1].set_xlabel("endpoint tenor")
    axes[1].set_ylabel("residual at this tenor")
    axes[1].set_title("Per-tenor residuals on the sample date: gap between the two = the advance correction")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plot_panel_correction_distribution(axis=axes[2], panel=panel, endpoint_days=endpoint_days)

    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_panel_correction_distribution(
    axis: plt.Axes, panel: ForwardVariancePanel, endpoint_days: np.ndarray,
) -> None:
    """Boxplot of log_V_advanced - log_V[t] per endpoint across all pairs in the panel."""
    log_v_advanced = panel_log_v_advanced(panel=panel)
    pair_start_indices = panel.pair_end_indices - 1
    log_v_start = panel.log_v_endpoints[pair_start_indices]
    panel_corrections = log_v_advanced - log_v_start

    n_tenors = len(endpoint_days)
    bar_positions = np.arange(n_tenors)
    box_data = [panel_corrections[:, tenor_index] for tenor_index in range(n_tenors)]
    axis.boxplot(
        box_data, positions=bar_positions, widths=0.6, showfliers=True,
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "grey", "alpha": 0.5},
    )
    axis.axhline(0, color="black", linewidth=0.5)
    axis.set_xticks(bar_positions)
    axis.set_xticklabels([f"{value}d" for value in endpoint_days])
    axis.set_xlabel("endpoint tenor")
    axis.set_ylabel("log_V_advanced - log_V[t]   (advance-correction magnitude)")
    axis.set_title(f"Distribution of the daily advance correction across the {len(panel.dates)}-date panel")
    axis.grid(visible=True, alpha=0.3)


def main() -> None:
    """Build the panel and show the three-panel V advance diagnostic."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=TENOR_DAYS_BENCHMARK,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=7, min_expiries=5, max_extrapolation_fraction=0.10,
    )
    sample_pair_index = find_target_pair_index(panel=panel, sample_date=SAMPLE_DATE)
    plot_advance_visual_v(panel=panel, sample_pair_index=sample_pair_index)
    plt.show()


if __name__ == "__main__":
    main()
