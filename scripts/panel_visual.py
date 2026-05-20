"""Interactive visual diagnostics for the assembled data panel.

Three views:

1. Term structure of total annualized variance V vs business-days-to-expiration for a few
   sample dates, with the 1BD-front sigma_S overlaid as a diamond marker at DTE = 1.  The
   marker is the local-vol anchor that drives the joint-likelihood spot-row scaling.
2. Cumulative variance V * tau vs business-days-to-expiration -- local non-monotonicity at
   third-Friday SPXW neighbors shows up here (the third-Friday listing inherits CBOE's
   deep-OTM strike chain, which inflates its truncated log-swap integral relative to
   adjacent regular weeklies that don't list those wings).
3. Strip forward variance time series across the assembled panel, one line per strip tenor.

Usage:
    .venv\\Scripts\\python panel_visual.py
"""

import datetime as dt
import logging

import matplotlib.pyplot as plt
import numpy as np

from utils.cache_paths import MIN_RAW_DAYS, PANEL_TENOR_DAYS, ROOT, to_image_path
from utils.calendar_utils import plus_days
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, assemble_panel,
    build_varswap_index, load_term_structure_for_date, resample_term_structure,
)
from utils.intraday_time import intraday_time_to_expiry, is_am_settled
from utils.spot_data import FWD_PROXY_ROOT, read_log_swap_mid_at_fixing

logger = logging.getLogger(__name__)

DATE_FROM = dt.date(2025, 9, 1)
DATE_TO = dt.date(2025, 12, 31)
FIXING_INDEX = FIXING_INDEX_DEFAULT
MIN_EXPIRIES = 5
MAX_EXTRAPOLATION_FRACTION = 0.10

TENOR_DAYS = PANEL_TENOR_DAYS

SAMPLE_DATES = (
    dt.date(2025, 4, 4),
    dt.date(2025, 4, 7),
    dt.date(2025, 4, 8),
)


def fixing_label(fixing_index: int) -> str:
    """Format the fixing index as HH:MM:SS ET.  index 0 = 08:00:00, step = 1 second."""
    total_seconds = 8 * 3600 + fixing_index
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def front_sigma_s_for_date(date: dt.date, fixing_index: int) -> tuple[float, float]:
    """Return (raw_days, sigma_S) for the 1BD-front SPXW varswap at this observation date."""
    end_date = plus_days(date=date, n_days=1)
    raw_days = 1
    am_settled = is_am_settled(root=FWD_PROXY_ROOT)
    tau_years = intraday_time_to_expiry(raw_days=raw_days, timestamp_index=fixing_index, am_settled=am_settled)
    log_swap = read_log_swap_mid_at_fixing(
        root=FWD_PROXY_ROOT, expiration=end_date, observation_date=date, fixing_index=fixing_index,
    )
    sigma_s = float(np.sqrt(2.0 * log_swap / tau_years))
    return float(raw_days), sigma_s


def plot_term_structure_for_dates(  # noqa: PLR0913
    root: str, dates: tuple[dt.date, ...], fixing_index: int, min_raw_days: int, tenor_grid_years: np.ndarray,
    max_extrapolation_fraction: float,
) -> None:
    """Per-date listed vol vs DTE plus a PCHIP-resampled overlay on the constant-tenor grid."""
    expirations_index = build_varswap_index(root=root, date_from=min(dates), date_to=max(dates))
    colormap = plt.get_cmap("Dark2")
    time_label = fixing_label(fixing_index=fixing_index)

    listed_color = colormap(0)
    resampled_color = colormap(2)
    sigma_s_color = colormap(3)

    figure_name = f"01 term_structure {root} {time_label}"
    figure, axes = plt.subplots(
        num=figure_name, nrows=len(dates), ncols=1, figsize=(11, 3.2 * len(dates)), sharex=False,
    )
    if len(dates) == 1:
        axes = [axes]

    for axis_index, date in enumerate(dates):
        axis = axes[axis_index]
        expirations = expirations_index.get(date)
        if not expirations:
            axis.set_title(f"{root} {date} -- no expirations available")
            continue
        points = load_term_structure_for_date(
            root=root, date=date, expirations=expirations, fixing_index=fixing_index, min_raw_days=min_raw_days,
        )
        if not points:
            axis.set_title(f"{root} {date} -- no valid points")
            continue

        raw_days_array = np.array([point.raw_days for point in points])
        vol_array = 100.0 * np.sqrt(np.array([point.total_variance for point in points]))

        axis.plot(raw_days_array, vol_array, "o", color=listed_color, label="Listed expirations", markersize=5)

        resampled = resample_term_structure(
            points=points, tenor_grid_years=tenor_grid_years, max_extrapolation_fraction=max_extrapolation_fraction,
        )
        if resampled is not None:
            axis.plot(
                tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR, 100.0 * np.sqrt(np.maximum(resampled, 0.0)), "x--",
                color=resampled_color, label="PCHIP resampled",
            )

        sigma_s_days, sigma_s_value = front_sigma_s_for_date(date=date, fixing_index=fixing_index)
        axis.plot(
            sigma_s_days, 100.0 * sigma_s_value, "D", color=sigma_s_color, markersize=8,
            label=f"sigma_S (1BD front) = {sigma_s_value * 100:.2f}%",
        )

        axis.set_xlabel("Business days to expiration")
        axis.set_ylabel("Annualized vol sqrt(V) (%)")
        axis.set_title(f"{root} {date} {time_label} ET -- term structure of vol")
        axis.grid(visible=True, alpha=0.3)
        axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_cumulative_variance_for_dates(
    root: str, dates: tuple[dt.date, ...], fixing_index: int, min_raw_days: int,
) -> None:
    """Total Black-Scholes vol sqrt(V*tau) vs DTE for sample dates."""
    expirations_index = build_varswap_index(root=root, date_from=min(dates), date_to=max(dates))
    colormap = plt.get_cmap("Dark2")
    time_label = fixing_label(fixing_index=fixing_index)

    figure_name = f"02 cumulative_variance {root} {time_label}"
    figure, axis = plt.subplots(num=figure_name, figsize=(11, 6))
    for date_index, date in enumerate(dates):
        expirations = expirations_index.get(date)
        if not expirations:
            continue
        points = load_term_structure_for_date(
            root=root, date=date, expirations=expirations, fixing_index=fixing_index, min_raw_days=min_raw_days,
        )
        if not points:
            continue
        raw_days_array = np.array([point.raw_days for point in points])
        total_vol_array = 100.0 * np.sqrt(np.array([point.total_variance * point.time_to_expiry for point in points]))
        line_color = colormap(date_index)
        axis.plot(raw_days_array, total_vol_array, "-o", color=line_color, alpha=0.7, label=f"{date}", markersize=4)

    axis.set_xlabel("Business days to expiration")
    axis.set_ylabel("Total vol sqrt(V * tau) (%)")
    axis.set_title(f"{root} total vol (Black-Scholes form) vs DTE ({time_label} ET)")
    axis.grid(visible=True, alpha=0.3)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_strip_panel_time_series(
    root: str, dates: tuple[dt.date, ...], strip_tenors_years: np.ndarray, log_xi: np.ndarray, fixing_index: int,
) -> None:
    """Forward variance per strip tenor, time series across the assembled panel."""
    colormap = plt.get_cmap("Dark2")
    time_label = fixing_label(fixing_index=fixing_index)
    strip_tenors_days = strip_tenors_years * N_BUSINESS_DAYS_PER_YEAR

    figure_name = f"03 strip_panel {root} {time_label}"
    figure, (axis_log, axis_vol) = plt.subplots(num=figure_name, nrows=2, ncols=1, figsize=(11, 7), sharex=True)
    forward_variance = np.exp(log_xi)
    forward_vol_pct = 100.0 * np.sqrt(np.maximum(forward_variance, 0.0))

    for tenor_index, tenor_days in enumerate(strip_tenors_days):
        line_color = colormap(tenor_index % 8)
        axis_log.plot(dates, log_xi[:, tenor_index], "-o", color=line_color, label=f"{tenor_days:.1f}d", markersize=3)
        axis_vol.plot(
            dates, forward_vol_pct[:, tenor_index], "-o", color=line_color, label=f"{tenor_days:.1f}d", markersize=3,
        )

    axis_log.set_ylabel("log forward variance ln(xi)")
    axis_log.set_title(f"{root} strip forward variance panel ({time_label} ET)")
    axis_log.grid(visible=True, alpha=0.3)
    axis_log.legend(loc="upper left", ncol=4, fontsize=8)

    axis_vol.set_ylabel("Forward vol sqrt(xi) (%)")
    axis_vol.set_xlabel("Date")
    axis_vol.grid(visible=True, alpha=0.3)
    axis_vol.legend(loc="upper left", ncol=4, fontsize=8)

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def main() -> None:
    """Build the panel, then render the three diagnostic views interactively."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=TENOR_DAYS, fixing_index=FIXING_INDEX,
        min_raw_days=MIN_RAW_DAYS, min_expiries=MIN_EXPIRIES, max_extrapolation_fraction=MAX_EXTRAPOLATION_FRACTION,
    )
    logger.info("Panel: n_dates=%d, n_strip_tenors=%d", len(panel.dates), len(panel.strip_tenors_years))

    tenor_grid_years = np.array(TENOR_DAYS, dtype=np.float64) / N_BUSINESS_DAYS_PER_YEAR
    plot_term_structure_for_dates(
        root=ROOT, dates=SAMPLE_DATES, fixing_index=FIXING_INDEX, min_raw_days=MIN_RAW_DAYS,
        tenor_grid_years=tenor_grid_years, max_extrapolation_fraction=MAX_EXTRAPOLATION_FRACTION,
    )
    plot_cumulative_variance_for_dates(
        root=ROOT, dates=SAMPLE_DATES, fixing_index=FIXING_INDEX, min_raw_days=MIN_RAW_DAYS,
    )
    plot_strip_panel_time_series(
        root=ROOT, dates=panel.dates, strip_tenors_years=panel.strip_tenors_years, log_xi=panel.log_xi,
        fixing_index=FIXING_INDEX,
    )
    plt.show()


if __name__ == "__main__":
    main()
