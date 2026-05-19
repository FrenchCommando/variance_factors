"""Vol-of-terminal-V diagnostics: apples-to-apples on the variance-swap-rate observable.

Terminal V = (1/tau) integral_0^tau xi_t^{t+u} du is the rough-vol-literature observable
(variance-swap rate, dimensionless V = 2 LogSwap / tau) that downstream analyses manipulate.

Under Bergomi, the daily change in log V at constant tenor tau (leading order in dt):

    d log V_t^{t+tau}  ~  (omega/tau) alpha [(1-theta) (1 - e^{-k_x tau})/k_x dW_X
                                              + theta   (1 - e^{-k_y tau})/k_y dW_Y]

Two scalar diagnostics per rolling window:

  1. Single-tenor magnitude: empirical vs model-implied annualized std at the endpoint
     closest to TARGET_TENOR_DAYS (= 60 BD; closest endpoint is 63 BD exactly).
     Output: v_volvol_*bd_SPX.png.

  2. Term-structure decay alpha: alpha = -OLS slope of log(std) vs log(tenor) over all 7
     endpoints per window.  Positive for decay; rough-vol benchmark (H ~ 0.1) is 0.4.
     Output: v_alpha_SPX.png.

The model side is signal-only (no sigma_R term: sigma_R was fit on log_xi, not log V).

Usage:
    .venv\\Scripts\\python empirical_nu_diagnostic.py
"""

import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: TC002
from pyarrow import feather

from scripts.rolling_calibration import run_name_for
from utils.bergomi_likelihood import innovation_covariance_with_spot
from utils.cache_paths import OUT_DIR, to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, TENOR_DAYS_BENCHMARK, ForwardVariancePanel, assemble_panel,
    slice_panel,
)

logger = logging.getLogger(__name__)

ROOT = "SPX"
WINDOW_SIZE = 60
RUN_NAME = run_name_for(window_size=WINDOW_SIZE)
TARGET_TENOR_DAYS = 60.0
ROUGH_VOL_ALPHA_BENCHMARK = 0.4


def feather_path() -> Path:
    """Path to the saved rolling fit's params_timeseries.feather."""
    return OUT_DIR / RUN_NAME / "params_timeseries.feather"


def panel_date_range_from_feather() -> tuple[dt.date, dt.date]:
    """Read the feather and derive (date_from, date_to) covering all rolling windows."""
    path = feather_path()
    if not path.exists():
        msg = f"{path} missing -- run rolling_calibration first"
        raise FileNotFoundError(msg)
    table = feather.read_table(path, columns=["date", "window_start_date"]).to_pandas()
    starts = [dt.date.fromisoformat(value) for value in table["window_start_date"].tolist()]
    ends = [dt.date.fromisoformat(value) for value in table["date"].tolist()]
    return min(starts), max(ends)


def closest_tenor_index(tenor_grid_years: np.ndarray, target_tenor_years: float) -> int:
    """Index of the endpoint tenor closest to target_tenor_years."""
    return int(np.argmin(np.abs(tenor_grid_years - target_tenor_years)))


def model_implied_v_signal_std(fitted_row: pd.Series, tenor_years: float, dt_years: float) -> float:
    """Annualized model-implied signal std of d log V_t^{t+tau} at constant tenor tau.

    H_V[j] = (omega alpha / tau) w_j (1 - exp(-k_j tau)) / k_j,  w = (1-theta, theta).
    """
    nu = float(fitted_row["nu"])
    k_x = float(fitted_row["k_x"])
    k_y = float(fitted_row["k_y"])
    theta = float(fitted_row["theta"])
    rho_xy = float(fitted_row["rho_xy"])
    rho_sx = float(fitted_row["rho_sx"])
    rho_sy = float(fitted_row["rho_sy"])

    omega = 2.0 * nu
    one_minus_theta = 1.0 - theta
    alpha = 1.0 / np.sqrt(
        one_minus_theta * one_minus_theta + theta * theta + 2.0 * one_minus_theta * theta * rho_xy,
    )
    factor = omega * alpha / tenor_years
    h_x = factor * one_minus_theta * (1.0 - np.exp(-k_x * tenor_years)) / k_x
    h_y = factor * theta * (1.0 - np.exp(-k_y * tenor_years)) / k_y

    q_full = innovation_covariance_with_spot(
        k_x=k_x, k_y=k_y, rho_xy=rho_xy, rho_sx=rho_sx, rho_sy=rho_sy, dt_years=dt_years,
    )
    q_xy_block = q_full[1:, 1:]
    h_row = np.array([h_x, h_y])
    signal_var_one_step = float(h_row @ q_xy_block @ h_row)
    return float(np.sqrt(signal_var_one_step / dt_years))


def empirical_v_std_at_tenor(window_panel: ForwardVariancePanel, tenor_index: int) -> float:
    """Annualized std of log_v_increments at tenor_index."""
    residuals_at_tenor = window_panel.log_v_increments[:, tenor_index]
    median_dt_years = float(np.median(window_panel.dt_years))
    return float(np.std(residuals_at_tenor) / np.sqrt(median_dt_years))


def per_window_v_stds(
    full_panel: ForwardVariancePanel, fitted_table: pd.DataFrame,
) -> tuple[list[dt.date], np.ndarray, np.ndarray]:
    """Return (end_dates, empirical, implied_signal), each shape (n_windows, n_tenors)."""
    n_dates = len(full_panel.dates)
    n_tenors = len(full_panel.tenor_grid_years)
    n_windows = n_dates - WINDOW_SIZE + 1
    if n_windows != len(fitted_table):
        msg = f"Window count mismatch: panel implies {n_windows}, feather has {len(fitted_table)}"
        raise ValueError(msg)
    end_dates: list[dt.date] = []
    empirical = np.empty((n_windows, n_tenors))
    implied = np.empty((n_windows, n_tenors))
    for window_index in range(n_windows):
        end_index = window_index + WINDOW_SIZE
        window_panel = slice_panel(full_panel=full_panel, start_index=window_index, end_index=end_index)
        median_dt_years = float(np.median(window_panel.dt_years))
        fitted_row = fitted_table.iloc[window_index]
        for tenor_index in range(n_tenors):
            tenor_years = float(full_panel.tenor_grid_years[tenor_index])
            empirical[window_index, tenor_index] = empirical_v_std_at_tenor(
                window_panel=window_panel, tenor_index=tenor_index,
            )
            implied[window_index, tenor_index] = model_implied_v_signal_std(
                fitted_row=fitted_row, tenor_years=tenor_years, dt_years=median_dt_years,
            )
        end_dates.append(window_panel.dates[-1])
    feather_dates = [dt.date.fromisoformat(value) for value in fitted_table["date"].tolist()]
    if feather_dates != end_dates:
        msg = "Feather end-date column does not align with positional rolling windows"
        raise ValueError(msg)
    return end_dates, empirical, implied


def alpha_per_window(stds: np.ndarray, log_tenors_years: np.ndarray) -> np.ndarray:
    """alpha = -OLS slope of log(std) vs log(tenor) across tenors, per window.  Positive = decay."""
    log_stds = np.log(stds.T)  # (n_tenors, n_windows); polyfit fits column-wise
    slopes_and_intercepts = np.polyfit(log_tenors_years, log_stds, deg=1)
    slopes = slopes_and_intercepts[0]
    return -slopes


def plot_empirical_vs_implied_at_tenor(
    dates: list[dt.date], empirical: np.ndarray, implied: np.ndarray, tenor_days: float,
) -> None:
    """Time series + scatter of empirical vs model-implied signal std at the fixed tenor."""
    figure_name = f"17 v_volvol {tenor_days:.0f}bd {ROOT}"
    figure, (axis_top, axis_bottom) = plt.subplots(num=figure_name, nrows=2, ncols=1, figsize=(11, 8))
    colormap = plt.get_cmap("Dark2")

    axis_top.plot(dates, empirical, "-", color=colormap(0), label="empirical d log V std", alpha=0.85)
    axis_top.plot(dates, implied, "-", color=colormap(1), label="model-implied (signal only)", alpha=0.85)
    axis_top.set_ylabel(f"annualized std of d log V at {tenor_days:.0f} BD")
    axis_top.set_xlabel("Window end date")
    axis_top.set_title(
        f"{ROOT} {WINDOW_SIZE}bd rolling: empirical vs model-implied vol-of-terminal-V at {tenor_days:.0f} BD",
    )
    axis_top.legend(loc="upper left")
    axis_top.grid(visible=True, alpha=0.3)

    upper_bound = float(max(empirical.max(), implied.max()) * 1.05)
    axis_bottom.scatter(empirical, implied, s=12, color=colormap(2), alpha=0.7)
    axis_bottom.plot([0, upper_bound], [0, upper_bound], "k--", alpha=0.4, label="y = x")
    axis_bottom.set_xlabel("empirical")
    axis_bottom.set_ylabel("model-implied (signal)")
    axis_bottom.set_title("scatter (apples to apples; signal-only on the model side)")
    axis_bottom.legend(loc="upper left")
    axis_bottom.grid(visible=True, alpha=0.3)
    axis_bottom.set_xlim(0, upper_bound)
    axis_bottom.set_ylim(0, upper_bound)

    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_alpha_empirical_vs_implied(
    dates: list[dt.date], alpha_empirical: np.ndarray, alpha_implied: np.ndarray,
) -> None:
    """Time series + scatter of empirical vs model-implied alpha (term-structure decay exponent)."""
    figure_name = f"18 v_alpha {ROOT}"
    figure, (axis_top, axis_bottom) = plt.subplots(num=figure_name, nrows=2, ncols=1, figsize=(11, 8))
    colormap = plt.get_cmap("Dark2")

    axis_top.plot(dates, alpha_empirical, "-", color=colormap(0), label="empirical", alpha=0.85)
    axis_top.plot(dates, alpha_implied, "-", color=colormap(1), label="model-implied", alpha=0.85)
    axis_top.axhline(
        ROUGH_VOL_ALPHA_BENCHMARK, color="grey", linestyle=":", alpha=0.6,
        label=f"rough-vol benchmark alpha={ROUGH_VOL_ALPHA_BENCHMARK} (H~0.1)",
    )
    axis_top.set_ylabel("alpha = -d log std / d log tenor")
    axis_top.set_xlabel("Window end date")
    axis_top.set_title(
        f"{ROOT} {WINDOW_SIZE}bd rolling: terminal-V term-structure decay exponent (OLS over 7 endpoints)",
    )
    axis_top.legend(loc="upper left")
    axis_top.grid(visible=True, alpha=0.3)

    lower = float(min(alpha_empirical.min(), alpha_implied.min()))
    upper = float(max(alpha_empirical.max(), alpha_implied.max()))
    pad = 0.05 * (upper - lower)
    axis_bottom.scatter(alpha_empirical, alpha_implied, s=12, color=colormap(2), alpha=0.7)
    axis_bottom.plot([lower - pad, upper + pad], [lower - pad, upper + pad], "k--", alpha=0.4, label="y = x")
    axis_bottom.set_xlabel("empirical alpha")
    axis_bottom.set_ylabel("model-implied alpha")
    axis_bottom.set_title("scatter")
    axis_bottom.legend(loc="upper left")
    axis_bottom.grid(visible=True, alpha=0.3)
    axis_bottom.set_xlim(lower - pad, upper + pad)
    axis_bottom.set_ylim(lower - pad, upper + pad)

    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def print_summary_stats(empirical: np.ndarray, implied: np.ndarray) -> None:
    """Median, mean, std, quantiles, signed difference summary."""
    for label, values in [("empirical", empirical), ("implied", implied)]:
        logger.info(
            "  %-9s  median=%+.4f  mean=%+.4f  std=%.4f  p10=%+.4f  p90=%+.4f  min=%+.4f  max=%+.4f",
            label, float(np.median(values)), values.mean(), values.std(),
            float(np.percentile(values, 10)), float(np.percentile(values, 90)), values.min(), values.max(),
        )
    diff = implied - empirical
    logger.info(
        "  implied - empirical: median=%+.4f  std=%.4f  max_abs=%.4f",
        float(np.median(diff)), diff.std(), float(np.abs(diff).max()),
    )


def main() -> None:
    """Compute and plot single-tenor vol-of-V and term-structure alpha empirical-vs-implied."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    date_from, date_to = panel_date_range_from_feather()
    logger.info("Assembling %s panel %s -> %s", ROOT, date_from, date_to)
    full_panel = assemble_panel(
        root=ROOT, date_from=date_from, date_to=date_to, tenor_days=TENOR_DAYS_BENCHMARK,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=7, min_expiries=5, max_extrapolation_fraction=0.10,
    )
    fitted_table = feather.read_table(feather_path()).to_pandas()
    dates, empirical_stds, implied_stds = per_window_v_stds(
        full_panel=full_panel, fitted_table=fitted_table,
    )

    target_tenor_years = TARGET_TENOR_DAYS / N_BUSINESS_DAYS_PER_YEAR
    tenor_index = closest_tenor_index(
        tenor_grid_years=full_panel.tenor_grid_years, target_tenor_years=target_tenor_years,
    )
    actual_tenor_days = float(full_panel.tenor_grid_years[tenor_index] * N_BUSINESS_DAYS_PER_YEAR)
    logger.info(
        "Single-tenor diagnostic: endpoint %d at %.0f BD (target %.0f BD), %d windows",
        tenor_index, actual_tenor_days, TARGET_TENOR_DAYS, len(dates),
    )
    print_summary_stats(empirical=empirical_stds[:, tenor_index], implied=implied_stds[:, tenor_index])
    plot_empirical_vs_implied_at_tenor(
        dates=dates, empirical=empirical_stds[:, tenor_index], implied=implied_stds[:, tenor_index],
        tenor_days=actual_tenor_days,
    )

    log_tenors_years = np.log(full_panel.tenor_grid_years)
    alpha_empirical = alpha_per_window(stds=empirical_stds, log_tenors_years=log_tenors_years)
    alpha_implied = alpha_per_window(stds=implied_stds, log_tenors_years=log_tenors_years)
    logger.info(
        "Term-structure decay alpha (OLS slope sign-flipped): %d windows, rough-vol benchmark = %.2f",
        len(dates), ROUGH_VOL_ALPHA_BENCHMARK,
    )
    print_summary_stats(empirical=alpha_empirical, implied=alpha_implied)
    plot_alpha_empirical_vs_implied(
        dates=dates, alpha_empirical=alpha_empirical, alpha_implied=alpha_implied,
    )

    plt.show()


if __name__ == "__main__":
    main()
