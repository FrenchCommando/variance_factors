"""Per-pair-matching-window V-endpoint diagnostics with full X / full Y factor decomposition.

For each rolling-window row in `params_timeseries.feather`:

  1. GLS-infer the joint daily innovation (Z_S, Z_X, Z_Y) from the joint xi-observation
     Y_xi = (spot_return, log_xi_increments) under the matching window's params.
  2. Project onto the terminal-V endpoint observable:
        predicted_log_v[pair, i] = H_V(tau_i, P) @ [Z_X, Z_Y]
     where H_V[i, j] = (omega alpha / tau_i) w_j (1 - e^{-k_j tau_i}) / k_j.
  3. Decompose per endpoint into full X / full Y / cross-cov:
        x_full[i]  =  H_V[i, X] * z_x   (full X contribution)
        y_full[i]  =  H_V[i, Y] * z_y   (full Y contribution)
        predicted_v[i] = x_full[i] + y_full[i]    (identity since M_V has no spot column)
     Across pairs:
        Var(predicted_v[i]) = Var(x_full) + Var(y_full) + 2 Cov(x_full, y_full)
     reported separately so the decomposition is order-symmetric (rho_xy > 0 makes z_x
     and z_y co-move; cross-cov is positive and meaningful).

Plots (under `out/`):
  * v_reconstruction_SPX.png       -- V residual time series + variance decomposition
  * v_residual_acf_SPX.png         -- per-endpoint normalized residual ACF + Bartlett bands

Pairs without a matching window (first WINDOW_SIZE-1 dates) are skipped.

Usage:
    .venv\\Scripts\\python diagnostics.py
"""

import datetime as dt
import logging

import matplotlib.pyplot as plt
import numpy as np
from pyarrow import feather

from scripts.rolling_calibration import run_name_for
from utils.bergomi_likelihood import (
    gls_shock_estimate, innovation_covariance_with_spot, joint_observation_matrix,
)
from utils.bergomi_two_factor import BergomiTwoFactorParams, observation_matrix_v_constant_tenor
from utils.cache_paths import OUT_DIR, to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, TENOR_DAYS_BENCHMARK, ForwardVariancePanel, assemble_panel,
)
from utils.spot_data import FWD_PROXY_ROOT, daily_log_fwd_returns_for_panel_pairs, local_vol_per_panel_pair

logger = logging.getLogger(__name__)

ROOT = "SPX"
DATE_FROM = dt.date(2025, 1, 2)
DATE_TO = dt.date(2026, 3, 20)
WINDOW_SIZE = 60
RUN_NAME = run_name_for(window_size=WINDOW_SIZE)
N_ACF_LAGS = 30


def acf_one_sided(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Unbiased autocorrelation, lags 0..n_lags inclusive (lag 0 = 1)."""
    centered = series - series.mean()
    n = len(centered)
    if n_lags >= n:
        msg = f"n_lags ({n_lags}) must be < series length ({n})"
        raise ValueError(msg)
    autocov = np.array([float(np.sum(centered[:n - lag] * centered[lag:])) / (n - lag) for lag in range(n_lags + 1)])
    return autocov / autocov[0]


def reconstruct_and_decompose_v(
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    fit_table: dict,
) -> dict:
    """Per-pair V-endpoint reconstruction with full X / full Y factor decomposition."""
    fit_dates = [dt.date.fromisoformat(value) for value in fit_table["date"]]
    fit_index = {date: index for index, date in enumerate(fit_dates)}
    sigma_r_columns = sorted(
        [col for col in fit_table if col.startswith("sigma_r_strip_")], key=lambda col: int(col.rsplit("_", 1)[1]),
    )

    pair_dates: list[dt.date] = []
    observed_rows: list[np.ndarray] = []
    predicted_rows: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    x_full_rows: list[np.ndarray] = []
    y_full_rows: list[np.ndarray] = []
    sigma_v_rows: list[np.ndarray] = []

    for pair_index, end_index in enumerate(full_panel.pair_end_indices):
        pair_end_date = full_panel.dates[end_index]
        if pair_end_date not in fit_index:
            continue
        fit_row = fit_index[pair_end_date]
        params = BergomiTwoFactorParams(
            k_x=float(fit_table["k_x"][fit_row]), k_y=float(fit_table["k_y"][fit_row]),
            theta=float(fit_table["theta"][fit_row]), rho_xy=float(fit_table["rho_xy"][fit_row]),
            nu=float(fit_table["nu"][fit_row]),
            sigma_r_vector=np.array([float(fit_table[col][fit_row]) for col in sigma_r_columns]),
        )
        rho_sx = float(fit_table["rho_sx"][fit_row])
        rho_sy = float(fit_table["rho_sy"][fit_row])
        sigma_s = float(full_sigma_s_per_pair[pair_index])
        dt_years = float(full_panel.dt_years[pair_index])

        process_covariance_dt = innovation_covariance_with_spot(
            k_x=params.k_x, k_y=params.k_y, rho_xy=params.rho_xy, rho_sx=rho_sx, rho_sy=rho_sy, dt_years=dt_years,
        )
        observation_matrix_xi = joint_observation_matrix(
            strip_tenors_years=full_panel.strip_tenors_years, params=params, sigma_s=sigma_s,
        )
        observation_xi = np.concatenate([
            full_spot_returns[pair_index : pair_index + 1], full_panel.log_xi_increments[pair_index],
        ])
        shock = gls_shock_estimate(
            observation=observation_xi, observation_matrix=observation_matrix_xi,
            process_covariance_dt=process_covariance_dt, sigma_r_vector=params.sigma_r_vector,
        )
        z_x = float(shock[1])
        z_y = float(shock[2])

        h_v = observation_matrix_v_constant_tenor(tenor_grid_years=full_panel.tenor_grid_years, params=params)
        x_full = h_v[:, 0] * z_x
        y_full = h_v[:, 1] * z_y
        predicted_v = x_full + y_full
        observed_v = full_panel.log_v_increments[pair_index]
        residual_v = observed_v - predicted_v
        sigma_v_diag = np.sqrt(
            h_v[:, 0] ** 2 * process_covariance_dt[1, 1] + h_v[:, 1] ** 2 * process_covariance_dt[2, 2]
            + 2.0 * h_v[:, 0] * h_v[:, 1] * process_covariance_dt[1, 2],
        )

        pair_dates.append(pair_end_date)
        observed_rows.append(observed_v)
        predicted_rows.append(predicted_v)
        residual_rows.append(residual_v)
        x_full_rows.append(x_full)
        y_full_rows.append(y_full)
        sigma_v_rows.append(sigma_v_diag)

    return {
        "pair_dates": pair_dates,
        "observed_v": np.array(observed_rows),
        "predicted_v": np.array(predicted_rows),
        "residual_v": np.array(residual_rows),
        "x_full": np.array(x_full_rows),
        "y_full": np.array(y_full_rows),
        "sigma_v_diagonal": np.array(sigma_v_rows),
    }


def plot_v_reconstruction_with_decomposition(panel: ForwardVariancePanel, decomposition: dict) -> None:
    """V residual time series + variance decomposition (full X / full Y / cross-cov / residual)."""
    pair_dates = decomposition["pair_dates"]
    observed_v = decomposition["observed_v"]
    residual_v = decomposition["residual_v"]
    x_full = decomposition["x_full"]
    y_full = decomposition["y_full"]
    endpoint_days = (panel.tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR).round(0).astype(int)
    n_endpoints = len(endpoint_days)

    figure_name = f"15 v_reconstruction {ROOT}"
    figure, (axis_residuals, axis_decomp) = plt.subplots(
        num=figure_name, nrows=2, ncols=1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]},
    )
    colormap = plt.get_cmap("Dark2")

    for endpoint_index, days in enumerate(endpoint_days):
        axis_residuals.plot(
            pair_dates, residual_v[:, endpoint_index], "-", color=colormap(endpoint_index % 8), alpha=0.7,
            label=f"{days}d",
        )
    axis_residuals.axhline(0, color="black", linewidth=0.4)
    axis_residuals.set_ylabel("residual = observed - predicted (log_V increment)")
    axis_residuals.set_xlabel("Date")
    axis_residuals.set_title(
        f"{ROOT} per-endpoint V residual, per-pair-matching-window fit -- structure here is what the model missed",
    )
    axis_residuals.grid(visible=True, alpha=0.3)
    axis_residuals.legend(loc="upper left", ncol=4, fontsize=8)

    bar_positions = np.arange(n_endpoints)
    var_x = x_full.var(axis=0)
    var_y = y_full.var(axis=0)
    twice_cov_xy = 2.0 * (
        ((x_full - x_full.mean(axis=0, keepdims=True)) * (y_full - y_full.mean(axis=0, keepdims=True))).mean(axis=0)
    )
    var_residual = residual_v.var(axis=0)
    var_observed = observed_v.var(axis=0)
    cross_positive = np.maximum(twice_cov_xy, 0.0)
    cross_negative = np.minimum(twice_cov_xy, 0.0)

    axis_decomp.bar(bar_positions, var_x, color=colormap(1), label="Var(H_V[X] * z_x) full X")
    axis_decomp.bar(bar_positions, var_y, bottom=var_x, color=colormap(2), label="Var(H_V[Y] * z_y) full Y")
    axis_decomp.bar(
        bar_positions, cross_positive, bottom=var_x + var_y, color="gold", alpha=0.85,
        edgecolor="black", linewidth=0.4, label="2 * Cov(H_V[X]*z_x, H_V[Y]*z_y)",
    )
    axis_decomp.bar(
        bar_positions, cross_negative, color="gold", alpha=0.5,
        edgecolor="black", linewidth=0.4,
    )
    axis_decomp.bar(
        bar_positions, var_residual, bottom=var_x + var_y + cross_positive, color="grey", alpha=0.5,
        label="Var(residual) unexplained",
    )
    axis_decomp.scatter(
        bar_positions, var_observed, color="black", marker="x", zorder=5, label="Var(observed) total",
    )
    axis_decomp.axhline(0, color="black", linewidth=0.4)
    axis_decomp.set_xticks(bar_positions)
    axis_decomp.set_xticklabels([f"{value}d" for value in endpoint_days])
    axis_decomp.set_ylabel("Var(log_V increment)")
    axis_decomp.set_xlabel("endpoint tenor")
    axis_decomp.grid(visible=True, alpha=0.3, axis="y")
    axis_decomp.legend(loc="upper right", fontsize=9)
    axis_decomp.set_title("Per-endpoint variance decomposition: full X + full Y + 2*Cov(X,Y) + residual = total observed")

    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_v_residual_acf(panel: ForwardVariancePanel, normalized: np.ndarray, pair_dates: list[dt.date]) -> None:
    """One ACF panel per endpoint, with Bartlett +/-2/sqrtN bands."""
    n_steps, n_endpoints = normalized.shape
    bartlett = 2.0 / np.sqrt(n_steps)
    endpoint_days = panel.tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR

    figure_name = f"16 v_residual_acf {ROOT}"
    figure, axes = plt.subplots(
        num=figure_name, nrows=n_endpoints, ncols=1, figsize=(11, 1.7 * n_endpoints), sharex=True,
    )
    if n_endpoints == 1:
        axes = [axes]
    colormap = plt.get_cmap("Dark2")

    lag_indices = np.arange(N_ACF_LAGS + 1)
    for endpoint_index, days in enumerate(endpoint_days):
        autocorrelation = acf_one_sided(series=normalized[:, endpoint_index], n_lags=N_ACF_LAGS)
        axis = axes[endpoint_index]
        axis.bar(
            lag_indices, autocorrelation, color=colormap(endpoint_index % 8), alpha=0.7, width=0.8,
            label=f"{days:.0f}d endpoint",
        )
        axis.axhline(bartlett, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
        axis.axhline(-bartlett, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
        axis.axhline(0, color="black", linewidth=0.4)
        axis.set_ylabel("ACF")
        axis.set_ylim(-0.5, 1.05)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Lag (trading days)")
    axes[0].set_title(
        f"{ROOT} V-endpoint normalized residual ACF (n={n_steps} pairs, "
        f"{pair_dates[0]}..{pair_dates[-1]}) -- bars outside Bartlett bands suggest missed structure",
    )
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def main() -> None:
    """Recompute V residuals + factor decomposition + ACF for the rolling 60bd fit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feather_path = OUT_DIR / RUN_NAME / "params_timeseries.feather"
    if not feather_path.exists():
        msg = f"{feather_path} missing -- run rolling_calibration first"
        raise FileNotFoundError(msg)
    table = feather.read_table(str(feather_path))
    fit_table = {name: list(table.column(name).to_pylist()) for name in table.column_names}

    full_panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=TENOR_DAYS_BENCHMARK,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=7, min_expiries=5, max_extrapolation_fraction=0.10,
    )
    full_spot_returns = daily_log_fwd_returns_for_panel_pairs(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices, fixing_index=FIXING_INDEX_DEFAULT,
        root=FWD_PROXY_ROOT,
    )
    full_sigma_s_per_pair = local_vol_per_panel_pair(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices, fixing_index=FIXING_INDEX_DEFAULT,
        root=FWD_PROXY_ROOT,
    )
    logger.info(
        "Per-pair sigma_S from front varswap: median=%.4f, min=%.4f, max=%.4f",
        float(np.median(full_sigma_s_per_pair)), float(full_sigma_s_per_pair.min()),
        float(full_sigma_s_per_pair.max()),
    )

    decomposition = reconstruct_and_decompose_v(
        full_panel=full_panel, full_spot_returns=full_spot_returns,
        full_sigma_s_per_pair=full_sigma_s_per_pair, fit_table=fit_table,
    )
    residual_v = decomposition["residual_v"]
    sigma_v_diagonal = decomposition["sigma_v_diagonal"]
    normalized = residual_v / np.maximum(sigma_v_diagonal, 1.0e-30)
    pair_dates = decomposition["pair_dates"]

    endpoint_days = full_panel.tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR
    x_full = decomposition["x_full"]
    y_full = decomposition["y_full"]
    var_x = x_full.var(axis=0)
    var_y = y_full.var(axis=0)
    twice_cov_xy = 2.0 * (
        ((x_full - x_full.mean(axis=0, keepdims=True)) * (y_full - y_full.mean(axis=0, keepdims=True))).mean(axis=0)
    )
    var_residual = residual_v.var(axis=0)
    var_observed = decomposition["observed_v"].var(axis=0)
    logger.info("Per-endpoint V variance decomposition (using each pair's matching rolling window):")
    logger.info("  %-8s %-12s %-12s %-12s %-12s %-12s %-10s", "tenor", "var_X", "var_Y", "2*cov_XY", "var_resid", "var_obs", "R^2")
    for endpoint_index, days in enumerate(endpoint_days):
        var_predicted = var_x[endpoint_index] + var_y[endpoint_index] + twice_cov_xy[endpoint_index]
        r_squared = 1.0 - var_residual[endpoint_index] / max(var_observed[endpoint_index], 1.0e-30)
        logger.info(
            "  %5.0fd  %11.6f  %11.6f  %+11.6f  %11.6f  %11.6f  %9.5f", days,
            var_x[endpoint_index], var_y[endpoint_index], twice_cov_xy[endpoint_index],
            var_residual[endpoint_index], var_observed[endpoint_index], r_squared,
        )
        if var_predicted + var_residual[endpoint_index] > var_observed[endpoint_index] * 1.05:
            logger.warning(
                "  %5.0fd: explained + residual (%.5f) exceeds total observed (%.5f) by >5%% -- "
                "model and residual not orthogonal in this sample.",
                days, var_predicted + var_residual[endpoint_index], var_observed[endpoint_index],
            )

    plot_v_reconstruction_with_decomposition(panel=full_panel, decomposition=decomposition)
    plot_v_residual_acf(panel=full_panel, normalized=normalized, pair_dates=pair_dates)
    logger.info("Wrote V diagnostic plots for window=%d.", WINDOW_SIZE)
    plt.show()


if __name__ == "__main__":
    main()
