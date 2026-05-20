"""In-sample vs out-of-sample predicted log_V-increment residuals at three calibration dates.

For chosen calibration date D (= a row in `params_timeseries.feather`):

    P_D                 = parameters from the rolling window ending at D
    For every panel pair p:
        (z_s, z_x, z_y) = GLS posterior mean of Y[p] under P_D's joint Gaussian
        predicted_v[p, endpoint] = H_V(endpoint, P_D) @ [z_x, z_y]
        residual_v[p, endpoint]  = log_v_increments[p, endpoint] - predicted_v[p, endpoint]
    Tag each pair as:
        IS (in-sample)     pair_end_date in [D - window_size + 1 BD, D]
        OOS (out-of-sample) pair_end_date > D
        pre                pair_end_date < D - window_size + 1 BD

Done at three D's hardcoded in CALIBRATION_DATES (roughly 25/50/75 quantile positions in
the 2025 rolling-fit range; fixed rather than quantile-derived so PNG filenames stay
stable across panel extensions).

Usage:
    .venv\\Scripts\\python predicted_residuals.py
"""

import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyarrow import feather

from scripts.realised_innovations import realised_innovations_with_global_params
from scripts.rolling_calibration import DATE_FROM, DATE_TO, run_name_for
from utils.bergomi_two_factor import BergomiTwoFactorParams, observation_matrix_v_constant_tenor
from utils.cache_paths import MIN_RAW_DAYS, PANEL_TENOR_DAYS, ROOT, run_subdir, to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, ForwardVariancePanel, assemble_panel,
)
from utils.spot_data import FWD_PROXY_ROOT, daily_log_fwd_returns_for_panel_pairs, local_vol_per_panel_pair

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60
RUN_NAME = run_name_for(window_size=WINDOW_SIZE)

# Fixed calibration dates (roughly 25/50/75 quantile of the 2025 rolling-fit range).
# Hardcoded rather than quantile-derived so the output filenames stay stable across panel
# extensions; `params_at_date` will raise if any of these is not in the current fit_dates.
CALIBRATION_DATES = (
    dt.date(2025, 7, 1),
    dt.date(2025, 9, 25),
    dt.date(2025, 12, 22),
)

# Numbered figure prefixes: one PNG per calibration date.  Aligned positionally with
# CALIBRATION_DATES so the earliest -> 19, latest -> 21.  See README.md.
FIGURE_NUMBERS = ("19", "20", "21")


def feather_path() -> Path:
    """Path to the canonical 60bd rolling fit's params_timeseries.feather."""
    return run_subdir(name=RUN_NAME) / "params_timeseries.feather"


def params_at_date(fit_table: dict, target_date: dt.date) -> tuple[BergomiTwoFactorParams, float, float, int]:
    """Return (params, rho_sx, rho_sy, fit_row) for the row whose date matches target_date."""
    fit_dates = [dt.date.fromisoformat(value) for value in fit_table["date"]]
    fit_row = fit_dates.index(target_date)
    sigma_r_columns = sorted(
        [col for col in fit_table if col.startswith("sigma_r_strip_")],
        key=lambda col: int(col.rsplit("_", 1)[1]),
    )
    params = BergomiTwoFactorParams(
        k_x=float(fit_table["k_x"][fit_row]), k_y=float(fit_table["k_y"][fit_row]),
        theta=float(fit_table["theta"][fit_row]), rho_xy=float(fit_table["rho_xy"][fit_row]),
        nu=float(fit_table["nu"][fit_row]),
        sigma_r_vector=np.array([float(fit_table[col][fit_row]) for col in sigma_r_columns]),
    )
    rho_sx = float(fit_table["rho_sx"][fit_row])
    rho_sy = float(fit_table["rho_sy"][fit_row])
    return params, rho_sx, rho_sy, fit_row


def predicted_v_endpoints(
    panel: ForwardVariancePanel, z_x: np.ndarray, z_y: np.ndarray, params: BergomiTwoFactorParams,
) -> np.ndarray:
    """Predicted log_v_increment per pair per endpoint = H_V @ [z_x, z_y].  Shape (n_pairs, n_tenors)."""
    h_v = observation_matrix_v_constant_tenor(tenor_grid_years=panel.tenor_grid_years, params=params)
    z_xy = np.column_stack([z_x, z_y])
    return z_xy @ h_v.T


def split_pairs_by_window(
    panel: ForwardVariancePanel, calibration_date: dt.date, window_size: int,
) -> dict:
    """Tag each panel pair as IS / OOS / pre relative to the window ending at calibration_date."""
    fit_dates_iso = [str(panel.dates[int(end_index)]) for end_index in panel.pair_end_indices]
    pair_end_dates = [dt.date.fromisoformat(value) for value in fit_dates_iso]
    calibration_index = panel.dates.index(calibration_date)
    window_start_index = max(0, calibration_index - window_size + 1)
    window_start_date = panel.dates[window_start_index]
    is_mask = np.array([window_start_date <= d <= calibration_date for d in pair_end_dates])
    oos_mask = np.array([d > calibration_date for d in pair_end_dates])
    pre_mask = np.array([d < window_start_date for d in pair_end_dates])
    return {"is_mask": is_mask, "oos_mask": oos_mask, "pre_mask": pre_mask, "pair_end_dates": pair_end_dates}


def per_strip_r_squared(observed: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """R^2 = 1 - SS_res / SS_tot per column, against the column's own mean."""
    ss_res = np.sum(residual * residual, axis=0)
    centered = observed - observed.mean(axis=0, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1.0e-30)


def plot_residuals_for_calibration_date(  # noqa: PLR0913
    calibration_date: dt.date, masks: dict, observed: np.ndarray, predicted: np.ndarray,
    residual: np.ndarray, tenor_labels_days: np.ndarray, figure_number: str,
) -> None:
    """Three-panel figure for one D: residual time series (per tenor, IS/OOS colored), R^2 bars, scatter."""
    figure_name = f"{figure_number} predicted_residuals v {calibration_date.isoformat()} {ROOT}"
    figure, (axis_top, axis_mid, axis_bottom) = plt.subplots(
        num=figure_name, nrows=3, ncols=1, figsize=(13, 12),
    )
    colormap = plt.get_cmap("Dark2")
    pair_end_dates = masks["pair_end_dates"]
    n_tenors = residual.shape[1]

    is_dates = [d for d, keep in zip(pair_end_dates, masks["is_mask"], strict=True) if keep]
    if is_dates:
        axis_top.axvspan(min(is_dates), max(is_dates), color="gray", alpha=0.15, label="IS window")
    for tenor_index in range(n_tenors):
        axis_top.plot(
            pair_end_dates, residual[:, tenor_index], "-", color=colormap(tenor_index % 8),
            linewidth=0.8, alpha=0.7, label=f"{tenor_labels_days[tenor_index]:.0f}d",
        )
    axis_top.axhline(0, color="black", linewidth=0.4)
    axis_top.set_ylabel("residual = observed - predicted (log_V increment)")
    axis_top.set_xlabel("Pair end date")
    axis_top.set_title(
        f"log_V increment residuals at calibration D = {calibration_date.isoformat()} ({WINDOW_SIZE}bd window).  "
        f"Shaded = IS pairs.",
    )
    axis_top.legend(loc="upper left", ncol=4, fontsize=8)
    axis_top.grid(visible=True, alpha=0.3)

    is_mask = masks["is_mask"]
    oos_mask = masks["oos_mask"]
    r2_is = per_strip_r_squared(observed=observed[is_mask], residual=residual[is_mask])
    r2_oos = per_strip_r_squared(observed=observed[oos_mask], residual=residual[oos_mask])
    bar_positions = np.arange(n_tenors)
    width = 0.4
    axis_mid.bar(bar_positions - width / 2, r2_is, width=width, color=colormap(0), label="IS")
    axis_mid.bar(bar_positions + width / 2, r2_oos, width=width, color=colormap(1), label="OOS")
    axis_mid.set_xticks(bar_positions)
    axis_mid.set_xticklabels([f"{days:.0f}d" for days in tenor_labels_days])
    axis_mid.set_ylabel("R^2 of log_V increment residual")
    axis_mid.set_ylim(min(0, float(min(r2_is.min(), r2_oos.min())) - 0.05), 1.05)
    axis_mid.axhline(0, color="black", linewidth=0.4)
    axis_mid.legend(loc="upper right")
    axis_mid.grid(visible=True, alpha=0.3)
    axis_mid.set_title(f"Per-tenor R^2: IS (n={is_mask.sum()}) vs OOS (n={oos_mask.sum()})")

    axis_bottom.scatter(
        predicted[is_mask].ravel(), observed[is_mask].ravel(),
        s=8, color=colormap(0), alpha=0.5, label="IS",
    )
    axis_bottom.scatter(
        predicted[oos_mask].ravel(), observed[oos_mask].ravel(),
        s=8, color=colormap(1), alpha=0.5, label="OOS",
    )
    combined = np.concatenate([observed.ravel(), predicted.ravel()])
    bound = float(np.abs(combined).max() * 1.05)
    axis_bottom.plot([-bound, bound], [-bound, bound], "k--", alpha=0.4, label="y = x")
    axis_bottom.set_xlabel("predicted log_V increment")
    axis_bottom.set_ylabel("observed log_V increment")
    axis_bottom.set_xlim(-bound, bound)
    axis_bottom.set_ylim(-bound, bound)
    axis_bottom.set_title("Predicted vs observed log_V increment (all tenors, IS/OOS colored)")
    axis_bottom.legend(loc="upper left")
    axis_bottom.grid(visible=True, alpha=0.3)

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def main() -> None:
    """Generate IS/OOS predicted-residual diagnostic at three calibration dates."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    full_panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=PANEL_TENOR_DAYS,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=MIN_RAW_DAYS, min_expiries=5,
        max_extrapolation_fraction=0.10,
    )
    full_spot_returns = daily_log_fwd_returns_for_panel_pairs(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices,
        fixing_index=FIXING_INDEX_DEFAULT, root=FWD_PROXY_ROOT,
    )
    full_sigma_s_per_pair = local_vol_per_panel_pair(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices,
        fixing_index=FIXING_INDEX_DEFAULT, root=FWD_PROXY_ROOT,
    )

    table = feather.read_table(str(feather_path()))
    fit_table = {name: list(table.column(name).to_pylist()) for name in table.column_names}
    logger.info("Calibration dates: %s", [d.isoformat() for d in CALIBRATION_DATES])

    endpoint_tenor_days = (full_panel.tenor_grid_years * N_BUSINESS_DAYS_PER_YEAR).round(0)
    for figure_number, calibration_date in zip(FIGURE_NUMBERS, CALIBRATION_DATES, strict=True):
        params, rho_sx, rho_sy, _row = params_at_date(fit_table=fit_table, target_date=calibration_date)
        records = realised_innovations_with_global_params(
            full_panel=full_panel, full_spot_returns=full_spot_returns,
            full_sigma_s_per_pair=full_sigma_s_per_pair, params=params, rho_sx=rho_sx, rho_sy=rho_sy,
        )
        z_x = np.asarray(records["z_x"])
        z_y = np.asarray(records["z_y"])
        predicted_v = predicted_v_endpoints(panel=full_panel, z_x=z_x, z_y=z_y, params=params)
        residual_v = full_panel.log_v_increments - predicted_v
        masks = split_pairs_by_window(panel=full_panel, calibration_date=calibration_date, window_size=WINDOW_SIZE)
        logger.info(
            "D=%s: IS pairs=%d, OOS pairs=%d, pre pairs=%d",
            calibration_date.isoformat(), int(masks["is_mask"].sum()), int(masks["oos_mask"].sum()),
            int(masks["pre_mask"].sum()),
        )
        plot_residuals_for_calibration_date(
            calibration_date=calibration_date, masks=masks,
            observed=full_panel.log_v_increments, predicted=predicted_v, residual=residual_v,
            tenor_labels_days=endpoint_tenor_days, figure_number=figure_number,
        )

    plt.show()


if __name__ == "__main__":
    main()
