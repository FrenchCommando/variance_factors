"""Compute and store realised OU innovations (Z_X, Z_Y) per panel pair.

For each panel pair (t_start, t_end) whose end date matches a row in a rolling fit's
`params_timeseries.feather`, GLS-infer the realised innovation (Z_S, Z_X, Z_Y) from the
joint observation Y = (spot_return, log_xi_increments) using that window's fitted params.

Z_S equals spot_return / sigma_S exactly (the spot row of the observation matrix is
noiseless), so we expose `spot_return` directly along with the GLS Z_X, Z_Y on the OU
innovation scale (Var(Z_j) ~ dt under the fitted params).

Output:
    out/rolling_{N}bd/realised_innovations.feather
    out/full_panel/realised_innovations.feather
columns: date, spot_return, z_x, z_y, dt_years.

Usage:
    .venv\\Scripts\\python realised_innovations.py
"""

import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from pyarrow import feather

from scripts.rolling_calibration import DATE_FROM, DATE_TO, ROOT, WINDOW_SIZES, run_name_for
from utils.bergomi_likelihood import gls_shock_estimate, innovation_covariance_with_spot, joint_observation_matrix
from utils.bergomi_two_factor import BergomiTwoFactorParams
from utils.cache_paths import OUT_DIR, to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, TENOR_DAYS_BENCHMARK, ForwardVariancePanel, assemble_panel,
)
from utils.spot_data import FWD_PROXY_ROOT, daily_log_fwd_returns_for_panel_pairs, local_vol_per_panel_pair

logger = logging.getLogger(__name__)

# Numbered figure prefixes for the realised-innovations plots; see README.md.
FIGURE_NUMBER_ROLLING = {20: "11", 40: "12", 60: "13"}
FIGURE_NUMBER_FULL_PANEL = "14"


def gls_innovation_for_pair(  # noqa: PLR0913
    panel: ForwardVariancePanel, full_spot_returns: np.ndarray, pair_index: int,
    params: BergomiTwoFactorParams, rho_sx: float, rho_sy: float, sigma_s_for_pair: float,
) -> tuple[str, float, float, float, float]:
    """Per-pair (date, spot_return, z_x, z_y, dt_years) given dynamic params and this pair's sigma_S."""
    end_index = int(panel.pair_end_indices[pair_index])
    pair_end_date = panel.dates[end_index]
    observation_matrix = joint_observation_matrix(
        strip_tenors_years=panel.strip_tenors_years, params=params, sigma_s=sigma_s_for_pair,
    )
    dt_years = float(panel.dt_years[pair_index])
    process_covariance_dt = innovation_covariance_with_spot(
        k_x=params.k_x, k_y=params.k_y, rho_xy=params.rho_xy, rho_sx=rho_sx, rho_sy=rho_sy,
        dt_years=dt_years,
    )
    observation = np.concatenate([
        full_spot_returns[pair_index : pair_index + 1], panel.log_xi_increments[pair_index],
    ])
    shock = gls_shock_estimate(
        observation=observation, observation_matrix=observation_matrix,
        process_covariance_dt=process_covariance_dt, sigma_r_vector=params.sigma_r_vector,
    )
    return str(pair_end_date), float(observation[0]), float(shock[1]), float(shock[2]), dt_years


def realised_innovations_from_panel(
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    fit_table: dict,
) -> dict:
    """Per-pair (date, spot_return, z_x, z_y, dt_years) using each pair's matching window's params."""
    fit_dates = [dt.date.fromisoformat(value) for value in fit_table["date"]]
    fit_index = {date: index for index, date in enumerate(fit_dates)}
    sigma_r_columns = sorted(
        [col for col in fit_table if col.startswith("sigma_r_strip_")],
        key=lambda col: int(col.rsplit("_", 1)[1]),
    )

    out_dates: list[str] = []
    out_spot: list[float] = []
    out_z_x: list[float] = []
    out_z_y: list[float] = []
    out_dt_years: list[float] = []
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
        date, spot, z_x, z_y, dt_years = gls_innovation_for_pair(
            panel=full_panel, full_spot_returns=full_spot_returns, pair_index=pair_index,
            params=params, rho_sx=float(fit_table["rho_sx"][fit_row]),
            rho_sy=float(fit_table["rho_sy"][fit_row]),
            sigma_s_for_pair=float(full_sigma_s_per_pair[pair_index]),
        )
        out_dates.append(date)
        out_spot.append(spot)
        out_z_x.append(z_x)
        out_z_y.append(z_y)
        out_dt_years.append(dt_years)
    return {
        "date": out_dates, "spot_return": out_spot, "z_x": out_z_x, "z_y": out_z_y, "dt_years": out_dt_years,
    }


def realised_innovations_with_global_params(  # noqa: PLR0913
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    params: BergomiTwoFactorParams, rho_sx: float, rho_sy: float,
) -> dict:
    """Per-pair (date, spot_return, z_x, z_y, dt_years) using one global params set."""
    out_dates: list[str] = []
    out_spot: list[float] = []
    out_z_x: list[float] = []
    out_z_y: list[float] = []
    out_dt_years: list[float] = []
    for pair_index, _end_index in enumerate(full_panel.pair_end_indices):
        date, spot, z_x, z_y, dt_years = gls_innovation_for_pair(
            panel=full_panel, full_spot_returns=full_spot_returns, pair_index=pair_index,
            params=params, rho_sx=rho_sx, rho_sy=rho_sy,
            sigma_s_for_pair=float(full_sigma_s_per_pair[pair_index]),
        )
        out_dates.append(date)
        out_spot.append(spot)
        out_z_x.append(z_x)
        out_z_y.append(z_y)
        out_dt_years.append(dt_years)
    return {
        "date": out_dates, "spot_return": out_spot, "z_x": out_z_x, "z_y": out_z_y, "dt_years": out_dt_years,
    }


def plot_realised_innovations(records: dict, label: str, figure_number: str) -> None:
    """Four-panel plot of realised innovations: spot, OU innovations, cumulative paths."""
    pair_dates = [dt.date.fromisoformat(value) for value in records["date"]]
    spot = np.asarray(records["spot_return"])
    z_x = np.asarray(records["z_x"])
    z_y = np.asarray(records["z_y"])
    cumulative_spot = np.cumsum(spot)
    cumulative_z_x = np.cumsum(z_x)
    cumulative_z_y = np.cumsum(z_y)

    figure_name = f"{figure_number} realised_innovations {label} {ROOT}"
    figure, axes = plt.subplots(num=figure_name, nrows=4, ncols=1, figsize=(13, 11), sharex=True)
    colormap = plt.get_cmap("Dark2")

    axes[0].plot(pair_dates, spot, "-", color=colormap(0), linewidth=0.8, label="spot return (= Z_S)")
    axes[0].axhline(0, color="black", linewidth=0.4)
    axes[0].set_ylabel("daily spot return")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].plot(pair_dates, z_x, "-", color=colormap(1), linewidth=0.8, label="Z_X (short factor)")
    axes[1].plot(pair_dates, z_y, "-", color=colormap(2), linewidth=0.8, label="Z_Y (long factor)", alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.4)
    axes[1].set_ylabel("OU innovation")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)

    axes[2].plot(pair_dates, cumulative_spot, "-", color=colormap(0), linewidth=1.2, label="cum spot (~ log F path)")
    axes[2].set_ylabel("cumulative spot return")
    axes[2].grid(visible=True, alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=9)

    axes[3].plot(pair_dates, cumulative_z_x, "-", color=colormap(1), linewidth=1.2, label="sum Z_X")
    axes[3].plot(pair_dates, cumulative_z_y, "-", color=colormap(2), linewidth=1.2, label="sum Z_Y")
    axes[3].set_ylabel("cumulative OU innovation")
    axes[3].set_xlabel("Date")
    axes[3].grid(visible=True, alpha=0.3)
    axes[3].legend(loc="upper left", fontsize=9)

    figure.suptitle(f"{ROOT} {label}: realised innovations and cumulative paths", fontsize=11)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def feather_path_for(window_size: int) -> Path:
    """Realised-innovations feather path for the given rolling window size."""
    return OUT_DIR / run_name_for(window_size=window_size) / "realised_innovations.feather"


def params_path_for(window_size: int) -> Path:
    """Params time-series feather path for the given rolling window size."""
    return OUT_DIR / run_name_for(window_size=window_size) / "params_timeseries.feather"


def main() -> None:
    """Compute realised innovations for every window size that has a fitted feather."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

    for window_size in WINDOW_SIZES:
        params_path = params_path_for(window_size=window_size)
        if not params_path.exists():
            logger.warning("Skipping %d BD -- %s missing", window_size, params_path)
            continue
        table = feather.read_table(str(params_path))
        fit_table = {name: list(table.column(name).to_pylist()) for name in table.column_names}
        records = realised_innovations_from_panel(
            full_panel=full_panel, full_spot_returns=full_spot_returns,
            full_sigma_s_per_pair=full_sigma_s_per_pair, fit_table=fit_table,
        )
        out_path = feather_path_for(window_size=window_size)
        feather.write_feather(pa.table(records), str(out_path))
        z_x_arr = np.asarray(records["z_x"])
        z_y_arr = np.asarray(records["z_y"])
        spot_arr = np.asarray(records["spot_return"])
        dt_arr = np.asarray(records["dt_years"])
        logger.info(
            "window=%d BD: %d pairs, std(z_x)=%.4f std(z_y)=%.4f std(spot)=%.5f median(dt)=%.4f -> %s",
            window_size, len(records["date"]), z_x_arr.std(), z_y_arr.std(), spot_arr.std(), np.median(dt_arr),
            out_path,
        )
        plot_realised_innovations(
            records=records, label=f"rolling_{window_size}bd", figure_number=FIGURE_NUMBER_ROLLING[window_size],
        )

    plt.show()


if __name__ == "__main__":
    main()
