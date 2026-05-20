"""Rolling 2-factor + spot Bergomi calibration via joint Gaussian likelihood, sigma_R free.

For each rolling window, optimize the (7 + n_strips) joint vector
    (k_x, k_y, theta, rho_xy, nu, rho_sx, rho_sy, sigma_r[0..n-1])
under `bergomi_likelihood.joint_negative_log_likelihood`.

sigma_S is **per-pair**, sqrt(2 * LogSwap_t_start^{t_end} / tau) from the front-tenor varswap.
sigma_R is **free per strip per window** -- pinning it drives k_y to its lower bound in
28-48 % of windows (see NOTES.md).

L-BFGS-B with warm starts: previous-window full fitted vector carries through.  On iter=0
or `not success` (PSD-barrier freeze), retry from the cold start.

Output (under `out/rolling_{N}bd/`):
    params_timeseries.feather  -- one row per window
    + parameter time-series + sigma_R-per-strip PNGs

Usage:
    .venv\\Scripts\\python rolling_calibration.py
"""

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from pyarrow import feather
from scipy.optimize import minimize

from utils.bergomi_likelihood import N_DYNAMIC_PARAMS_WITH_SPOT, joint_negative_log_likelihood
from utils.cache_paths import MIN_RAW_DAYS, PANEL_TENOR_DAYS, ROOT, run_subdir, to_image_path
from utils.data_assembly import (
    FIXING_INDEX_DEFAULT, N_BUSINESS_DAYS_PER_YEAR, ForwardVariancePanel, assemble_panel, slice_panel,
)
from utils.spot_data import FWD_PROXY_ROOT, daily_log_fwd_returns_for_panel_pairs, local_vol_per_panel_pair

logger = logging.getLogger(__name__)

DATE_FROM = dt.date(2025, 1, 2)
DATE_TO = dt.date(2026, 3, 20)
WINDOW_SIZES = (20, 40, 60)
PROGRESS_LOG_EVERY = 1

# Numbered figure prefixes; PNG outputs are referenced by these numbers in README.md.
FIGURE_NUMBER_PARAMS = {20: "05", 40: "07", 60: "09"}
FIGURE_NUMBER_SIGMA_R = {20: "06", 40: "08", 60: "10"}


def run_name_for(window_size: int) -> str:
    """Output sub-directory name for a rolling fit at the given window size."""
    return f"rolling_{window_size}bd"


PANEL_SPECS = (
    ("k_x", "k_X (1/yr)", True), ("k_y", "k_Y (1/yr)", True), ("theta", "theta", False),
    ("rho_xy", "rho_XY", False), ("nu", "nu", False), ("log_likelihood", "log_likelihood", False),
)

# Cold start for the 7 dynamic params.  rho_sx, rho_sy at -0.7 (typical SPX leverage); rho_xy at
# 0.5 places the start at det ~ 0.21, well inside the PSD region so cold-restart has somewhere
# to go (rho_xy = 0 with the same rho_S triples sits at det ~ 0.02 -- nominally PSD but
# practically on the boundary, defeating cold-restart).
COLD_START_DYNAMIC = np.array([8.0, 0.5, 0.2, 0.5, 1.0, -0.7, -0.7])
DYNAMIC_BOUNDS_TUPLE = (
    (0.0001, 50.0),     # k_x
    (0.0001, 50.0),     # k_y
    (0.01, 0.99),       # theta
    (-0.5, 0.999),      # rho_xy
    (0.1, 50.0),        # nu
    (-0.999, 0.999),    # rho_sx
    (-0.999, 0.999),    # rho_sy
)
SIGMA_R_BOUND = (0.001, 1.0)
SIGMA_R_COLD = 0.05


def make_full_bounds(n_strips: int) -> tuple:
    """Append n_strips sigma_R bounds to the dynamic bounds."""
    return DYNAMIC_BOUNDS_TUPLE + tuple(SIGMA_R_BOUND for _ in range(n_strips))


def make_cold_start(n_strips: int) -> np.ndarray:
    """Concatenate the dynamic cold start with n_strips sigma_R initial guesses."""
    return np.concatenate([COLD_START_DYNAMIC, np.full(n_strips, SIGMA_R_COLD)])


@dataclass(frozen=True)
class WindowFitResult:
    """One row of the rolling joint parameter time series."""

    date: str
    window_start_date: str
    k_x: float
    k_y: float
    theta: float
    rho_xy: float
    nu: float
    rho_sx: float
    rho_sy: float
    sigma_r_vector: tuple[float, ...]
    log_likelihood: float
    n_dates: int
    iterations: int
    function_evaluations: int
    success: bool
    message: str


def slice_pair_array(
    pair_array: np.ndarray, full_panel: ForwardVariancePanel, start_index: int, end_index: int,
) -> np.ndarray:
    """Apply slice_panel's pair-mask to any per-pair-indexed array (spot returns, sigma_S)."""
    pair_mask = (full_panel.pair_end_indices > start_index) & (full_panel.pair_end_indices < end_index)
    return pair_array[pair_mask]


def initial_guess(fallback_start: np.ndarray, warm_start: np.ndarray | None) -> np.ndarray:
    """Per-window starting vector: warm-start if available, else cold."""
    base = warm_start if warm_start is not None else fallback_start
    return base.copy()


def joint_nll_free_sigma_r(
    full_vector: np.ndarray, panel: ForwardVariancePanel, spot_returns: np.ndarray, sigma_s_per_pair: np.ndarray,
) -> float:
    """Split the joint vector into (dynamic, sigma_R) and call the joint NLL."""
    dynamic_vector = full_vector[:N_DYNAMIC_PARAMS_WITH_SPOT]
    sigma_r_vector = full_vector[N_DYNAMIC_PARAMS_WITH_SPOT:]
    return joint_negative_log_likelihood(
        dynamic_vector=dynamic_vector, panel=panel, spot_returns=spot_returns, sigma_s=sigma_s_per_pair,
        fixed_sigma_r=sigma_r_vector,
    )


def fit_window(
    window_panel: ForwardVariancePanel, window_spot_returns: np.ndarray, start_vector: np.ndarray, bounds: tuple,
    sigma_s_per_pair: np.ndarray,
) -> dict:
    """Run one L-BFGS-B 2-factor + spot MLE on the window."""
    result = minimize(
        fun=joint_nll_free_sigma_r, x0=start_vector,
        args=(window_panel, window_spot_returns, sigma_s_per_pair), method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000},
    )
    return {
        "Vector": result.x, "NegLogLikelihood": float(result.fun), "Success": bool(result.success),
        "Message": str(result.message), "Iterations": int(result.nit), "FunctionEvaluations": int(result.nfev),
    }


def fit_one_window(  # noqa: PLR0913
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    window_index: int, window_size: int, fallback_start: np.ndarray, warm_start: np.ndarray | None,
    bounds: tuple, n_windows: int,
) -> WindowFitResult:
    """Fit one rolling window and return the row record.

    Detect-and-cold-restart: warm-start can land in the PSD-barrier dead zone where the
    objective is constant (1e12 penalty), gradient is zero, L-BFGS-B reports ABNORMAL with
    iter=0.  Retry from the cold start.
    """
    end_index = window_index + window_size
    window_panel = slice_panel(full_panel=full_panel, start_index=window_index, end_index=end_index)
    window_spot_returns = slice_pair_array(
        pair_array=full_spot_returns, full_panel=full_panel, start_index=window_index, end_index=end_index,
    )
    window_sigma_s = slice_pair_array(
        pair_array=full_sigma_s_per_pair, full_panel=full_panel, start_index=window_index, end_index=end_index,
    )
    start_vector = initial_guess(fallback_start=fallback_start, warm_start=warm_start)
    fit_meta = fit_window(
        window_panel=window_panel, window_spot_returns=window_spot_returns, start_vector=start_vector, bounds=bounds,
        sigma_s_per_pair=window_sigma_s,
    )
    if not fit_meta["Success"] or fit_meta["Iterations"] == 0:
        logger.warning(
            "Window %d (end=%s) warm-start failed (%s, iter=%d); retrying from cold start",
            window_index, window_panel.dates[-1], fit_meta["Message"], fit_meta["Iterations"],
        )
        cold_start = initial_guess(fallback_start=fallback_start, warm_start=None)
        fit_meta = fit_window(
            window_panel=window_panel, window_spot_returns=window_spot_returns, start_vector=cold_start,
            bounds=bounds, sigma_s_per_pair=window_sigma_s,
        )
    vector = fit_meta["Vector"]
    log_likelihood = -fit_meta["NegLogLikelihood"]
    if not fit_meta["Success"]:
        logger.warning(
            "Window %d (end=%s) did not converge after cold-restart: %s",
            window_index, window_panel.dates[-1], fit_meta["Message"],
        )
    if (window_index + 1) % PROGRESS_LOG_EVERY == 0 or window_index == n_windows - 1:
        logger.info(
            "Window %d/%d end=%s k_x=%.2f k_y=%.3f theta=%.3f rho=%.3f nu=%.3f rho_sx=%.3f rho_sy=%.3f LL=%.2f iter=%d",
            window_index + 1, n_windows, window_panel.dates[-1], vector[0], vector[1], vector[2], vector[3], vector[4],
            vector[5], vector[6], log_likelihood, fit_meta["Iterations"],
        )
    sigma_r_vector = tuple(float(value) for value in vector[N_DYNAMIC_PARAMS_WITH_SPOT:])
    return WindowFitResult(
        date=str(window_panel.dates[-1]), window_start_date=str(window_panel.dates[0]),
        k_x=float(vector[0]), k_y=float(vector[1]), theta=float(vector[2]), rho_xy=float(vector[3]),
        nu=float(vector[4]), rho_sx=float(vector[5]), rho_sy=float(vector[6]), sigma_r_vector=sigma_r_vector,
        log_likelihood=log_likelihood, n_dates=len(window_panel.dates), iterations=fit_meta["Iterations"],
        function_evaluations=fit_meta["FunctionEvaluations"], success=fit_meta["Success"], message=fit_meta["Message"],
    )


def warm_start_from_result(result: WindowFitResult) -> np.ndarray:
    """Convert a successful fit row back into the full optimizer vector (dynamic + sigma_R)."""
    dynamic = [result.k_x, result.k_y, result.theta, result.rho_xy, result.nu, result.rho_sx, result.rho_sy]
    return np.array([*dynamic, *result.sigma_r_vector])


def run_rolling(
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    window_size: int,
) -> list[WindowFitResult]:
    """Loop over windows, fit each with the full warm-start vector."""
    n_dates = len(full_panel.dates)
    n_strips = len(full_panel.strip_tenors_years)
    if n_dates < window_size:
        msg = f"panel has {n_dates} dates, fewer than window {window_size}"
        raise ValueError(msg)
    n_windows = n_dates - window_size + 1
    fallback_start = make_cold_start(n_strips=n_strips)
    bounds = make_full_bounds(n_strips=n_strips)

    results: list[WindowFitResult] = []
    warm_start: np.ndarray | None = None
    for window_index in range(n_windows):
        result = fit_one_window(
            full_panel=full_panel, full_spot_returns=full_spot_returns,
            full_sigma_s_per_pair=full_sigma_s_per_pair, window_index=window_index,
            window_size=window_size, fallback_start=fallback_start, warm_start=warm_start, bounds=bounds,
            n_windows=n_windows,
        )
        results.append(result)
        if result.success:
            warm_start = warm_start_from_result(result=result)
    return results


def results_to_columns(results: list[WindowFitResult]) -> dict:
    """Flatten per-row records into a dict-of-lists; expand sigma_r_vector to per-strip columns."""
    columns: dict = {}
    for field in dataclasses.fields(WindowFitResult):
        if field.name == "sigma_r_vector":
            n_strips = len(results[0].sigma_r_vector) if results else 0
            for strip_index in range(n_strips):
                columns[f"sigma_r_strip_{strip_index}"] = [
                    result.sigma_r_vector[strip_index] for result in results
                ]
        else:
            columns[field.name] = [getattr(result, field.name) for result in results]
    return columns


def write_feather_table(results: list[WindowFitResult], run_dir: Path) -> Path:
    """Write the rolling joint parameter time series to feather."""
    feather_path = run_dir / "params_timeseries.feather"
    feather.write_feather(pa.table(results_to_columns(results=results)), str(feather_path))
    return feather_path


PLOT_PANEL_SPECS = (
    *PANEL_SPECS,
    ("rho_sx", "rho_SX", False),
    ("rho_sy", "rho_SY", False),
)


def plot_parameters_timeseries(results: list[WindowFitResult], window_size: int) -> None:
    """Multi-panel plot of the rolling parameter time series."""
    figure_name = f"{FIGURE_NUMBER_PARAMS[window_size]} rolling {window_size}bd params {ROOT}"
    n_panels = len(PLOT_PANEL_SPECS)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    figure, axes = plt.subplots(num=figure_name, nrows=n_rows, ncols=n_cols, figsize=(13, 11), sharex=True)
    dates = [dt.date.fromisoformat(result.date) for result in results]
    colormap = plt.get_cmap("Dark2")
    for index, (field_name, label, show_halflife) in enumerate(PLOT_PANEL_SPECS):
        axis = axes[index // n_cols, index % n_cols]
        line_color = colormap(index % 8)
        values = np.asarray([getattr(result, field_name) for result in results])
        axis.plot(dates, values, "-", color=line_color, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(visible=True, alpha=0.3)
        if show_halflife:
            secondary = axis.twinx()
            half_life_days = np.log(2) / values * N_BUSINESS_DAYS_PER_YEAR
            secondary.plot(dates, half_life_days, ":", color=line_color, alpha=0.4)
            secondary.set_ylabel("half-life (BD)", color=line_color, fontsize=8)
            secondary.tick_params(axis="y", labelcolor=line_color, labelsize=8)
    for empty_index in range(n_panels, n_rows * n_cols):
        figure.delaxes(axes[empty_index // n_cols, empty_index % n_cols])
    figure.suptitle(
        f"{ROOT} 2-factor + spot rolling parameters (joint MLE), window={window_size} BD, "
        f"per-pair sigma_S from front varswap", fontsize=11,
    )
    axes[-1, 0].set_xlabel("Evaluation date (window end)")
    axes[-1, 1].set_xlabel("Evaluation date (window end)")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def plot_sigma_r_timeseries(
    results: list[WindowFitResult], strip_tenors_years: np.ndarray, window_size: int,
) -> None:
    """One panel per strip of the free rolling sigma_R."""
    n_strips = len(strip_tenors_years)
    figure_name = f"{FIGURE_NUMBER_SIGMA_R[window_size]} rolling {window_size}bd sigma_r {ROOT}"
    n_cols = 2
    n_rows = (n_strips + n_cols - 1) // n_cols
    figure, axes = plt.subplots(num=figure_name, nrows=n_rows, ncols=n_cols, figsize=(13, 3 * n_rows), sharex=True)
    dates = [dt.date.fromisoformat(result.date) for result in results]
    colormap = plt.get_cmap("Dark2")
    for strip_index in range(n_strips):
        axis = axes[strip_index // n_cols, strip_index % n_cols]
        line_color = colormap(strip_index % 8)
        values = np.asarray([result.sigma_r_vector[strip_index] for result in results])
        axis.plot(dates, values, "-", color=line_color, linewidth=1.5)
        tenor_days = round(strip_tenors_years[strip_index] * N_BUSINESS_DAYS_PER_YEAR)
        axis.set_ylabel(f"sigma_R strip {strip_index} ({tenor_days}d)")
        axis.grid(visible=True, alpha=0.3)
    for empty_index in range(n_strips, n_rows * n_cols):
        figure.delaxes(axes[empty_index // n_cols, empty_index % n_cols])
    figure.suptitle(
        f"{ROOT} rolling sigma_R per strip (free per window), window={window_size} BD", fontsize=11,
    )
    axes[-1, 0].set_xlabel("Evaluation date (window end)")
    if n_strips > 1:
        axes[-1, 1].set_xlabel("Evaluation date (window end)")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(to_image_path(name=figure_name))


def run_one_window_size(
    full_panel: ForwardVariancePanel, full_spot_returns: np.ndarray, full_sigma_s_per_pair: np.ndarray,
    window_size: int,
) -> None:
    """Run, save, and plot one rolling fit at the given window size."""
    logger.info("=== Rolling fit: window_size=%d ===", window_size)
    results = run_rolling(
        full_panel=full_panel, full_spot_returns=full_spot_returns,
        full_sigma_s_per_pair=full_sigma_s_per_pair, window_size=window_size,
    )
    run_dir = run_subdir(name=run_name_for(window_size=window_size))
    run_dir.mkdir(parents=True, exist_ok=True)
    feather_path = write_feather_table(results=results, run_dir=run_dir)
    logger.info("Wrote %s (%d rows)", feather_path, len(results))
    plot_parameters_timeseries(results=results, window_size=window_size)
    plot_sigma_r_timeseries(
        results=results, strip_tenors_years=full_panel.strip_tenors_years, window_size=window_size,
    )


def main() -> None:
    """Run the rolling with-spot fit on the SPX panel for every WINDOW_SIZES entry."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    full_panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=PANEL_TENOR_DAYS,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=MIN_RAW_DAYS, min_expiries=5,
        max_extrapolation_fraction=0.10,
    )
    logger.info(
        "Panel: %d dates, %d pairs, %d strips", len(full_panel.dates), len(full_panel.pair_end_indices),
        len(full_panel.strip_tenors_years),
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
        "Per-pair sigma_S from front varswap: median=%.4f, p10=%.4f, p90=%.4f, min=%.4f, max=%.4f",
        float(np.median(full_sigma_s_per_pair)), float(np.percentile(full_sigma_s_per_pair, 10)),
        float(np.percentile(full_sigma_s_per_pair, 90)), float(full_sigma_s_per_pair.min()),
        float(full_sigma_s_per_pair.max()),
    )

    for window_size in WINDOW_SIZES:
        run_one_window_size(
            full_panel=full_panel, full_spot_returns=full_spot_returns,
            full_sigma_s_per_pair=full_sigma_s_per_pair, window_size=window_size,
        )


if __name__ == "__main__":
    main()
