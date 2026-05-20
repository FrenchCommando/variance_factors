"""Full-panel (non-rolling) 2-factor + spot Bergomi calibration.

Single MLE on the entire panel using `bergomi_likelihood.joint_negative_log_likelihood`.
Produces a stable global parameter set that complements the rolling fit's regime-tracking
output.

Output (under `out/full_panel/`):
    params.feather              -- single row of fitted parameters
    realised_innovations.feather -- per-pair (date, spot_return, z_x, z_y, dt_years) under
                                   the global params

Usage:
    .venv\\Scripts\\python run_calibration.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from pyarrow import feather

from scripts.realised_innovations import (
    FIGURE_NUMBER_FULL_PANEL, plot_realised_innovations, realised_innovations_with_global_params,
)
from scripts.rolling_calibration import DATE_FROM, DATE_TO, fit_window, make_cold_start, make_full_bounds
from utils.bergomi_likelihood import N_DYNAMIC_PARAMS_WITH_SPOT
from utils.bergomi_two_factor import BergomiTwoFactorParams
from utils.cache_paths import MIN_RAW_DAYS, PANEL_TENOR_DAYS, ROOT, run_subdir
from utils.data_assembly import FIXING_INDEX_DEFAULT, assemble_panel
from utils.spot_data import FWD_PROXY_ROOT, daily_log_fwd_returns_for_panel_pairs, local_vol_per_panel_pair

logger = logging.getLogger(__name__)

RUN_NAME = "full_panel"


def run_dir() -> Path:
    """Return the output directory for the full-panel fit (namespaced by ROOT)."""
    return run_subdir(name=RUN_NAME)


def write_params_feather(
    vector: np.ndarray, fit_meta: dict, n_dates: int, n_strips: int, out_path: Path,
) -> None:
    """Write the single-row params feather."""
    columns: dict = {
        "k_x": [float(vector[0])], "k_y": [float(vector[1])], "theta": [float(vector[2])],
        "rho_xy": [float(vector[3])], "nu": [float(vector[4])], "rho_sx": [float(vector[5])],
        "rho_sy": [float(vector[6])],
    }
    for strip_index in range(n_strips):
        columns[f"sigma_r_strip_{strip_index}"] = [
            float(vector[N_DYNAMIC_PARAMS_WITH_SPOT + strip_index]),
        ]
    columns["log_likelihood"] = [-float(fit_meta["NegLogLikelihood"])]
    columns["n_dates"] = [n_dates]
    columns["iterations"] = [int(fit_meta["Iterations"])]
    columns["function_evaluations"] = [int(fit_meta["FunctionEvaluations"])]
    columns["success"] = [bool(fit_meta["Success"])]
    columns["message"] = [str(fit_meta["Message"])]
    feather.write_feather(pa.table(columns), str(out_path))


def main() -> None:
    """Run the full-panel fit, write params + realised innovations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    full_panel = assemble_panel(
        root=ROOT, date_from=DATE_FROM, date_to=DATE_TO, tenor_days=PANEL_TENOR_DAYS,
        fixing_index=FIXING_INDEX_DEFAULT, min_raw_days=MIN_RAW_DAYS, min_expiries=5,
        max_extrapolation_fraction=0.10,
    )
    n_strips = len(full_panel.strip_tenors_years)
    spot_returns = daily_log_fwd_returns_for_panel_pairs(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices,
        fixing_index=FIXING_INDEX_DEFAULT, root=FWD_PROXY_ROOT,
    )
    sigma_s_per_pair = local_vol_per_panel_pair(
        dates=full_panel.dates, pair_end_indices=full_panel.pair_end_indices,
        fixing_index=FIXING_INDEX_DEFAULT, root=FWD_PROXY_ROOT,
    )
    logger.info(
        "Panel: %d dates, %d pairs, %d strips; sigma_S per-pair: median=%.4f, min=%.4f, max=%.4f",
        len(full_panel.dates), len(full_panel.pair_end_indices), n_strips,
        float(np.median(sigma_s_per_pair)), float(sigma_s_per_pair.min()), float(sigma_s_per_pair.max()),
    )

    cold_start = make_cold_start(n_strips=n_strips)
    bounds = make_full_bounds(n_strips=n_strips)

    fit_meta = fit_window(
        window_panel=full_panel, window_spot_returns=spot_returns, start_vector=cold_start,
        bounds=bounds, sigma_s_per_pair=sigma_s_per_pair,
    )
    vector = fit_meta["Vector"]
    log_likelihood = -float(fit_meta["NegLogLikelihood"])
    logger.info(
        "Full-panel fit: success=%s iter=%d LL=%.2f", fit_meta["Success"], fit_meta["Iterations"], log_likelihood,
    )
    logger.info(
        "  k_x=%.3f k_y=%.3f theta=%.3f rho_xy=%.3f nu=%.3f rho_sx=%.3f rho_sy=%.3f",
        vector[0], vector[1], vector[2], vector[3], vector[4], vector[5], vector[6],
    )
    sigma_r_vector = vector[N_DYNAMIC_PARAMS_WITH_SPOT:]
    logger.info("  sigma_r per strip: %s", " ".join(f"{value:.4f}" for value in sigma_r_vector))

    output_dir = run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "params.feather"
    write_params_feather(
        vector=vector, fit_meta=fit_meta, n_dates=len(full_panel.dates), n_strips=n_strips, out_path=params_path,
    )
    logger.info("Wrote %s", params_path)

    params = BergomiTwoFactorParams(
        k_x=float(vector[0]), k_y=float(vector[1]), theta=float(vector[2]),
        rho_xy=float(vector[3]), nu=float(vector[4]),
        sigma_r_vector=np.asarray(sigma_r_vector, dtype=np.float64).copy(),
    )
    records = realised_innovations_with_global_params(
        full_panel=full_panel, full_spot_returns=spot_returns, full_sigma_s_per_pair=sigma_s_per_pair,
        params=params, rho_sx=float(vector[5]), rho_sy=float(vector[6]),
    )
    rli_path = output_dir / "realised_innovations.feather"
    feather.write_feather(pa.table(records), str(rli_path))
    z_x_arr = np.asarray(records["z_x"])
    z_y_arr = np.asarray(records["z_y"])
    spot_arr = np.asarray(records["spot_return"])
    logger.info(
        "Wrote %s (%d pairs).  std(z_x)=%.4f std(z_y)=%.4f std(spot)=%.5f",
        rli_path, len(records["date"]), z_x_arr.std(), z_y_arr.std(), spot_arr.std(),
    )
    plot_realised_innovations(records=records, label="full_panel", figure_number=FIGURE_NUMBER_FULL_PANEL)
    plt.show()


if __name__ == "__main__":
    main()
