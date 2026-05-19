"""Joint Gaussian likelihood of (front-fwd return, log_xi_increments) under 2-factor Bergomi + spot.

The joint observation per panel pair (t_start, t_end):

    Y_t  =  [ log F^{T=t_end}_{t_end} - log F^{T=t_end}_{t_start} ;  log_xi_increments[t] ]

dimension n_strips + 1.  Under Bergomi with spot-vol correlations,

    Y_t = M_t @ Z_t + [ 0 ; epsilon_t ]
    Z_t = (Z_S_t, Z_X_t, Z_Y_t)  ~  N(0, Q_dt_full)
    epsilon_t ~ N(0, diag(sigma_R^2))

with M_t the (n_strips + 1) x 3 mapping (Z_S, Z_X, Z_Y) to observations:
    M_t[0, :]   = (sigma_S(t), 0, 0)         (spot innovation)
    M_t[i+1, :] = (0, H[i, X], H[i, Y])      (strip i)

and Q_dt the 3x3 OU-with-spot covariance:
    Q[S, S] = dt
    Q[X, X] = (1 - exp(-2 k_x dt)) / (2 k_x)
    Q[Y, Y] = (1 - exp(-2 k_y dt)) / (2 k_y)
    Q[S, X] = rho_SX * (1 - exp(-k_x dt)) / k_x
    Q[S, Y] = rho_SY * (1 - exp(-k_y dt)) / k_y
    Q[X, Y] = rho_xy * (1 - exp(-(k_x + k_y) dt)) / (k_x + k_y)

Sigma_t = M_t @ Q_dt @ M_t.T + diag(0, sigma_R^2).  Spot row carries no extra noise -- the
"noise" on that row comes from sigma_S(t) itself.  sigma_S is per-pair, set to the front-tenor
variance-swap rate (`spot_data.local_vol_per_panel_pair`).  sigma_R is per-strip and free in
the optimizer.  The dynamic vector

    (k_x, k_y, theta, rho_xy, nu, rho_sx, rho_sy),  length 7

is appended with n_strips sigma_R entries by the calibration scripts.
"""

from typing import TYPE_CHECKING

import numpy as np

from utils.bergomi_two_factor import (
    BergomiTwoFactorParams, observation_matrix_constant_tenor, observation_matrix_v_constant_tenor,
)

if TYPE_CHECKING:
    from utils.data_assembly import ForwardVariancePanel

N_DYNAMIC_PARAMS_WITH_SPOT = 7
RHO_SX_INDEX = 5
RHO_SY_INDEX = 6
# Soft barrier when the (rho_SX, rho_SY, rho_XY) triple gives a non-PSD 3x3 correlation.
# L-BFGS-B is unconstrained beyond box bounds, so without this barrier it can wander into
# infeasible territory and return -inf log-det / negative quadratic form.  A large finite
# constant steers the optimizer back without the gradient blowup that NaN/Inf would cause.
NON_PSD_PENALTY = 1.0e12


def innovation_covariance_with_spot(  # noqa: PLR0913
    k_x: float, k_y: float, rho_xy: float, rho_sx: float, rho_sy: float, dt_years: float,
) -> np.ndarray:
    """3x3 covariance of (Z_S, Z_X, Z_Y) at horizon dt_years."""
    q_ss = dt_years
    q_xx = (1.0 - np.exp(-2.0 * k_x * dt_years)) / (2.0 * k_x)
    q_yy = (1.0 - np.exp(-2.0 * k_y * dt_years)) / (2.0 * k_y)
    q_sx = rho_sx * (1.0 - np.exp(-k_x * dt_years)) / k_x
    q_sy = rho_sy * (1.0 - np.exp(-k_y * dt_years)) / k_y
    q_xy = rho_xy * (1.0 - np.exp(-(k_x + k_y) * dt_years)) / (k_x + k_y)
    return np.array([
        [q_ss, q_sx, q_sy],
        [q_sx, q_xx, q_xy],
        [q_sy, q_xy, q_yy],
    ])


def joint_observation_matrix(
    strip_tenors_years: np.ndarray, params: BergomiTwoFactorParams, sigma_s: float,
) -> np.ndarray:
    """(n_strips + 1, 3) matrix mapping (Z_S, Z_X, Z_Y) to (spot_return, log_xi_increments)."""
    n_strips = len(strip_tenors_years)
    matrix = np.zeros((n_strips + 1, 3))
    matrix[0, 0] = sigma_s
    matrix[1:, 1:] = observation_matrix_constant_tenor(strip_tenors_years=strip_tenors_years, params=params)
    return matrix


def joint_observation_matrix_v(
    tenor_grid_years: np.ndarray, params: BergomiTwoFactorParams, sigma_s: float,
) -> np.ndarray:
    """(n_endpoints + 1, 3) matrix mapping (Z_S, Z_X, Z_Y) to (spot_return, log_v_increments)."""
    n_endpoints = len(tenor_grid_years)
    matrix = np.zeros((n_endpoints + 1, 3))
    matrix[0, 0] = sigma_s
    matrix[1:, 1:] = observation_matrix_v_constant_tenor(tenor_grid_years=tenor_grid_years, params=params)
    return matrix


def unpack_dynamic_with_spot(
    dynamic_vector: np.ndarray, fixed_sigma_r: np.ndarray,
) -> tuple[BergomiTwoFactorParams, float, float]:
    """Split (k_x, k_y, theta, rho_xy, nu, rho_sx, rho_sy) -> (params, rho_sx, rho_sy)."""
    if dynamic_vector.shape[0] != N_DYNAMIC_PARAMS_WITH_SPOT:
        msg = f"dynamic_vector must be length {N_DYNAMIC_PARAMS_WITH_SPOT}; got {dynamic_vector.shape}"
        raise ValueError(msg)
    params = BergomiTwoFactorParams(
        k_x=float(dynamic_vector[0]), k_y=float(dynamic_vector[1]), theta=float(dynamic_vector[2]),
        rho_xy=float(dynamic_vector[3]), nu=float(dynamic_vector[4]), sigma_r_vector=fixed_sigma_r,
    )
    return params, float(dynamic_vector[RHO_SX_INDEX]), float(dynamic_vector[RHO_SY_INDEX])


def gls_shock_estimate(
    observation: np.ndarray, observation_matrix: np.ndarray, process_covariance_dt: np.ndarray,
    sigma_r_vector: np.ndarray,
) -> np.ndarray:
    """Posterior mean of (shock_S, shock_X, shock_Y) given the joint observation.

    `noise_diag = (0, sigma_R^2)`: the spot row carries no extra noise.  Solves the ridge
        shock_hat = (Q^{-1} + M.T diag(1/noise) M)^{-1} M.T diag(1/noise) observation
    treating the zero-noise spot row via a small numerical floor only inside the inverse.
    """
    inverse_noise = np.concatenate([np.array([1.0e12]), 1.0 / (sigma_r_vector * sigma_r_vector)])
    weighted_observation_matrix = observation_matrix * inverse_noise[:, None]
    precision = np.linalg.inv(process_covariance_dt) + observation_matrix.T @ weighted_observation_matrix
    rhs = observation_matrix.T @ (inverse_noise * observation)
    return np.linalg.solve(precision, rhs)


def joint_negative_log_likelihood(
    dynamic_vector: np.ndarray, panel: "ForwardVariancePanel", spot_returns: np.ndarray, sigma_s: np.ndarray,
    fixed_sigma_r: np.ndarray,
) -> float:
    """Joint Gaussian NLL of (spot_return, delta_strips) per pair.

    sigma_s is per-pair (length n_pairs) -- per-pair local annualized vol from the front-tenor
    variance swap.  Each pair's joint observation matrix uses its own sigma_s for the spot row.
    """
    params, rho_sx, rho_sy = unpack_dynamic_with_spot(dynamic_vector=dynamic_vector, fixed_sigma_r=fixed_sigma_r)
    correlation_determinant = (
        1.0 + 2.0 * rho_sx * rho_sy * params.rho_xy
        - rho_sx * rho_sx - rho_sy * rho_sy - params.rho_xy * params.rho_xy
    )
    if correlation_determinant <= 0.0:
        return NON_PSD_PENALTY
    n_strips = len(panel.strip_tenors_years)
    n_obs = n_strips + 1
    log_two_pi = float(np.log(2.0 * np.pi))
    noise_diag = np.concatenate([np.zeros(1), fixed_sigma_r * fixed_sigma_r])

    n_pairs = panel.log_xi_increments.shape[0]
    if spot_returns.shape[0] != n_pairs:
        msg = f"spot_returns length {spot_returns.shape[0]} != n_pairs {n_pairs}"
        raise ValueError(msg)
    if sigma_s.shape[0] != n_pairs:
        msg = f"sigma_s length {sigma_s.shape[0]} != n_pairs {n_pairs}"
        raise ValueError(msg)

    total = 0.0
    for step_index in range(n_pairs):
        matrix = joint_observation_matrix(
            strip_tenors_years=panel.strip_tenors_years, params=params, sigma_s=float(sigma_s[step_index]),
        )
        observation = np.concatenate([spot_returns[step_index : step_index + 1], panel.log_xi_increments[step_index]])
        dt_years = float(panel.dt_years[step_index])
        process_covariance_dt = innovation_covariance_with_spot(
            k_x=params.k_x, k_y=params.k_y, rho_xy=params.rho_xy, rho_sx=rho_sx, rho_sy=rho_sy, dt_years=dt_years,
        )
        sigma = matrix @ process_covariance_dt @ matrix.T + np.diag(noise_diag)
        sign, log_determinant = np.linalg.slogdet(sigma)
        if sign <= 0 or not np.isfinite(log_determinant):
            return NON_PSD_PENALTY
        quadratic = float(observation @ np.linalg.solve(sigma, observation))
        total += 0.5 * (quadratic + log_determinant + n_obs * log_two_pi)
    return total
