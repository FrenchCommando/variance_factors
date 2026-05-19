"""Bergomi 2-factor model dataclass and observation-matrix primitives.

Pieces consumed by `bergomi_likelihood.joint_negative_log_likelihood`: parameter dataclass,
alpha normalization, observation matrix at constant tenors.  See NOTES.md for the model.

    State (OU): d Z^j_t = -k_j Z^j_t dt + d W^j_t,  Cov(W^X, W^Y) = rho_xy dt
    Forward variance:
        x_t^T = alpha [(1 - theta) e^{-k_X (T - t)} X_t + theta e^{-k_Y (T - t)} Y_t]
        alpha = ((1 - theta)^2 + theta^2 + 2 (1 - theta) theta rho_xy)^{-1/2}
        d log xi_t^{t+Delta}  =  omega alpha [(1-theta) e^{-k_X Delta} dW^X
                                              + theta e^{-k_Y Delta} dW^Y]
                                 + (deterministic drift)
        omega = 2 nu
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BergomiTwoFactorParams:
    """Five dynamic parameters + per-strip noise std.

    sigma_r_vector holds one observation-noise std per strip (length n_strips).  Free per
    strip because empirical residual std varies by ~3x across the SPX strip grid.
    """

    k_x: float
    k_y: float
    theta: float
    rho_xy: float
    nu: float
    sigma_r_vector: np.ndarray


def alpha_normalization(theta: float, rho_xy: float) -> float:
    """alpha = (sum_ij w_i w_j rho_ij)^{-1/2}, w = (1-theta, theta), rho_xy = rho_xy."""
    one_minus_theta = 1.0 - theta
    quadratic = one_minus_theta * one_minus_theta + theta * theta + 2.0 * one_minus_theta * theta * rho_xy
    return 1.0 / np.sqrt(quadratic)


def observation_matrix_constant_tenor(strip_tenors_years: np.ndarray, params: BergomiTwoFactorParams) -> np.ndarray:
    """H[i, j] = omega * alpha * w_j * e^{-k_j Delta_i}, shape (n_tenors, 2)."""
    omega = 2.0 * params.nu
    alpha = alpha_normalization(theta=params.theta, rho_xy=params.rho_xy)
    one_minus_theta = 1.0 - params.theta
    column_x = omega * alpha * one_minus_theta * np.exp(-params.k_x * strip_tenors_years)
    column_y = omega * alpha * params.theta * np.exp(-params.k_y * strip_tenors_years)
    return np.column_stack([column_x, column_y])


def observation_matrix_v_constant_tenor(
    tenor_grid_years: np.ndarray, params: BergomiTwoFactorParams,
) -> np.ndarray:
    """H_V[i, j] = (omega alpha / tau_i) w_j (1 - e^{-k_j tau_i}) / k_j, shape (n_tenors, 2).

    Derived from d log V_t^{t+tau} = (1/tau) integral_0^tau d log xi_t^{t+u} du; the tau-averaged
    H_xi over [0, tau].  Used by V-observable diagnostics; calibration target is the xi-strip H.
    """
    omega = 2.0 * params.nu
    alpha = alpha_normalization(theta=params.theta, rho_xy=params.rho_xy)
    one_minus_theta = 1.0 - params.theta
    factor = omega * alpha / tenor_grid_years
    column_x = factor * one_minus_theta * (1.0 - np.exp(-params.k_x * tenor_grid_years)) / params.k_x
    column_y = factor * params.theta * (1.0 - np.exp(-params.k_y * tenor_grid_years)) / params.k_y
    return np.column_stack([column_x, column_y])
