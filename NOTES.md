# variance_factors -- model + load-bearing choices

2-factor Bergomi + spot calibration via maximum likelihood on the per-day
joint Gaussian distribution of one-day log returns of forward variance
plus the matching front-fwd 1BD return.  The dynamic state is

    dW_S, dW_X, dW_Y    Cov(dW_S, dW_X) = rho_SX dt
                        Cov(dW_S, dW_Y) = rho_SY dt
                        Cov(dW_X, dW_Y) = rho_xy dt
    dZ^j_t = -k_j Z^j_t dt + dW^j_t,    j in {X, Y}    (k_S = 0)

Forward variance at expiry T as observed at t (the level itself has no
spot dependence; spot enters only through factor correlations):

    xi_t^T  =  xi_0^T * exp(omega x_t^T - 0.5 omega^2 chi(t, T))
    omega   =  2 nu
    x_t^T   =  alpha [(1 - theta) e^{-k_X (T-t)} Z_t^X + theta e^{-k_Y (T-t)} Z_t^Y]
    alpha   =  ((1-theta)^2 + theta^2 + 2 (1-theta) theta rho_xy)^{-1/2}

Front-fwd return: `log F_front[t+1] / F_front[t] ~ sigma_S sqrt(dt) Z_S`,
`Z_S` correlated with `(Z_X, Z_Y)` via `(rho_SX, rho_SY)`.

For each panel pair (t_start, t_end), the joint observation

    Y_t  =  [ spot_return_t ;  log_xi_increments_t ]

is `N(0, Sigma_t)` with

    Sigma_t  =  M_t @ Q_dt(t) @ M_t.T  +  diag(0, sigma_R^2)
    M_t[0, :]    =  (sigma_S(t), 0, 0)                       (spot row)
    M_t[i+1, :]  =  (0, H[i, X], H[i, Y])                    (strip i)
    H[i, j]      =  omega alpha w_j exp(-k_j Delta_i),  w = (1 - theta, theta)
    Q_dt(t)      =  one-step covariance of (Z_S, Z_X, Z_Y) over dt_years[t]
                    (closed form -- see bergomi_likelihood.innovation_covariance_with_spot)

The optimizer vector has length `7 + n_strips`:

    (k_x, k_y, theta, rho_xy, nu, rho_sx, rho_sy, sigma_r[0..n-1])

`sigma_S(t)` is **per-pair**, set to the front-tenor variance-swap rate
at the pair's start: `sqrt(2 * LogSwap_t_start^{t_end} / tau_years)`.
`sigma_R[i]` is **free per-strip per-window**.

Bounds in `rolling_calibration.DYNAMIC_BOUNDS_TUPLE`:

    k_x, k_y in [0.0001, 50] /yr
    theta    in [0.01, 0.99]
    rho_xy   in [-0.5, 0.999]
    nu       in [0.1, 50]
    rho_sx, rho_sy in [-0.999, 0.999]
    sigma_r[i]     in [0.001, 1.0]

## Load-bearing modelling choices

### Increment fit, not Kalman on levels

Under Bergomi at fixed expiration T, forward variance is a martingale,
so the constant-tenor residual `delta_t = log_xi[t+1] - log_xi_advanced[t]`
is mean zero and a linear combination of one-day OU innovations weighted
by the cross-section.  No panel-mean anchor needed; both `xi_t` and
`xi_{t+1}` enter each row of the likelihood at observable dates -- causal
by construction.

An earlier Kalman level fit needed a panel-mean anchor to subtract from
each row of `log_xi`.  That introduces look-ahead bias: every date's
filtered state depends on a baseline computed from dates after it.  The
"Kalman" naming has been dropped from this repo because the production
fit is plain MLE on increments; nothing is actually filtered.

### sigma_R must stay free per window

Pinning `sigma_R` to *anything* -- panel-fit values, the average rolling
PCA-residual std, or bid-ask-derived noise -- drives `k_y` to its lower
bound in 28-48 % of windows.  A too-small `sigma_R` leaves the model
nowhere to absorb persistent per-strip bias, and the cheapest workaround
for the optimizer is `k_y -> 0` (factor Y becomes permanent, infinite
memory, stationary variance -> infinity).  Free per-window `sigma_R`
removes this artifact: zero `k_y` pegs across the panel, with no other
peg appearing.

### Per-pair sigma_S and model-free advance

Two `sigma_S` fixes vs the prior global-sigma_S / V[front]*dt-advance
baseline:

1. Spot row of M uses **per-pair** `sigma_S(t_start, t_end)` from the
   front-tenor variance swap (per pair, time-varying), not a global
   panel-level empirical std.
2. Per-option advance uses the **model-free** cumulative variance
   `X = 2 * LogSwap_t_start^{t_end}` instead of the proxy
   `V[t, tau_first] * dt`.

Together these unwind a `rho_SX` bias and push `rho_xy / nu`
meaningfully higher on the rolling 60bd fit.

### PSD barrier + cold restart

The 3x3 `(Z_S, Z_X, Z_Y)` correlation must satisfy

    det = 1 + 2 rho_SX rho_SY rho_xy - rho_SX^2 - rho_SY^2 - rho_xy^2  >= 0.

L-BFGS-B is unconstrained beyond box bounds, so the likelihood returns
`NON_PSD_PENALTY = 1e12` whenever the candidate triple is infeasible.
Two mitigations live in `rolling_calibration`:

1. **Cold-restart on iter=0 or `not success`** -- a warm start sitting
   at the PSD barrier produces zero gradient, so the optimizer returns
   ABNORMAL with `iter=0`.  Retry from the cold start.
2. **PSD-interior cold start** -- `COLD_START_DYNAMIC` has
   `(rho_xy, rho_SX, rho_SY) = (0.5, -0.7, -0.7)`, det ~ 0.21.  Earlier
   `rho_xy = 0` at the same `rho_S` triples sat at det ~ 0.02 (boundary).

A Cholesky reparameterization would eliminate PSD failures by
construction; deferred unless the iter=0 rate after both mitigations
remains > 5 %.

### Skip dates

`data_assembly.SKIP_DATES` drops 2025-04-07/08/09 (US "Liberation Day"
tariff-shock window).  Those days drove 5-8 sigma residuals at long
strips on consecutive dates; any rolling window containing them
collapses `k_y` to its lower bound to absorb the non-Bergomi residual.

## Empirical findings (60bd rolling, SPX 2025-01-02 .. 2026-03-20)

| param | median | std |
|---|---:|---:|
| k_X (1/yr) | 7.89 | 2.27 |
| k_Y (1/yr) | 2.18 | 0.80 |
| theta      | 0.34 | 0.29 |
| rho_xy     | 0.74 | 0.15 |
| nu         | 1.72 | 0.47 |
| rho_SX     | -0.89 | 0.16 |
| rho_SY     | -0.90 | 0.04 |

242/242 windows successful, zero iter=0 retries.  Pegs: `k_y < 0.01` 0
times; `nu > 49` 0; `theta < 0.011` 0; `rho_xy > 0.99` 16/242;
`rho_SX < -0.99` 2/242 (down from 21 before the advance fix);
`rho_SY < -0.99` 0/242.  42/242 windows have det < 0.01 (close to PSD
frontier).

Cumulative-V vol-of-V at 63 BD: empirical median annualized std ~ 1.7,
implied 1.6 (signal only).  Term-structure decay alpha (= -OLS slope of
log std vs log tenor across 7 endpoints):  empirical median 0.58,
implied 0.48; rough-vol benchmark (H ~ 0.1) is 0.4.

## Extension

Currently using SPX expiries with a lot of trimming (front-week
expirations carry a vol risk premium that doesn't lie on the long-tenor
forward-variance curve; min_raw_days = 7).  SPXW gives more tenor
coverage and a cleaner front -- the natural next step is to repeat the
calibration on the SPXW panel.

## References

- Lorenzo Bergomi, *Stochastic Volatility Modeling*, Chapman & Hall, 2016
  (n-factor forward-variance model with spot-vol correlations).
- Bergomi, *Smile Dynamics II*, Risk magazine, 2005.
