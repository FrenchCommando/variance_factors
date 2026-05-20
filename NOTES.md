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

### Per-pair sigma_S

Spot row of `M` uses **per-pair** `sigma_S(t_start, t_end)` from the
front-tenor variance swap (per pair, time-varying), not a global panel
empirical std.  Replaces a global-sigma_S baseline that was off by 3-4x
on regime-shock-adjacent dates; together with the advance-step fix below
it unwinds a `rho_SX` bias and pushes `rho_xy / nu` meaningfully higher
on the rolling 60bd fit.

### Daily-step cumulative variance

The Bergomi advance step

    V_advanced_i * (tau_i - dt)  =  V_i * tau_i  -  daily_step_cumulative_variance

requires `daily_step_cumulative_variance`, the cumulative variance over
one business day `[snap_t, snap_t + 1 BD]`.  Production formula (see
`utils/data_assembly.daily_step_cumulative_variance_min_dtes`):

```
daily_step_cumulative_variance  =  min(X_1dte_snap,  X_2dte_scaled)

X_n_dte  =  2 * LogSwap_t^{t + n BD}  *  (advance_years / tau_to_close_n)
```

with both DTE candidates read from the SPXW varswap at the snap.  Two
choices baked in:

1. **Snap-to-close correction.**  The cached `LogSwapMid` integrates
   from the snap time to the expiry's PM close.  At a 15:55 snap this
   carries `~0.009 BD` more variance window than the snap-to-snap
   advance step needs; scaling by `advance_years / tau_to_close`
   removes it.  Tiny correction now (`~ x 0.991`) but material if the
   snap moves -- at 14:00 it would be `~ x 0.823`.
2. **`min` of 1-DTE and 2-DTE.**  The truncated `sum dK / K^2 * OTM`
   integral is structurally inflated on third-Friday SPXW listings
   because they inherit CBOE's third-Friday strike chain (deep-OTM puts
   down to `K ~ 200`) that adjacent regular weeklies miss (stop near
   `K ~ 2400`) -- even though they are the PM-settled sibling of the
   AM-settled SPX SET that "OPEX" canonically refers to.  On any pair
   the third-Friday SPXW listing falls at exactly one of 1-DTE
   (Thu -> Fri) or 2-DTE (Wed -> Fri); `min` always picks the cleaner
   regular-weekly reading and rejects the inflated one.  On pairs that
   don't touch a third-Friday SPXW expiration, both candidates are
   close and `min` is just slightly conservative.

The min-of-two-DTEs approach trades a small downward bias in
`daily_step_cumulative_variance` on third-Friday-SPXW-adjacent pairs
(the cleaner weekly is still missing some wing contribution) for
mechanical robustness against the strike-listing artifact.  The
artifact itself is the subject of the sibling `jelly_roll` project,
which is working on a wing-extrapolated `LogSwapMidCorrected` column.
Once that lands, the underlying `LogSwapMid` reads here can switch to
the corrected column; the `min` collapse will then be defensive cheap
insurance rather than the load-bearing fix.

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

### Half-day snap shift

NYSE half-days (1 pm ET close: day after Thanksgiving, July 3 when
July 4 is a weekday, December 24 when Christmas falls Tuesday-Friday)
need both a different snap and a different obs-day time accrual.  The
default 15:55 ET snap (cache index 28500) reads ~3 hours of stale
post-close data on those dates; the regular 6.5-hour market session
also over-weights what's actually available to accrue.

`utils.calendar_utils.half_days` lists the affected dates;
`utils.data_assembly.fixing_for_obs_date` shifts the cache slot to
12:55 ET (`HALF_DAY_FIXING_INDEX = 17700`, the 5-minute pre-close offset
of the regular snap mirrored into a half-day) and flags the obs day as
early-close.  `intraday_time_to_expiry(..., is_early_close=True)` then
uses a 3.5-hour market session for the obs-day fraction.  Half-day
handling on the *expiration* day (a PM-settled option whose expiry
itself falls on a half-day) is not modelled -- the expiry day still
contributes a full day of variance.

Empirical impact (60bd rolling fit, panel range covers three half-days
2025-07-03, 2025-11-28, 2025-12-24): `k_y < 0.01` pegs drop 26 -> 19
on SPX and 11 -> 4 on SPXW, the same magnitude as one SKIP_DATES block
removes.  Median parameters move <= 8% (k_y SPX: 1.25 -> 1.35); LL on
the full-panel single fit barely moves (4576 -> 4561 SPX, 3658 -> 3638
SPXW).  The fix is correcting a small per-strip bias on the 3 half-day
obs dates that was previously absorbed by `k_y` collapsing on windows
spanning them.

## Empirical findings - SPX (60bd rolling, 2025-01-02 .. 2026-03-20)

| param | median | std |
|---|---:|---:|
| k_X (1/yr) | 5.60 | 1.58 |
| k_Y (1/yr) | 1.35 | 0.86 |
| theta      | 0.31 | 0.12 |
| rho_xy     | 0.79 | 0.26 |
| nu         | 1.20 | 0.29 |
| rho_SX     | -0.73 | 0.14 |
| rho_SY     | -0.84 | 0.13 |

Full-panel single fit: `k_x=7.85, k_y=0.97, theta=0.24, rho_xy=0.97,
nu=1.57, rho_sx=-0.75, rho_sy=-0.89`; `LL=4561`.  rho_xy lands near the
upper bound (det ~ 0.004) -- a shift from the prior pre-half-day basin
(rho_xy=0.83, det ~ 0.16) at essentially the same LL (4576 -> 4561):
the full-panel likelihood has a wide near-optimum plateau.

242/242 windows successful.  Pegs: `k_y < 0.01` 19/242; `nu > 49` 0;
`theta < 0.011` 0; `rho_xy > 0.99` 25/242; `rho_SX < -0.99` 0/242;
`rho_SY < -0.99` 8/242.  PSD frontier (det < 0.01): 74/242 (31%).

V-endpoint R^2 across the 7 endpoints (matching-window decomposition):
0.88 at 21d, 0.96 at 42d, 0.97 at 63d, 0.98 at 126d, 0.96 at 189d,
0.94 at 252d, 0.90 at 378d.

Cumulative-V vol-of-V at 63 BD: empirical median 1.40, implied 1.45.
Term-structure decay alpha (= -OLS slope of log std vs log tenor
across the 7 endpoints): empirical median 0.62, implied 0.49;
rough-vol benchmark (H ~ 0.1) is 0.4.

## Empirical findings - SPXW (60bd rolling, 2025-01-02 .. 2026-03-20)

Truncated 4-strip grid (no 252/378 BD endpoints -- SPXW listings don't
reach that far), so direct comparison to SPX is not apples-to-apples in
tenor coverage.

| param | median | std |
|---|---:|---:|
| k_X (1/yr) | 7.71 | 0.90 |
| k_Y (1/yr) | 1.66 | 1.28 |
| theta      | 0.22 | 0.20 |
| rho_xy     | 0.82 | 0.35 |
| nu         | 1.38 | 0.30 |
| rho_SX     | -0.78 | 0.18 |
| rho_SY     | -0.76 | 0.15 |

Full-panel single fit: `k_x=7.82, k_y=0.70, theta=0.17, rho_xy=0.994,
nu=1.62, rho_sx=-0.89, rho_sy=-0.87`; `LL=3638`.  Lands at the rho_xy
upper bound (det ~ 0.003) -- the SPXW data is pushing the
inter-factor correlation harder than SPX does.

242/242 windows successful.  Pegs: `k_y < 0.01` 4/242 (vs 19 for SPX);
`nu > 49` 0; `theta < 0.011` 0; `rho_xy > 0.99` 23/242 (vs 25 for SPX);
`rho_SX < -0.99` 7/242 (vs 0); `rho_SY < -0.99` 10/242 (vs 8).  PSD
frontier (det < 0.01): **127/242 (52%)** vs 74/242 (31%) for SPX --
SPXW binds the PSD constraint much more often, consistent with the
shorter tenor grid giving the optimizer less long-end leverage to
separate the two factors.

V-endpoint R^2 across the 5 endpoints (matching-window decomposition):
0.89 at 21d, 0.94 at 42d, 0.96 at 63d, 0.97 at 126d, 0.96 at 189d --
front R^2 essentially matches SPX (0.88 at 21d); SPXW misses the
long-tenor (252/378 BD) endpoints SPX has.

Cumulative-V vol-of-V at 63 BD: empirical median 1.37, implied 1.44
(essentially matches SPX at 1.40 / 1.45).  Term-structure decay alpha
across 5 endpoints: empirical median 0.64, implied 0.53; rough-vol
benchmark 0.40.  Both alphas are further from the rough-vol target
than SPX's (0.62 / 0.49) -- the truncated grid biases the OLS slope
steeper.

## Roots

Both `SPX` and `SPXW` are supported; pick via the `VARIANCE_FACTORS_ROOT`
env var.  Defaults wired in `utils/cache_paths.py`:

| ROOT | tenor grid (BD) | min_raw_days | reason |
|---|---|---:|---|
| SPX  | 21, 42, 63, 126, 189, 252, 378 | 7 | LEAPS chain reaches 18 months; front-week SPX carries a vol risk premium that doesn't lie on the long-tenor forward-variance curve |
| SPXW | 21, 42, 63, 126, 189            | 2 | only ~10 months of forward listings, so the 252/378 BD endpoints can't be reached; daily-weekly chain reaches 2 BD safely under the min-DTE advance step |

The `daily_step_cumulative_variance` reads above use SPXW expiries
regardless of panel ROOT (`spot_data.FWD_PROXY_ROOT = "SPXW"`).

## References

- Lorenzo Bergomi, *Stochastic Volatility Modeling*, Chapman & Hall, 2016
  (n-factor forward-variance model with spot-vol correlations).
- Bergomi, *Smile Dynamics II*, Risk magazine, 2005.
