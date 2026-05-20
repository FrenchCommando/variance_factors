"""Assemble the (date, tenor) panel of log forward variance for the increment fit.

Pipeline:
1. For each business date in the window, load all available expirations from the
   cache and extract LogSwapMid at the daily fixing index (15:55 ET).  Strict read.
2. Convert each (date, expiration) cell to total annualized variance V = 2 LogSwap / tau.
3. Resample each date's term structure onto a constant tenor grid via PCHIP in
   sqrt(tau) -> V.  Drop dates whose grid would extrapolate too far past the longest
   observed expiration.
4. Convert to forward variance over consecutive strips, take logs.  Strip variance over
   (Delta_i, Delta_{i+1}) sits at midpoint tenor (Delta_i + Delta_{i+1}) / 2.

Output: the raw log_xi panel + Bergomi-advance increments (predicted-tomorrow residuals,
mean zero under Bergomi at fixed expiration).  Observation noise enters the likelihood as
a free per-strip vector parameter, not derived from bid-ask spreads.
"""

import datetime as dt
import logging
import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.interpolate import PchipInterpolator

from utils.cache_paths import cache_folder_log_swap, log_swap_path
from utils.calendar_utils import count_business_days, dates_iter, is_half_day, plus_days
from utils.intraday_time import intraday_time_to_expiry, is_am_settled
from utils.spot_data import FWD_PROXY_ROOT, read_log_swap_mid_at_fixing

logger = logging.getLogger(__name__)

N_BUSINESS_DAYS_PER_YEAR = 252

# Daily fixing in the 32401-point 08:00..17:00 ET cache: 15:55 ET = 28500.
# On NYSE half-days the snap shifts to 12:55 ET = 17700 (5 minutes before the 13:00 close,
# matching the default's 5-minute offset from the regular 16:00 close).
FIXING_INDEX_DEFAULT = 28500
HALF_DAY_FIXING_INDEX = 17700


def fixing_for_obs_date(date: dt.date, base_index: int = FIXING_INDEX_DEFAULT) -> tuple[int, bool]:
    """Effective (cache_index, is_early_close) for an observation date.

    On NYSE half-days (1pm close) the cache index shifts from 15:55 ET to 12:55 ET and the
    obs-day market session shrinks from 6.5 to 3.5 hours.  Half-day list lives in
    `utils.calendar_utils.half_days`.
    """
    if is_half_day(date=date):
        return HALF_DAY_FIXING_INDEX, True
    return base_index, False

# Tenor grids in business days.  Benchmark matches the 1m..2y configuration used in
# Bergomi's published 2-factor calibrations; 504d (2y) endpoint dropped because the
# 441d-midpoint strip carries large measurement noise.
TENOR_DAYS_FULL = (1, 2, 3, 5, 10, 21, 42, 63, 126, 189, 252, 378)
TENOR_DAYS_BENCHMARK = (21, 42, 63, 126, 189, 252, 378)
# SPXW has ~10-12 months of forward visibility (no LEAPS chain like SPX), so the long end of
# the benchmark grid is unreachable.  Truncated grid covers what SPXW consistently provides.
TENOR_DAYS_SPXW = (21, 42, 63, 126, 189)

# April 2025 tariff-shock days drove 5-8 sigma residuals at long strips; rolling windows
# containing them collapse k_y to its lower bound to absorb the non-Bergomi residual.
SKIP_DATES: frozenset[dt.date] = frozenset({
    dt.date(2025, 4, 7),
    dt.date(2025, 4, 8),
    dt.date(2025, 4, 9),
})

MIN_POINTS_FOR_INTERPOLATION = 2
MIN_PANEL_DATES = 2

_DATE_FILE_PATTERN = re.compile(r"^(\d{8})\.feather$")
_DATE_DIR_PATTERN = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class TermStructurePoint:
    """One (expiration, total annualized variance) cell for a single date."""

    expiration: dt.date
    raw_days: int
    timestamp_index: int
    time_to_expiry: float
    total_variance: float


@dataclass(frozen=True)
class ForwardVariancePanel:
    """Constant-tenor log forward variance ready for the increment fit.

    Pairs whose end date is in SKIP_DATES, or where SKIP_DATES sits strictly between start
    and end, are dropped (no NaN sentinel).  log_xi_increments / log_v_increments / dt_years /
    pair_end_indices are aligned and have length n_pairs <= n_dates - 1.

    Fields:
        dates: business dates that survived all filters, length n_dates.
        log_xi: log strip forward variance at constant midpoint tenor, shape (n_dates, n_strips).
        tenor_grid_years: endpoint tenor grid used for interpolation, length n_tenors.
        strip_tenors_years: midpoint tenors for forward-variance strips, length n_strips.
            These are the Delta = T - t the observation matrix uses.
        log_xi_increments: per-pair Bergomi-advance martingale residual
            log_xi[pair_end] - log_xi_advanced[pair].  Mean zero under Bergomi.  Shape (n_pairs, n_strips).
        dt_years: business years between the two dates of each pair, shape (n_pairs,).
        pair_end_indices: index into `dates` of each pair's end (later date), shape (n_pairs,).
        log_v_endpoints: log of constant-tenor total annualized variance at tenor_grid_years
            (the variance-swap-rate observable, dimensionless), shape (n_dates, n_tenors).
        log_v_increments: per-pair Bergomi-advance residual on the V observable,
            log_v_endpoints[pair_end] - log_v_advanced[pair], shape (n_pairs, n_tenors).

    """

    dates: tuple[dt.date, ...]
    log_xi: np.ndarray
    tenor_grid_years: np.ndarray
    strip_tenors_years: np.ndarray
    log_xi_increments: np.ndarray
    dt_years: np.ndarray
    pair_end_indices: np.ndarray
    log_v_endpoints: np.ndarray
    log_v_increments: np.ndarray


def build_varswap_index(root: str, date_from: dt.date, date_to: dt.date) -> dict[dt.date, list[dt.date]]:
    """Scan cache_log_swap/{root}/ and build {observation_date: [expirations]} for the date range.

    Replaces the contracts-cache lookup from the source repo with a direct filesystem walk:
    one pass over `cache_log_swap/{root}/{expiry}/`, collect every `<obs_date>.feather`, group
    by obs_date.  The cache is the source of truth here; if a file is present, the (expiry,
    obs_date) cell is available.
    """
    root_dir = cache_folder_log_swap() / root
    if not root_dir.exists():
        msg = f"Cache directory missing: {root_dir}"
        raise FileNotFoundError(msg)
    index: dict[dt.date, list[dt.date]] = {}
    for expiry_dir in root_dir.iterdir():
        if not expiry_dir.is_dir() or not _DATE_DIR_PATTERN.match(expiry_dir.name):
            continue
        expiry = dt.datetime.strptime(expiry_dir.name, "%Y%m%d").date()  # noqa: DTZ007
        for entry in expiry_dir.iterdir():
            match = _DATE_FILE_PATTERN.match(entry.name)
            if not match:
                continue
            obs_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()  # noqa: DTZ007
            if not (date_from <= obs_date <= date_to):
                continue
            index.setdefault(obs_date, []).append(expiry)
    for expirations in index.values():
        expirations.sort()
    return index


def extract_fixing_log_swap(log_swap_array: np.ndarray, fixing_index: int) -> float:
    """Return LogSwapMid at exactly fixing_index.  Raise on NaN or non-positive value."""
    value = float(log_swap_array[fixing_index])
    if not np.isfinite(value) or value <= 0:
        msg = f"LogSwapMid at fixing_index={fixing_index} is {value} (expected positive finite)"
        raise ValueError(msg)
    return value


def load_log_swap_mid_array(root: str, expiration: dt.date, observation_date: dt.date) -> np.ndarray | None:
    """Read the full 32401-point LogSwapMid array; None if the cache file is absent."""
    from pyarrow import feather  # noqa: PLC0415

    path = log_swap_path(root=root, expiration=expiration, observation_date=observation_date)
    if not path.exists():
        return None
    table = feather.read_table(str(path), columns=["LogSwapMid"])
    return table["LogSwapMid"].to_numpy(zero_copy_only=False).astype(np.float64)


def load_term_structure_for_date(
    root: str, date: dt.date, expirations: list[dt.date], fixing_index: int, min_raw_days: int,
) -> list[TermStructurePoint]:
    """Cross-section of total annualized variance across expirations for one date.

    Skips expirations with raw_days < min_raw_days (front weeklies carry a short-dated vol
    risk premium that doesn't lie on the long-tenor forward-variance term structure).  Also
    skips literal 0DTE.  On NYSE half-days, `fixing_index` is treated as the base
    (non-half-day) snap index and shifted to 12:55 ET internally; `time_to_expiry` is also
    computed with a 3.5-hour obs session.
    """
    am_settled = is_am_settled(root=root)
    effective_index, is_early_close = fixing_for_obs_date(date=date, base_index=fixing_index)
    points: list[TermStructurePoint] = []
    for expiration in expirations:
        raw_days = count_business_days(date_from=date, date_to=expiration)
        if raw_days < min_raw_days:
            continue
        log_swap_array = load_log_swap_mid_array(root=root, expiration=expiration, observation_date=date)
        if log_swap_array is None:
            msg = f"log_swap cache file for {root} expiration={expiration} date={date} is missing"
            raise ValueError(msg)
        log_swap_mid = extract_fixing_log_swap(log_swap_array=log_swap_array, fixing_index=effective_index)
        time_to_expiry = intraday_time_to_expiry(
            raw_days=raw_days, timestamp_index=effective_index, am_settled=am_settled,
            is_early_close=is_early_close,
        )
        if time_to_expiry <= 0:
            continue
        total_variance = 2.0 * log_swap_mid / time_to_expiry
        points.append(
            TermStructurePoint(
                expiration=expiration, raw_days=raw_days, timestamp_index=effective_index,
                time_to_expiry=time_to_expiry, total_variance=total_variance,
            ),
        )
    points.sort(key=lambda point: point.time_to_expiry)
    return points


def fit_pchip(points: list[TermStructurePoint]) -> tuple[PchipInterpolator, float] | None:
    """PCHIP in (sqrt(tau), V) for one date's term structure; None if too few distinct points.

    Holiday expirations can collide with the next business day's expiration when
    count_business_days walks past the holiday; average the collided rows so PCHIP's
    strictly-increasing-x precondition holds.
    """
    if len(points) < MIN_POINTS_FOR_INTERPOLATION:
        return None
    raw_taus = np.array([point.time_to_expiry for point in points])
    raw_variances = np.array([point.total_variance for point in points])
    taus, inverse_index = np.unique(raw_taus, return_inverse=True)
    summed_variances = np.bincount(inverse_index, weights=raw_variances)
    duplicate_counts = np.bincount(inverse_index)
    variances = summed_variances / duplicate_counts
    if len(taus) < MIN_POINTS_FOR_INTERPOLATION:
        return None
    interpolator = PchipInterpolator(x=np.sqrt(taus), y=variances, extrapolate=True)
    return interpolator, float(taus[-1])


def resample_term_structure(
    points: list[TermStructurePoint], tenor_grid_years: np.ndarray, max_extrapolation_fraction: float,
) -> np.ndarray | None:
    """PCHIP-resampled total variance at tenor_grid_years; None if grid extrapolates too far."""
    fit_result = fit_pchip(points=points)
    if fit_result is None:
        return None
    interpolator, longest_observed = fit_result
    if tenor_grid_years[-1] > (1.0 + max_extrapolation_fraction) * longest_observed:
        return None
    return np.asarray(interpolator(np.sqrt(tenor_grid_years)))


def forward_strip_variance(
    constant_tenor_variance: np.ndarray, tenor_grid_years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward variance over each consecutive (Delta_i, Delta_{i+1}) strip.

    xi_strip[i] = (V[i+1] * Delta_{i+1} - V[i] * Delta_i) / (Delta_{i+1} - Delta_i).
    Output sits at midpoint tenors (Delta_i + Delta_{i+1}) / 2.
    """
    cumulative_variance = constant_tenor_variance * tenor_grid_years[None, :]
    strip_lengths = np.diff(tenor_grid_years)
    strip_variance = np.diff(cumulative_variance, axis=1) / strip_lengths[None, :]
    strip_tenors_years = (tenor_grid_years[:-1] + tenor_grid_years[1:]) / 2.0
    return strip_tenors_years, strip_variance


@dataclass(frozen=True)
class CandidatePanel:
    """Pre-filter accumulator from the per-date assembly loop."""

    dates: list[dt.date]
    variance_rows: list[np.ndarray]
    points: list[list[TermStructurePoint]]
    n_no_expirations: int
    n_thin: int
    n_extrapolated: int


def collect_candidate_panel(  # noqa: PLR0913
    root: str, date_from: dt.date, date_to: dt.date, tenor_grid_years: np.ndarray, fixing_index: int,
    min_raw_days: int, min_expiries: int, max_extrapolation_fraction: float,
) -> CandidatePanel:
    """Iterate business days and produce per-date V rows that pass term-structure filters."""
    expirations_index = build_varswap_index(root=root, date_from=date_from, date_to=date_to)
    candidate_dates: list[dt.date] = []
    candidate_variance_rows: list[np.ndarray] = []
    candidate_points: list[list[TermStructurePoint]] = []
    n_no_expirations = 0
    n_thin = 0
    n_extrapolated = 0

    for date in dates_iter(date_from=date_from, date_to=date_to):
        if date in SKIP_DATES:
            continue
        expirations = expirations_index.get(date, [])
        if not expirations:
            n_no_expirations += 1
            continue
        points = load_term_structure_for_date(
            root=root, date=date, expirations=expirations, fixing_index=fixing_index, min_raw_days=min_raw_days,
        )
        if len(points) < min_expiries:
            n_thin += 1
            continue
        variance_row = resample_term_structure(
            points=points, tenor_grid_years=tenor_grid_years, max_extrapolation_fraction=max_extrapolation_fraction,
        )
        if variance_row is None:
            n_extrapolated += 1
            continue
        candidate_dates.append(date)
        candidate_variance_rows.append(variance_row)
        candidate_points.append(points)

    return CandidatePanel(
        dates=candidate_dates, variance_rows=candidate_variance_rows, points=candidate_points,
        n_no_expirations=n_no_expirations, n_thin=n_thin, n_extrapolated=n_extrapolated,
    )


def assemble_panel(  # noqa: PLR0913
    root: str, date_from: dt.date, date_to: dt.date, tenor_days: tuple[int, ...], fixing_index: int,
    min_raw_days: int, min_expiries: int, max_extrapolation_fraction: float,
) -> ForwardVariancePanel:
    """Build the constant-tenor log forward variance panel for the increment fit.

    Args:
        root: option root, e.g. "SPX" or "SPXW".
        date_from: first business date considered.
        date_to: last business date considered (inclusive).
        tenor_days: endpoint tenor grid in business days, strictly increasing.
        fixing_index: base snapshot index in the 32401-point cache (28500 = 15:55 ET on a
            regular session day; shifted to HALF_DAY_FIXING_INDEX on half-days).
        min_raw_days: drop expirations with fewer business days than this.
        min_expiries: minimum populated expirations for a date to be retained.
        max_extrapolation_fraction: dates whose tenor grid extrapolates beyond this fraction
            past the longest observed expiration are dropped.

    """
    tenor_grid_years = np.array(tenor_days, dtype=np.float64) / N_BUSINESS_DAYS_PER_YEAR
    if not np.all(np.diff(tenor_grid_years) > 0):
        msg = f"tenor_days must be strictly increasing, got {tenor_days}"
        raise ValueError(msg)

    candidate = collect_candidate_panel(
        root=root, date_from=date_from, date_to=date_to, tenor_grid_years=tenor_grid_years,
        fixing_index=fixing_index, min_raw_days=min_raw_days, min_expiries=min_expiries,
        max_extrapolation_fraction=max_extrapolation_fraction,
    )

    if not candidate.dates:
        msg = f"No candidate panel dates for {root} between {date_from} and {date_to}"
        raise ValueError(msg)

    constant_tenor_variance = np.vstack(candidate.variance_rows)
    strip_tenors_years, strip_variance = forward_strip_variance(
        constant_tenor_variance=constant_tenor_variance, tenor_grid_years=tenor_grid_years,
    )
    positive_strip_mask = np.all(strip_variance > 0, axis=1)
    n_non_positive_strip = int(np.sum(~positive_strip_mask))

    accepted_dates = [date for date, keep in zip(candidate.dates, positive_strip_mask, strict=True) if keep]
    accepted_strip_variance = strip_variance[positive_strip_mask, :]
    accepted_constant_tenor_variance = constant_tenor_variance[positive_strip_mask, :]
    accepted_points = [points for points, keep in zip(candidate.points, positive_strip_mask, strict=True) if keep]

    logger.info(
        "Assembly summary: kept=%d, no_expirations=%d, thin=%d, extrapolated=%d, non_positive_strip=%d",
        len(accepted_dates), candidate.n_no_expirations, candidate.n_thin, candidate.n_extrapolated,
        n_non_positive_strip,
    )

    if len(accepted_dates) < MIN_PANEL_DATES:
        msg = f"Insufficient panel dates ({len(accepted_dates)}) for {root} between {date_from} and {date_to}"
        raise ValueError(msg)

    log_xi = np.log(accepted_strip_variance)
    log_v_endpoints = np.log(accepted_constant_tenor_variance)

    valid_pair_indices = [
        index for index, (earlier, later) in enumerate(pairwise(accepted_dates))
        if not any(earlier < skipped < later for skipped in SKIP_DATES)
    ]
    pair_dt_years = np.array(
        [
            count_business_days(date_from=accepted_dates[index], date_to=accepted_dates[index + 1])
            for index in valid_pair_indices
        ], dtype=np.float64,
    ) / N_BUSINESS_DAYS_PER_YEAR
    pair_end_indices = np.array([index + 1 for index in valid_pair_indices], dtype=np.int64)
    log_xi_advanced, log_v_advanced = compute_advanced_predictors(
        accepted_points=accepted_points, valid_pair_indices=valid_pair_indices,
        pair_dt_years=pair_dt_years, tenor_grid_years=tenor_grid_years,
        accepted_dates=accepted_dates, fixing_index=fixing_index,
    )
    log_xi_increments = log_xi[pair_end_indices] - log_xi_advanced
    log_v_increments = log_v_endpoints[pair_end_indices] - log_v_advanced

    return ForwardVariancePanel(
        dates=tuple(accepted_dates), log_xi=log_xi, tenor_grid_years=tenor_grid_years,
        strip_tenors_years=strip_tenors_years, log_xi_increments=log_xi_increments,
        dt_years=pair_dt_years, pair_end_indices=pair_end_indices,
        log_v_endpoints=log_v_endpoints, log_v_increments=log_v_increments,
    )


def slice_panel(full_panel: ForwardVariancePanel, start_index: int, end_index: int) -> ForwardVariancePanel:
    """Build a ForwardVariancePanel from full_panel[start_index:end_index].

    Keeps pairs whose end index sits strictly inside (start_index, end_index); pair_end_indices
    are rebased to the sliced log_xi.
    """
    sliced_dates = full_panel.dates[start_index:end_index]
    sliced_log_xi = full_panel.log_xi[start_index:end_index, :]
    sliced_log_v_endpoints = full_panel.log_v_endpoints[start_index:end_index, :]
    pair_mask = (full_panel.pair_end_indices > start_index) & (full_panel.pair_end_indices < end_index)
    sliced_log_xi_increments = full_panel.log_xi_increments[pair_mask]
    sliced_log_v_increments = full_panel.log_v_increments[pair_mask]
    sliced_dt_years = full_panel.dt_years[pair_mask]
    sliced_pair_end_indices = full_panel.pair_end_indices[pair_mask] - start_index
    return ForwardVariancePanel(
        dates=sliced_dates, log_xi=sliced_log_xi, tenor_grid_years=full_panel.tenor_grid_years,
        strip_tenors_years=full_panel.strip_tenors_years, log_xi_increments=sliced_log_xi_increments,
        dt_years=sliced_dt_years, pair_end_indices=sliced_pair_end_indices,
        log_v_endpoints=sliced_log_v_endpoints, log_v_increments=sliced_log_v_increments,
    )


def compute_advanced_predictors(  # noqa: PLR0913
    accepted_points: list[list[TermStructurePoint]], valid_pair_indices: list[int],
    pair_dt_years: np.ndarray, tenor_grid_years: np.ndarray, accepted_dates: list[dt.date],
    fixing_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair predictor curves for tomorrow's strip-xi and endpoint-V via PCHIP advance.

    For each retained (start, end) pair:
        (1) Advance each option observed at start with tenor tau_i and total variance V_i to
            its predicted (V, tau) at end under Bergomi's martingale-at-fixed-expiration:
                tau_advanced  =  tau_i - dt
                V_advanced    =  (V_i * tau_i - daily_step_cumulative_variance) / tau_advanced
            daily_step_cumulative_variance is computed via
            `daily_step_cumulative_variance_min_dtes`: min of the snap-corrected 1-DTE and
            2-DTE SPXW varswap readings.
        (2) PCHIP-fit on (sqrt(tau_advanced), V_advanced) -- the predicted end-date curve in
            tenor units from end.
        (3) Evaluate at tenor_grid_years -> log V at endpoint tenors (log_v_advanced).
        (4) Forward-strip differencing of cumulative_advanced -> log_xi_advanced at midpoint tenors.
    """
    n_pairs = len(valid_pair_indices)
    n_tenors = tenor_grid_years.shape[0]
    n_strips = n_tenors - 1
    log_xi_advanced = np.empty((n_pairs, n_strips))
    log_v_advanced = np.empty((n_pairs, n_tenors))
    for output_index, pair_start_index in enumerate(valid_pair_indices):
        advance_years = float(pair_dt_years[output_index])
        start_date = accepted_dates[pair_start_index]
        end_date = accepted_dates[pair_start_index + 1]
        daily_step_cumulative_variance = daily_step_cumulative_variance_min_dtes(
            start_date=start_date, fixing_index=fixing_index, advance_years=advance_years,
        )
        advanced_pchip_fit = build_advanced_pchip(
            points=accepted_points[pair_start_index], advance_years=advance_years,
            daily_step_cumulative_variance=daily_step_cumulative_variance,
        )
        v_at_grid = np.asarray(advanced_pchip_fit(np.sqrt(tenor_grid_years)))
        if np.any(v_at_grid <= 0):
            msg = (
                f"Non-positive advanced V at pair {start_date}->{end_date} -- "
                f"PCHIP-extrapolated curve dips below zero"
            )
            raise ValueError(msg)
        cumulative_advanced = v_at_grid * tenor_grid_years
        strip_lengths = np.diff(tenor_grid_years)
        advanced_strip_variance = np.diff(cumulative_advanced) / strip_lengths
        if np.any(advanced_strip_variance <= 0):
            msg = (
                f"Non-positive advanced strip variance at pair {start_date}->{end_date} -- "
                f"term structure is not monotone enough"
            )
            raise ValueError(msg)
        log_xi_advanced[output_index] = np.log(advanced_strip_variance)
        log_v_advanced[output_index] = np.log(v_at_grid)
    return log_xi_advanced, log_v_advanced


def snap_corrected_dte_cumulative_variance(
    start_date: dt.date, raw_days: int, fixing_index: int, advance_years: float,
) -> float:
    """Cumulative variance over [snap, snap + advance_years] from one SPXW varswap reading.

    Reads 2 * LogSwap_t^{t + raw_days BD} at the obs-date's effective snap index and
    rescales by (advance_years / tau_to_close) to convert from the snap-to-PM-close
    integration window into the snap-to-snap window the advance step uses.  The scaling
    assumes variance is flat over the sub-day fraction between snap and close on the expiry
    date.  At a 15:55 ET snap the correction is ~0.991; at 14:00 ET it would be ~0.823 --
    worth doing structurally so the formula stays right if the snap time moves.

    On NYSE half-days, `fixing_index` is treated as the base (non-half-day) snap index;
    `start_date`'s effective values (12:55 ET, 3.5-hour session) are derived internally.
    """
    effective_index, is_early_close = fixing_for_obs_date(date=start_date, base_index=fixing_index)
    end_date = plus_days(date=start_date, n_days=raw_days)
    log_swap = read_log_swap_mid_at_fixing(
        root=FWD_PROXY_ROOT, expiration=end_date, observation_date=start_date, fixing_index=effective_index,
    )
    tau_to_close = intraday_time_to_expiry(
        raw_days=raw_days, timestamp_index=effective_index, am_settled=is_am_settled(root=FWD_PROXY_ROOT),
        is_early_close=is_early_close,
    )
    return 2.0 * log_swap * (advance_years / tau_to_close)


def daily_step_cumulative_variance_min_dtes(
    start_date: dt.date, fixing_index: int, advance_years: float,
) -> float:
    """Daily-step cumulative variance: min of snap-corrected 1-DTE and 2-DTE SPXW varswap reads.

    Both DTE candidates give scaled cumulative-variance estimates for the same advance-step
    window [snap, snap + advance_years].  `min` rejects whichever happens to span a third-Friday
    SPXW expiration, where the truncated `sum dK/K^2 * OTM` integral is structurally inflated
    by deep-OTM put strikes that adjacent weeklies are missing (third-Friday SPXW lists down to
    K~200 vs weeklies' K~2400 -- it inherits the CBOE third-Friday strike chain even though it
    is the PM-settled sibling of the AM-settled SPX SET that "OPEX" canonically refers to).
    The third-Friday listing falls at 1-DTE on Thu->Fri pairs and at 2-DTE on Wed->Fri pairs;
    either way `min` picks the cleaner regular-weekly reading.

    This is the production formula -- see NOTES.md "Daily-step cumulative variance" for why
    `min` over the two front DTEs is the right object to use here.
    """
    x_1dte = snap_corrected_dte_cumulative_variance(
        start_date=start_date, raw_days=1, fixing_index=fixing_index, advance_years=advance_years,
    )
    x_2dte = snap_corrected_dte_cumulative_variance(
        start_date=start_date, raw_days=2, fixing_index=fixing_index, advance_years=advance_years,
    )
    return min(x_1dte, x_2dte)


def build_advanced_pchip(
    points: list[TermStructurePoint], advance_years: float, daily_step_cumulative_variance: float,
) -> PchipInterpolator:
    """Advance each option's (tau, V) to its time-(t+dt) predicted value, then PCHIP-fit.

    Each option's total annualized variance V_i over [t, t+tau_i] decomposes as
        V_i * tau_i  =  daily_step_cumulative_variance  +  V_advanced_i * (tau_i - dt)
    where the caller supplies daily_step_cumulative_variance (the cumulative variance over
    the one-business-day step [t, t+dt]) from the chain via
    `daily_step_cumulative_variance_from_chain`.
    """
    raw_taus = np.array([point.time_to_expiry for point in points])
    raw_variances = np.array([point.total_variance for point in points])
    unique_taus, inverse_index = np.unique(raw_taus, return_inverse=True)
    summed_variances = np.bincount(inverse_index, weights=raw_variances)
    duplicate_counts = np.bincount(inverse_index)
    variances = summed_variances / duplicate_counts
    if len(unique_taus) < MIN_POINTS_FOR_INTERPOLATION:
        msg = f"Insufficient distinct tenors ({len(unique_taus)}) for advanced PCHIP fit"
        raise ValueError(msg)

    tau_first = float(unique_taus[0])
    if tau_first <= advance_years:
        msg = f"Shortest tenor {tau_first} <= advance {advance_years}; cannot advance"
        raise ValueError(msg)

    advanced_taus = unique_taus - advance_years
    advanced_cumulative_variance = variances * unique_taus - daily_step_cumulative_variance
    advanced_variances = advanced_cumulative_variance / advanced_taus
    if np.any(advanced_taus <= 0) or np.any(advanced_variances <= 0):
        msg = (
            f"Non-positive advanced (tau, V) for options at {len(unique_taus)} tenors "
            f"(tau_first={tau_first}, daily_step_cumulative_variance={daily_step_cumulative_variance})"
        )
        raise ValueError(msg)
    return PchipInterpolator(x=np.sqrt(advanced_taus), y=advanced_variances, extrapolate=True)
