"""Constant-expiry SPXW forward returns and per-pair local vol from the panel.

Two per-pair signals consumed by the joint Bergomi+spot likelihood:

1. Constant-expiry log forward return (`daily_log_fwd_returns_for_panel_pairs`).
   For each panel pair (t_start, t_end), the underlying-move signal is the daily log return
   on a single SPXW forward expiring on t_end, observed at both t_start and t_end:
       spot_return  =  log F_{t_end}^{T = t_end}  -  log F_{t_start}^{T = t_end}
   Same contract at both dates, so the difference is a clean martingale spot increment.

2. Per-pair local vol (`local_vol_per_panel_pair`).  The annualized vol matching the pair's
   increment, derived from the front-tenor variance swap:
       sigma_S(t_start, t_end)  =  sqrt(2 * LogSwapMid_t_start^{t_end} / tau_years)
   Per-pair, time-varying.  Replaces the panel-level empirical std placeholder.

Both are forwards (front SPXW), not the index-cash spot -- funding/dividend carry is
factored out.
"""

import datetime as dt  # noqa: TC003

import numpy as np
from pyarrow import feather

from utils.cache_paths import fwd_path, log_swap_path
from utils.calendar_utils import count_business_days, is_half_day
from utils.intraday_time import intraday_time_to_expiry, is_am_settled

# Half-day cache snap: 12:55 ET on the 32401-point 08:00..17:00 grid.  Mirror of
# data_assembly.HALF_DAY_FIXING_INDEX, duplicated here to avoid the import cycle
# (data_assembly already imports from spot_data).
_HALF_DAY_FIXING_INDEX = 17700


def _fixing_for_obs_date(date: dt.date, base_index: int) -> tuple[int, bool]:
    """(effective_index, is_early_close) for an observation date -- see data_assembly.fixing_for_obs_date."""
    if is_half_day(date=date):
        return _HALF_DAY_FIXING_INDEX, True
    return base_index, False

FWD_PROXY_ROOT = "SPXW"


def read_fwd_mid_at_fixing(
    root: str, expiration: dt.date, observation_date: dt.date, fixing_index: int,
) -> float:
    """Read FwdMid = (FwdBid + FwdAsk) / 2 at fixing_index for one (root, exp, obs-date) cell."""
    table = feather.read_table(
        str(fwd_path(root=root, expiration=expiration, observation_date=observation_date)),
        columns=["FwdBid", "FwdAsk"],
    )
    bid = float(np.asarray(table.column("FwdBid"))[fixing_index])
    ask = float(np.asarray(table.column("FwdAsk"))[fixing_index])
    return 0.5 * (bid + ask)


def daily_log_fwd_returns_for_panel_pairs(
    dates: tuple[dt.date, ...], pair_end_indices: np.ndarray, fixing_index: int, root: str,
) -> np.ndarray:
    """Constant-EXPIRY daily log forward returns aligned with each panel pair.

    For pair i whose end index is e in `dates`:
        T_i        = dates[e]
        start_date = dates[e - 1]
        return     = log F^{T_i}_{T_i, fixing} - log F^{T_i}_{start_date, fixing}
    Same contract (expiry = T_i) at both reads -> martingale-clean spot increment.
    """
    n_pairs = len(pair_end_indices)
    returns = np.empty(n_pairs)
    for pair_index, end_index in enumerate(pair_end_indices):
        end_date = dates[int(end_index)]
        start_date = dates[int(end_index) - 1]
        start_effective_index, _ = _fixing_for_obs_date(date=start_date, base_index=fixing_index)
        end_effective_index, _ = _fixing_for_obs_date(date=end_date, base_index=fixing_index)
        fwd_at_start = read_fwd_mid_at_fixing(
            root=root, expiration=end_date, observation_date=start_date, fixing_index=start_effective_index,
        )
        fwd_at_end = read_fwd_mid_at_fixing(
            root=root, expiration=end_date, observation_date=end_date, fixing_index=end_effective_index,
        )
        returns[pair_index] = np.log(fwd_at_end) - np.log(fwd_at_start)
    return returns


def read_log_swap_mid_at_fixing(
    root: str, expiration: dt.date, observation_date: dt.date, fixing_index: int,
) -> float:
    """Read LogSwapMid at fixing_index for one (root, exp, obs-date) cell.  Strict on NaN / <= 0."""
    table = feather.read_table(
        str(log_swap_path(root=root, expiration=expiration, observation_date=observation_date)),
        columns=["LogSwapMid"],
    )
    log_swap = float(np.asarray(table.column("LogSwapMid"))[fixing_index])
    if not np.isfinite(log_swap) or log_swap <= 0:
        msg = f"LogSwapMid at fixing={fixing_index} for {root} exp={expiration} obs={observation_date} is {log_swap}"
        raise ValueError(msg)
    return log_swap


def local_vol_per_panel_pair(
    dates: tuple[dt.date, ...], pair_end_indices: np.ndarray, fixing_index: int, root: str,
) -> np.ndarray:
    """Annualized local vol sigma_S per panel pair, from the front-tenor variance swap.

    For pair (t_start, t_end), reads LogSwapMid from cache_log_swap/{root}/{t_end}/{t_start}.feather
    at fixing_index, computes annualized variance V = 2 * LogSwapMid / tau_years, returns sqrt(V).
    """
    n_pairs = len(pair_end_indices)
    sigma_s = np.empty(n_pairs)
    am_settled = is_am_settled(root=root)
    for pair_index, end_index in enumerate(pair_end_indices):
        end_date = dates[int(end_index)]
        start_date = dates[int(end_index) - 1]
        effective_index, is_early_close = _fixing_for_obs_date(date=start_date, base_index=fixing_index)
        raw_days = count_business_days(date_from=start_date, date_to=end_date)
        tau_years = intraday_time_to_expiry(
            raw_days=raw_days, timestamp_index=effective_index, am_settled=am_settled,
            is_early_close=is_early_close,
        )
        log_swap = read_log_swap_mid_at_fixing(
            root=root, expiration=end_date, observation_date=start_date, fixing_index=effective_index,
        )
        annualized_variance = 2.0 * log_swap / tau_years
        sigma_s[pair_index] = float(np.sqrt(annualized_variance))
    return sigma_s
