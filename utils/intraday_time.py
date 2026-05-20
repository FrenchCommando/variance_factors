"""Intraday time-to-expiry: overnight + market-hours split with AM-settled override.

A business day of variance is split between overnight (previous close to open) and the
market session.  Regular sessions close at 16:00 ET (6.5 market hours); NYSE half-days
close at 13:00 ET (3.5 market hours) -- the obs-day session is shorter so a given
snapshot hour leaves less of the day's variance still to accrue.

    market_hours        = (early-close ? 13.0 : 16.0) - 9.5
    elapsed             = OVERNIGHT + MARKET_WEIGHT * (snapshot_hour - 9.5) / market_hours
    fraction_remaining  = max(0, 1 - elapsed)

The remaining fraction of today plus full future business days, divided by 252, is the
annualized time to expiry.  For AM-settled options (SPX SET, SET = Friday-open
calculation), the expiration day's market session is excluded; OVERNIGHT_AM_SETTLEMENT
captures the empirical fraction that the overnight + opening auction realizes.

`is_early_close` flags the *observation* day as a half-day.  Half-day handling on the
*expiration* day (a PM-settled option whose expiry happens to fall on a half-day) is not
modelled here -- the expiry day contributes a full day of variance regardless.
"""

MARKET_OPEN_HOUR = 9.5
MARKET_CLOSE_HOUR = 16.0
HALF_DAY_CLOSE_HOUR = 13.0
MARKET_HOURS = MARKET_CLOSE_HOUR - MARKET_OPEN_HOUR
HALF_DAY_MARKET_HOURS = HALF_DAY_CLOSE_HOUR - MARKET_OPEN_HOUR
OVERNIGHT = 0.3
MARKET_WEIGHT = 1.0 - OVERNIGHT
OVERNIGHT_AM_SETTLEMENT = 0.5

AM_SETTLED_ROOTS: frozenset[str] = frozenset({"SPX", "NDX", "VIX", "RUT"})


def is_am_settled(root: str) -> bool:
    """Return True if the root settles at the SET (open-of-expiration-day calculation)."""
    return root in AM_SETTLED_ROOTS


def intraday_time_to_expiry(  # noqa: FBT001
    raw_days: int, timestamp_index: int, am_settled: bool, is_early_close: bool,
) -> float:
    """Time to expiry in years.

    raw_days: business days between observation date and expiration (0 for 0DTE).
    timestamp_index: position in the 32401-point cache array (0 = 08:00 ET, 32400 = 17:00 ET).
    am_settled: True for AM-settled roots (SPX SET); False for PM-settled (SPXW etc.).
    is_early_close: True if the observation day is a NYSE half-day (13:00 ET close); the
        obs day's market session is then 3.5 hours instead of 6.5.
    """
    market_hours = HALF_DAY_MARKET_HOURS if is_early_close else MARKET_HOURS
    snapshot_hour = 8.0 + timestamp_index / 3600.0
    elapsed = OVERNIGHT + MARKET_WEIGHT * max(0.0, snapshot_hour - MARKET_OPEN_HOUR) / market_hours
    fraction_remaining = max(0.0, 1.0 - elapsed)
    if am_settled and raw_days > 0:
        return (raw_days - 1 + OVERNIGHT_AM_SETTLEMENT + fraction_remaining) / 252.0
    return (raw_days + fraction_remaining) / 252.0
