"""Intraday time-to-expiry: overnight + market-hours split with AM-settled override.

A business day of variance is split between overnight (previous close to open) and the
market session (09:30-16:00 ET):

    elapsed = OVERNIGHT + MARKET_WEIGHT * (snapshot_hour - 9.5) / 6.5

The remaining fraction of today plus full future business days, divided by 252, is the
annualized time to expiry.  For AM-settled options (SPX SET, SET = Friday-open
calculation), the expiration day's market session is excluded; OVERNIGHT_AM_SETTLEMENT
captures the empirical fraction that the overnight + opening auction realizes.
"""

MARKET_OPEN_HOUR = 9.5
MARKET_CLOSE_HOUR = 16.0
MARKET_HOURS = MARKET_CLOSE_HOUR - MARKET_OPEN_HOUR
OVERNIGHT = 0.3
MARKET_WEIGHT = 1.0 - OVERNIGHT
OVERNIGHT_AM_SETTLEMENT = 0.5

AM_SETTLED_ROOTS: frozenset[str] = frozenset({"SPX", "NDX", "VIX", "RUT"})


def is_am_settled(root: str) -> bool:
    """Return True if the root settles at the SET (open-of-expiration-day calculation)."""
    return root in AM_SETTLED_ROOTS


def intraday_time_to_expiry(raw_days: int, timestamp_index: int, am_settled: bool) -> float:  # noqa: FBT001
    """Time to expiry in years.

    raw_days: business days between observation date and expiration (0 for 0DTE).
    timestamp_index: position in the 32401-point cache array (0 = 08:00 ET, 32400 = 17:00 ET).
    am_settled: True for AM-settled roots (SPX SET); False for PM-settled (SPXW etc.).
    """
    snapshot_hour = 8.0 + timestamp_index / 3600.0
    elapsed = OVERNIGHT + MARKET_WEIGHT * max(0.0, snapshot_hour - MARKET_OPEN_HOUR) / MARKET_HOURS
    fraction_remaining = max(0.0, 1.0 - elapsed)
    if am_settled and raw_days > 0:
        return (raw_days - 1 + OVERNIGHT_AM_SETTLEMENT + fraction_remaining) / 252.0
    return (raw_days + fraction_remaining) / 252.0
