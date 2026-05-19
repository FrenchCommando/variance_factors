"""US market calendar: holidays + business-day arithmetic."""

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

holidays: frozenset[dt.date] = frozenset({
    dt.date(2023, 6, 19),    # Juneteenth
    dt.date(2023, 7, 4),     # Independence Day
    dt.date(2023, 9, 4),     # Labor Day
    dt.date(2023, 11, 23),   # Thanksgiving
    dt.date(2023, 12, 25),   # Christmas
    dt.date(2024, 1, 1),     # New Year
    dt.date(2024, 1, 15),    # MLK
    dt.date(2024, 2, 19),    # Presidents' Day
    dt.date(2024, 3, 29),    # Good Friday
    dt.date(2024, 5, 27),    # Memorial Day
    dt.date(2024, 6, 19),    # Juneteenth
    dt.date(2024, 7, 4),     # Independence Day
    dt.date(2024, 9, 2),     # Labor Day
    dt.date(2024, 11, 28),   # Thanksgiving
    dt.date(2024, 12, 25),   # Christmas
    dt.date(2025, 1, 1),     # New Year
    dt.date(2025, 1, 9),     # President Carter mourning
    dt.date(2025, 1, 20),    # MLK
    dt.date(2025, 2, 17),    # Presidents' Day
    dt.date(2025, 4, 18),    # Good Friday
    dt.date(2025, 5, 26),    # Memorial Day
    dt.date(2025, 6, 19),    # Juneteenth
    dt.date(2025, 7, 4),     # Independence Day
    dt.date(2025, 9, 1),     # Labor Day
    dt.date(2025, 11, 27),   # Thanksgiving
    dt.date(2025, 12, 25),   # Christmas
    dt.date(2026, 1, 1),     # New Year
    dt.date(2026, 1, 19),    # MLK
    dt.date(2026, 2, 16),    # Presidents' Day
    dt.date(2026, 4, 3),     # Good Friday
    dt.date(2026, 5, 25),    # Memorial Day
    dt.date(2026, 6, 19),    # Juneteenth
    dt.date(2026, 7, 3),     # Independence Day
    dt.date(2026, 9, 7),     # Labor Day
    dt.date(2026, 11, 26),   # Thanksgiving
    dt.date(2026, 12, 25),   # Christmas
    dt.date(2027, 1, 1),     # New Year
    dt.date(2027, 1, 18),    # MLK
    dt.date(2027, 2, 15),    # Presidents' Day
    dt.date(2027, 3, 26),    # Good Friday
    dt.date(2027, 5, 31),    # Memorial Day
    dt.date(2027, 6, 18),    # Juneteenth observed
    dt.date(2027, 7, 5),     # Independence Day observed
    dt.date(2027, 9, 6),     # Labor Day
    dt.date(2027, 11, 25),   # Thanksgiving
    dt.date(2027, 12, 24),   # Christmas observed
})


def is_business_day(date: dt.date) -> bool:
    """Return True if date is a NYSE trading day."""
    return date not in holidays and date.weekday() < 5


def plus_days(date: dt.date, n_days: int, increment: int = 1) -> dt.date:
    """Advance date by n_days business days; increment=-1 walks backwards."""
    if n_days < 0:
        msg = f"plus_days requires n_days >= 0, got {n_days}"
        raise ValueError(msg)
    current_date = date
    remaining = n_days
    while remaining > 0:
        current_date += dt.timedelta(days=increment)
        while not is_business_day(date=current_date):
            current_date += dt.timedelta(days=increment)
        remaining -= 1
    return current_date


def count_business_days(date_from: dt.date, date_to: dt.date) -> int:
    """Business days from date_from up to (not including) date_to."""
    current_date = date_from
    count = 0
    while current_date < date_to:
        current_date = plus_days(date=current_date, n_days=1)
        count += 1
    return count


def dates_iter(date_from: dt.date, date_to: dt.date) -> "Generator[dt.date, None, None]":
    """Yield each business day in [date_from, date_to]."""
    current_date = date_from
    while current_date <= date_to:
        if is_business_day(date=current_date):
            yield current_date
        current_date += dt.timedelta(days=1)
