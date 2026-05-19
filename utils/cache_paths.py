"""Data cache locations and output paths.

The input cache layout the code assumes (out-of-scope to populate):

    $CACHE_ROOT/cache_log_swap/{ROOT}/{EXPIRY_YYYYMMDD}/{OBS_DATE_YYYYMMDD}.feather
        columns: LogSwapBid, LogSwapAsk, LogSwapMid (32401-point arrays)
    $CACHE_ROOT/cache_fwd/{ROOT}/{EXPIRY_YYYYMMDD}/{OBS_DATE_YYYYMMDD}.feather
        columns: FwdBid, FwdAsk (32401-point arrays)

$CACHE_ROOT defaults to ~/option_cache; override with the env var
VARIANCE_FACTORS_CACHE_ROOT.
"""

import datetime as dt
import os
from pathlib import Path


def cache_root() -> Path:
    """Root directory containing cache_log_swap/ and cache_fwd/."""
    override = os.environ.get("VARIANCE_FACTORS_CACHE_ROOT")
    if override:
        return Path(override)
    return Path.home() / "option_cache"


def cache_folder_log_swap() -> Path:
    """Directory containing per-root log-swap feathers."""
    return cache_root() / "cache_log_swap"


def cache_folder_fwd() -> Path:
    """Directory containing per-root forward feathers."""
    return cache_root() / "cache_fwd"


def log_swap_path(root: str, expiration: dt.date, observation_date: dt.date) -> Path:
    """Per-(root, exp, obs-date) log-swap feather path."""
    return (
        cache_folder_log_swap() / root
        / expiration.strftime("%Y%m%d")
        / f"{observation_date.strftime('%Y%m%d')}.feather"
    )


def fwd_path(root: str, expiration: dt.date, observation_date: dt.date) -> Path:
    """Per-(root, exp, obs-date) forward feather path."""
    return (
        cache_folder_fwd() / root
        / expiration.strftime("%Y%m%d")
        / f"{observation_date.strftime('%Y%m%d')}.feather"
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"


def to_image_path(name: str, subdir: str | None = None) -> Path:
    """Build the output PNG path for a plot, replacing whitespace and punctuation with underscores."""
    clean = name.replace(" ", "_").replace(".", "_").replace("|", "_").replace(":", "")
    folder = OUT_DIR / subdir if subdir else OUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{clean}.png"
