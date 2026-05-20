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


# Calibration root and min-tenor filter, both env-overridable so the same scripts can run
# either SPX or SPXW panels without code edits.  SPXW has daily expirations -- the 7 BD floor
# is a SPX-only fix to drop the front-week vol risk premium that doesn't lie on the long-tenor
# forward-variance term structure.  For SPXW, drop only literal 0/1 BD expirations.
ROOT = os.environ.get("VARIANCE_FACTORS_ROOT", "SPX")
_DEFAULT_MIN_RAW_DAYS = {"SPX": 7, "SPXW": 2}
MIN_RAW_DAYS = int(os.environ.get("VARIANCE_FACTORS_MIN_RAW_DAYS", _DEFAULT_MIN_RAW_DAYS.get(ROOT, 2)))

# Tenor grid: SPX reaches 378 BD via its LEAPS chain; SPXW typically goes ~210 BD out so
# the long end of the benchmark grid is unreachable and we use a truncated 5-endpoint grid.
_DEFAULT_TENOR_DAYS: dict[str, tuple[int, ...]] = {
    "SPX": (21, 42, 63, 126, 189, 252, 378),
    "SPXW": (21, 42, 63, 126, 189),
}
PANEL_TENOR_DAYS = _DEFAULT_TENOR_DAYS.get(ROOT, _DEFAULT_TENOR_DAYS["SPX"])


def run_subdir(name: str) -> Path:
    """Output sub-directory namespaced by ROOT so SPX and SPXW results coexist."""
    return OUT_DIR / ROOT / name


def to_image_path(name: str) -> Path:
    """Build the output PNG path, nested under `out/{ROOT}/` alongside this root's feathers."""
    clean = name.replace(" ", "_").replace(".", "_").replace("|", "_").replace(":", "")
    folder = OUT_DIR / ROOT
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{clean}.png"
