# variance_factors

Bergomi 2-factor + spot calibration on SPX, on rolling windows.  Showcase
of the methodology and the kind of results it produces; not a turn-key
pipeline a new user can reproduce from scratch (data setup is not
documented).

## What it does

Joint Gaussian MLE on `(spot_return, log_xi_increments)` per panel pair
under 2-factor Bergomi with spot-vol correlations.  Sliding 20 / 40 / 60
BD windows produce a parameter time series; full-panel single fit
produces a global parameter set.  Diagnostics compare model-implied vs
empirical vol-of-V and term-structure decay at the cumulative-V
(variance-swap-rate) observable.

See `NOTES.md` for the model, the load-bearing modelling choices, and
empirical findings.

## Data

The code consumes per-`(root, expiry, observation-date)` feather files
in two layouts:

```
$CACHE_ROOT/cache_log_swap/{ROOT}/{EXPIRY_YYYYMMDD}/{OBS_DATE_YYYYMMDD}.feather
    columns: LogSwapBid, LogSwapAsk, LogSwapMid (32401-point arrays, 08:00..17:00 ET,
             values populated at 60-second offsets, daily fixing read at index 28500 = 15:55 ET)

$CACHE_ROOT/cache_fwd/{ROOT}/{EXPIRY_YYYYMMDD}/{OBS_DATE_YYYYMMDD}.feather
    columns: FwdBid, FwdAsk (32401-point arrays, daily fixing at the same index)
```

`$CACHE_ROOT` defaults to `~/option_cache`.  Override via the
`VARIANCE_FACTORS_CACHE_ROOT` env var if your cache lives elsewhere.
Generation of these files is out of scope here -- the assumption is
that the cache already exists.

## Layout

```
utils/         pure libraries (no main, no side effects beyond loading)
scripts/       runnable mains -- one PNG-producing job each
out/           generated outputs (feather gitignored, PNGs tracked)
```

Invoke scripts as modules from the repo root so the `utils.` and
`scripts.` import paths resolve:

```cmd
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

REM Run every script in dependency order (~25 min for the rolling sweep)
run_all.bat

REM Or run individually:

REM Rolling fit (20/40/60 BD) + parameter / sigma_R time series PNGs (#05-10)
.venv\Scripts\python -m scripts.rolling_calibration

REM Realised innovations under each rolling window's matching params (#11-13)
.venv\Scripts\python -m scripts.realised_innovations

REM Full-panel single fit + realised innovations PNG (#14)
.venv\Scripts\python -m scripts.run_calibration

REM V-observable diagnostics on the 60bd rolling fit (#15-18)
.venv\Scripts\python -m scripts.diagnostics
.venv\Scripts\python -m scripts.empirical_nu_diagnostic
.venv\Scripts\python -m scripts.predicted_residuals

REM Panel-data sanity checks (data only, no fit needed) (#01-04)
.venv\Scripts\python -m scripts.panel_visual
.venv\Scripts\python -m scripts.advance_visual
```

Outputs are namespaced by ROOT so SPX and SPXW results coexist:

```
out/
  {ROOT}/                              (e.g. SPX/, SPXW/)
    *.png                              (tracked)
    rolling_{N}bd/
      params_timeseries.feather        (gitignored)
      realised_innovations.feather     (gitignored)
    full_panel/
      params.feather                   (gitignored)
      realised_innovations.feather     (gitignored)
```

ROOT comes from `VARIANCE_FACTORS_ROOT` (default `SPX`).  PNG filenames keep
their `_{ROOT}` suffix so the file alone tells you which root produced it
even after a copy out of its folder.

## PNGs

Each script writes one or more PNGs to `out/`, prefixed with a stable
number so individual charts can be referenced by index.

| #   | Script                       | What                                                          |
|----:|------------------------------|---------------------------------------------------------------|
| 01  | `panel_visual.py`            | per-date term structure of vol                                |
| 02  | `panel_visual.py`            | per-date cumulative variance V*tau                            |
| 03  | `panel_visual.py`            | strip forward variance time series (one line per strip tenor) |
| 04  | `advance_visual.py`          | three-panel V advance correction diagnostic                   |
| 05  | `rolling_calibration.py`     | rolling parameter time series, 20 BD window                   |
| 06  | `rolling_calibration.py`     | rolling sigma_R per strip, 20 BD window                       |
| 07  | `rolling_calibration.py`     | rolling parameter time series, 40 BD window                   |
| 08  | `rolling_calibration.py`     | rolling sigma_R per strip, 40 BD window                       |
| 09  | `rolling_calibration.py`     | rolling parameter time series, 60 BD window                   |
| 10  | `rolling_calibration.py`     | rolling sigma_R per strip, 60 BD window                       |
| 11  | `realised_innovations.py`    | realised innovations + cumulative paths, 20 BD rolling        |
| 12  | `realised_innovations.py`    | realised innovations + cumulative paths, 40 BD rolling        |
| 13  | `realised_innovations.py`    | realised innovations + cumulative paths, 60 BD rolling        |
| 14  | `run_calibration.py`         | realised innovations under the global full-panel params       |
| 15  | `diagnostics.py`             | per-endpoint V-reconstruction residuals + variance decomposition |
| 16  | `diagnostics.py`             | per-endpoint V residual ACF with Bartlett bands               |
| 17  | `empirical_nu_diagnostic.py` | vol-of-V at the 63 BD endpoint, empirical vs model-implied    |
| 18  | `empirical_nu_diagnostic.py` | term-structure decay alpha, empirical vs model-implied        |
| 19  | `predicted_residuals.py`     | IS/OOS predicted-V residuals, calibration D at 25% quantile   |
| 20  | `predicted_residuals.py`     | IS/OOS predicted-V residuals, calibration D at 50% quantile   |
| 21  | `predicted_residuals.py`     | IS/OOS predicted-V residuals, calibration D at 75% quantile   |

Dependencies: scripts #11-21 all read `params_timeseries.feather`, so
`rolling_calibration` (#05-10) must run first.  `panel_visual` and
`advance_visual` (#01-04) need only the data cache.

## Modules

`utils/`:
- `bergomi_two_factor.py` -- model dataclass + observation matrices (xi and V)
- `bergomi_likelihood.py` -- joint Gaussian NLL with PSD barrier + GLS shock estimator
- `data_assembly.py` -- panel build: log_xi at constant tenors + Bergomi-advance increments
- `spot_data.py` -- per-pair forward returns and sigma_S from front varswap
- `calendar_utils.py`, `intraday_time.py`, `cache_paths.py` -- calendar, time accrual, paths

`scripts/`:
- `rolling_calibration.py` -- rolling MLE driver, cold-restart on PSD-barrier freeze
- `run_calibration.py` -- full-panel single fit
- `realised_innovations.py` -- GLS-infer (Z_X, Z_Y) per pair
- `diagnostics.py` -- V-endpoint reconstruction + variance decomposition + ACF
- `empirical_nu_diagnostic.py` -- vol-of-V + term-structure decay alpha
- `predicted_residuals.py` -- IS/OOS predicted-V residuals at three calibration dates
- `advance_visual.py` -- per-tenor advance correction sanity check
- `panel_visual.py` -- raw term-structure + cumulative-variance + strip-time-series sanity check
