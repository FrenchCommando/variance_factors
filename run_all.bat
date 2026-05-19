@echo off
setlocal
set MPLBACKEND=Agg
set PY=.venv\Scripts\python

echo === [01-03] panel_visual ===
%PY% -m scripts.panel_visual || goto :error

echo === [04] advance_visual ===
%PY% -m scripts.advance_visual || goto :error

echo === [05-10] rolling_calibration (slow: ~25 min for 20/40/60 BD sweep) ===
%PY% -m scripts.rolling_calibration || goto :error

echo === [11-13] realised_innovations ===
%PY% -m scripts.realised_innovations || goto :error

echo === [14] run_calibration (full panel) ===
%PY% -m scripts.run_calibration || goto :error

echo === [15-16] diagnostics ===
%PY% -m scripts.diagnostics || goto :error

echo === [17-18] empirical_nu_diagnostic ===
%PY% -m scripts.empirical_nu_diagnostic || goto :error

echo === [19-21] predicted_residuals ===
%PY% -m scripts.predicted_residuals || goto :error

echo.
echo === done: 21 PNGs in out\ ===
exit /b 0

:error
echo.
echo *** failed at the step above with errorlevel %errorlevel% ***
exit /b %errorlevel%
