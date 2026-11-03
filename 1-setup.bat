@echo off
REM =====================================================================
REM  SeeThrough-GGUF-Win installer (Windows)
REM  Requires: Python 3.12 (or uv). Git is NOT required.
REM
REM  Steps: bundled STM -> .venv -> PyTorch -> deps -> patches
REM         -> runtime components (parallel) -> self check
REM
REM  This file is ASCII-only on purpose. cmd.exe mis-parses UTF-8 Chinese
REM  inside .bat files (line offsets get corrupted, commands get skipped),
REM  so all Chinese messages are printed by the Python scripts instead.
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "WARNINGS="

echo ============================================================
echo   SeeThrough-GGUF-Win  -  installer
echo ============================================================

REM ---------------------------------------------------------------------
echo.
echo --- [1/7] Checking bundled STM code -----------------------------------
if not exist "stm\inference\scripts\inference_psd.py" (
    echo [ERROR] stm\ not found. Please clone the repository completely.
    echo         expected: stm\inference\scripts\inference_psd.py
    echo.
    pause
    exit /b 1
)
echo [OK] stm\ found

REM ---------------------------------------------------------------------
echo.
echo --- [2/7] Preparing virtual environment -------------------------------
if exist ".venv\Scripts\python.exe" (
    for /f "tokens=2" %%v in ('.venv\Scripts\python.exe -V 2^>^&1') do set "PYVER=%%v"
    echo [OK] reusing .venv ^(Python !PYVER!^)
) else (
    echo creating .venv ...
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv --python 3.12 .venv
    ) else (
        py -3.12 -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] failed to create .venv.
        echo         Install Python 3.12 from python.org, or install uv, then re-run.
        echo.
        pause
        exit /b 1
    )
    echo [OK] .venv created
)

set "PY=.venv\Scripts\python.exe"

REM ---------------------------------------------------------------------
echo.
echo --- [3/7] Installing PyTorch cu128 ------------------------------------
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements-torch.txt
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements-torch.txt
)
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch installation failed.
    echo         Check your network or proxy, then re-run this script.
    echo.
    pause
    exit /b 1
)
echo [OK] PyTorch cu128 installed

REM ---------------------------------------------------------------------
echo.
echo --- [4/7] Installing STM and backend dependencies ---------------------
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements.txt
) else (
    "%PY%" -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo [ERROR] dependency installation failed. See the pip output above.
    echo.
    pause
    exit /b 1
)
echo [OK] dependencies installed

REM ---------------------------------------------------------------------
echo.
echo --- [5/7] Verifying patches -------------------------------------------
"%PY%" backend\patches\apply_use_lama.py
if errorlevel 1 (
    echo [WARN] use_lama patch not applied
    set "WARNINGS=!WARNINGS! patch:use_lama"
)
"%PY%" backend\patches\apply_layer_fix.py
if errorlevel 1 (
    echo [WARN] layer_fix patch not applied
    set "WARNINGS=!WARNINGS! patch:layer_fix"
)

REM ---------------------------------------------------------------------
echo.
echo --- [6/7] Fetching runtime components ---------------------------------
REM components go into the project's components\ folder, so deleting the
REM project folder removes everything: models, outputs, components, .venv.
for /f "delims=" %%p in ('"%PY%" backend\components_paths.py --ensure') do set "%%p"
if defined HF_HOME echo components: %HF_HOME%
"%PY%" backend\netcheck.py
"%PY%" backend\download_models.py --components-only --jobs 0
if errorlevel 1 (
    echo [WARN] some components are not ready; the first run will retry them
    set "WARNINGS=!WARNINGS! components"
)

echo.
echo [NOTE] The two main GGUF models are NOT downloaded by this script.
echo        Put layerdiff3d-*.gguf and marigold3d-*.gguf under models\,
echo        or run: "%PY%" backend\download_models.py

REM ---------------------------------------------------------------------
echo.
echo --- [7/7] Environment self check --------------------------------------
"%PY%" backend\selfcheck.py
if errorlevel 1 (
    echo [WARN] self check reported problems, see the output above
    set "WARNINGS=!WARNINGS! selfcheck"
)

REM ---------------------------------------------------------------------
echo.
if defined WARNINGS (
    echo ============================================================
    echo   Finished with warnings:%WARNINGS%
    echo   The app can still start. Details are printed above.
    echo ============================================================
) else (
    echo ============================================================
    echo   All steps finished.
    echo ============================================================
)
echo   Next: double-click 2-launch.bat, then open http://localhost:7860
echo   Skip component download: set STG_SKIP_MODELS=1 before this script.
echo.

pause
endlocal
