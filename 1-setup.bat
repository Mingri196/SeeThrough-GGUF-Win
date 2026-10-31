@echo off
REM =====================================================================
REM  SeeThrough-GGUF-Win installer (Windows)
REM  Requires: Python 3.12 (or uv). Git is NOT required.
REM  Steps: check bundled STM -> create .venv -> install deps -> patch
REM         -> detect network and fetch runtime components -> self check
REM
REM  This file is ASCII-only on purpose. cmd.exe mis-parses UTF-8 Chinese
REM  inside .bat files (line offsets get corrupted, commands get skipped),
REM  so all Chinese messages are printed by the Python scripts instead.
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   SeeThrough-GGUF-Win installer
echo ============================================================

REM 1. bundled upstream code
if not exist "stm\inference\scripts\inference_psd.py" (
    echo [1/5] [ERROR] stm\ not found. Please clone the repository completely.
    pause
    exit /b 1
)
echo [1/5] bundled STM code found

REM 2. virtual environment
if exist ".venv\Scripts\python.exe" (
    for /f "tokens=2" %%v in ('.venv\Scripts\python.exe -V 2^>^&1') do set "PYVER=%%v"
    echo [2/5] reusing .venv ^(Python !PYVER!^)
) else (
    echo [2/5] creating .venv ...
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv --python 3.12 .venv
    ) else (
        py -3.12 -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] failed to create .venv. Install Python 3.12 ^(python.org^) or uv first.
        pause
        exit /b 1
    )
)

set "PY=.venv\Scripts\python.exe"

REM 3. dependencies (PyTorch from the cu128 index)
echo [3/5] installing PyTorch cu128 ...
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements-torch.txt
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements-torch.txt
)
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

echo [3/5] installing STM and backend dependencies ...
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements.txt
) else (
    "%PY%" -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo [ERROR] dependency installation failed. See the log above.
    pause
    exit /b 1
)

REM 4. use_lama patch (idempotent)
echo [4/5] verifying use_lama patch ...
"%PY%" backend\patches\apply_use_lama.py
echo [4/5] verifying layer_fix patch ...
"%PY%" backend\patches\apply_layer_fix.py

REM 5. network probe + runtime components + self check
echo [5/5] detecting download source, fetching runtime components ...
REM runtime components (HF / torch.hub) go into the project's components/ folder,
REM so deleting the folder removes everything: models, outputs, components, .venv.
for /f "delims=" %%p in ('"%PY%" backend\components_paths.py --ensure') do set "%%p"
if defined HF_HOME echo   components: %HF_HOME%
"%PY%" backend\netcheck.py
"%PY%" backend\download_models.py --components-only
if errorlevel 1 (
    echo [WARN] some components failed to download. The first run will retry them.
)

echo.
echo [NOTE] Main GGUF models are NOT fetched by this script.
echo        Put layerdiff3d-*.gguf and marigold3d-*.gguf under models\ yourself,
echo        or run: "%PY%" backend\download_models.py

echo.
echo [5/5] environment self check ...
"%PY%" backend\selfcheck.py
if errorlevel 1 (
    echo [WARN] self check reported problems, see the output above.
)

echo ============================================================
echo   Done. Double-click 2-launch.bat and open http://localhost:7860
echo   Skip component download: set STG_SKIP_MODELS=1 before running.
echo ============================================================
pause
endlocal
