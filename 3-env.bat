@echo off
REM =====================================================================
REM  SeeThrough-GGUF-Win - project venv terminal (Windows)
REM
REM  Opens a prompt that uses ./.venv python and the same PYTHONPATH /
REM  env vars the server passes to inference, so "pip install ..." or
REM  "python backend\selfcheck.py" behave exactly like inside the app.
REM
REM  ASCII-only on purpose: cmd.exe mis-parses UTF-8 Chinese in .bat files.
REM
REM  Usage:
REM    - from the settings page button ("open command prompt")
REM    - cmd /k 3-env.bat
REM    - double-click (window stays open)
REM =====================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run 1-setup.bat first.
    pause
    exit /b 1
)

set "VIRTUAL_ENV=%~dp0.venv"
set "PATH=%~dp0.venv\Scripts;%PATH%"
set "PYTHONPATH=%~dp0stm\common;%~dp0stm"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
REM Windows has no MSVC by default: torch.compile dequant must stay off
set "SEETHROUGH_COMPILE_DEQUANT=0"
REM layer fixes patch switch (drop / eye order); 0 = upstream behaviour
set "STG_LAYER_FIX=1"

title SeeThrough venv
prompt (stg) $P$G

echo.
echo  SeeThrough-GGUF-Win venv terminal
echo  ------------------------------------------------
echo  python   : %VIRTUAL_ENV%\Scripts\python.exe
echo  workdir  : %CD%
echo  NOTE     : everything you install here lands in .venv only.
echo             Delete .venv to undo everything.
echo  PACKAGES : this venv has no pip; use uv instead:
echo             uv pip install --python .venv\Scripts\python.exe PKG
echo.

python -c "import sys;print('  python', sys.version.split()[0], 'ready.')"
echo.

REM Double-click starts cmd with /c; keep the window open in that case.
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 cmd /k
