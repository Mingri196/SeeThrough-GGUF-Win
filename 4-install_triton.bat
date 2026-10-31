@echo off
REM =====================================================================
REM  triton-windows installer (Windows)
REM  ASCII-only on purpose: cmd.exe mis-parses UTF-8 Chinese in .bat files.
REM  All interaction and messages are printed by backend\install_triton.py
REM  Pass through arguments, e.g.:
REM      4-install_triton.bat --yes
REM      4-install_triton.bat --uninstall
REM      4-install_triton.bat --channel 2
REM =====================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run 1-setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" backend\install_triton.py %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo Exit code: %RC%
pause
exit /b %RC%
