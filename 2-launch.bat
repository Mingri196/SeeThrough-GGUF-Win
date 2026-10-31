@echo off
REM =====================================================================
REM  SeeThrough-GGUF-Win launcher (Windows)
REM  ASCII-only on purpose: cmd.exe mis-parses UTF-8 Chinese in .bat files.
REM =====================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run 1-setup.bat first.
    pause
    exit /b 1
)

echo Starting SeeThrough-GGUF-Win ...
echo URL: http://localhost:7860   ^(close this window to stop^)

start "" cmd /c "timeout /t 2 >nul && start http://localhost:7860"
".venv\Scripts\python.exe" backend\server.py
pause
endlocal
