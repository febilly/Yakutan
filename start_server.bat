@echo off
rem Yakutan Remote Inference Server - one-click entry (ASCII-only to avoid codepage issues)
rem Closing this console window stops the server: all child processes share this console
rem and are terminated together by Windows.
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    for %%P in (python.exe) do set "PYTHON_EXE=%%~$PATH:P"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Please install Python 3.10+ or create .venv
    pause
    exit /b 1
)

rem Unified entry: launcher.py handles dependency/model diagnosis, port check and cleanup.
rem 0.0.0.0 = accept LAN connections (the service has NO authentication; trusted networks only).
"%PYTHON_EXE%" "%~dp0remote_inference_service\launcher.py" --host 0.0.0.0 %*

if errorlevel 1 (
    echo.
    echo Server stopped with exit code %ERRORLEVEL%.
    pause
)
