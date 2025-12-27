@echo off
title Vidzeo Local
color 0A

echo Starting Vidzeo Local...

:: Set paths
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%venv310"
set "FFMPEG_DIR=%APP_DIR%tools\ffmpeg"

:: Add FFmpeg to PATH
set "PATH=%FFMPEG_DIR%;%PATH%"

:: Check if installed
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo [!] Vidzeo Local not installed yet.
    echo [!] Running installer...
    echo.
    call "%APP_DIR%INSTALL_AND_RUN.bat"
    exit /b
)

:: Open browser
start http://localhost:5000

:: Start server
cd /d "%APP_DIR%server"
"%VENV_DIR%\Scripts\python.exe" app.py

pause
