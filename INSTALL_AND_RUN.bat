@echo off
title Vidzeo Local - Installer and Launcher
color 0A
setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║          VIDZEO LOCAL - AI Video Generator                   ║
echo  ║               One-Click Auto-Installer                       ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Set paths
set "APP_DIR=%~dp0"
set "TOOLS_DIR=%APP_DIR%tools"
set "PYTHON_DIR=%TOOLS_DIR%\python"
set "FFMPEG_DIR=%TOOLS_DIR%\ffmpeg"
set "VENV_DIR=%APP_DIR%venv310"


:: Create tools directory
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"

:: ================================================================
:: STEP 1: Check/Download Python
:: ================================================================
echo [1/5] Checking Python...

:: First check if Python 3.10+ is installed system-wide
where python >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo       Found system Python: !PY_VER!
    set "PYTHON_EXE=python"
    goto :check_venv
)

:: Check for portable Python
if exist "%PYTHON_DIR%\python.exe" (
    echo       Found portable Python
    set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
    goto :check_venv
)

:: Download portable Python
echo       Python not found - downloading portable Python 3.10...
echo.

:: Download Python embeddable package
powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%TOOLS_DIR%\python.zip'}"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download Python. Check internet connection.
    pause
    exit /b 1
)

:: Extract Python
echo       Extracting Python...
powershell -Command "Expand-Archive -Path '%TOOLS_DIR%\python.zip' -DestinationPath '%PYTHON_DIR%' -Force"
del "%TOOLS_DIR%\python.zip"

:: Download get-pip.py and install pip
echo       Installing pip...
powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'}"

:: Modify python310._pth to enable pip
echo python310.zip> "%PYTHON_DIR%\python310._pth"
echo .>> "%PYTHON_DIR%\python310._pth"
echo import site>> "%PYTHON_DIR%\python310._pth"

"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
echo       [OK] Python installed
echo.

:check_venv
:: ================================================================
:: STEP 2: Check/Create Virtual Environment  
:: ================================================================
echo [2/5] Setting up virtual environment...

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo       Virtual environment exists
    goto :check_ffmpeg
)

echo       Creating virtual environment...
"%PYTHON_EXE%" -m venv "%VENV_DIR%" 2>nul
if %errorlevel% neq 0 (
    :: If venv fails (embedded Python), use pip directly
    echo       Using portable mode...
    set "VENV_DIR=%PYTHON_DIR%"
)
echo       [OK] Environment ready
echo.

:check_ffmpeg
:: ================================================================
:: STEP 3: Check/Download FFmpeg
:: ================================================================
echo [3/5] Checking FFmpeg...

:: Check if FFmpeg is in PATH
where ffmpeg >nul 2>&1
if %errorlevel% == 0 (
    echo       Found system FFmpeg
    goto :install_deps
)

:: Check portable FFmpeg
if exist "%FFMPEG_DIR%\ffmpeg.exe" (
    echo       Found portable FFmpeg
    set "PATH=%FFMPEG_DIR%;%PATH%"
    goto :install_deps
)

:: Download FFmpeg
echo       FFmpeg not found - downloading...
echo.

powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile '%TOOLS_DIR%\ffmpeg.zip'}"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download FFmpeg. Check internet connection.
    pause
    exit /b 1
)

echo       Extracting FFmpeg...
powershell -Command "Expand-Archive -Path '%TOOLS_DIR%\ffmpeg.zip' -DestinationPath '%TOOLS_DIR%' -Force"

:: Find and move ffmpeg binaries
for /d %%d in ("%TOOLS_DIR%\ffmpeg-*") do (
    if exist "%%d\bin\ffmpeg.exe" (
        mkdir "%FFMPEG_DIR%" 2>nul
        move "%%d\bin\*" "%FFMPEG_DIR%\" >nul
        rmdir /s /q "%%d"
    )
)
del "%TOOLS_DIR%\ffmpeg.zip"
set "PATH=%FFMPEG_DIR%;%PATH%"
echo       [OK] FFmpeg installed
echo.

:install_deps
:: ================================================================
:: STEP 4: Install Python Dependencies
:: ================================================================
echo [4/5] Installing dependencies (this may take a few minutes on first run)...

:: Activate venv if it exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
    set "PIP_CMD=pip"
) else (
    set "PIP_CMD=%PYTHON_DIR%\Scripts\pip.exe"
)

:: Check if dependencies already installed
"%VENV_DIR%\Scripts\python.exe" -c "import flask; import groq; import kokoro" 2>nul
if %errorlevel% == 0 (
    echo       Dependencies already installed
    goto :start_server
)

echo       Installing core packages...
%PIP_CMD% install --quiet --disable-pip-version-check flask flask-cors Pillow requests numpy soundfile pydub groq openai duckduckgo-search together

echo       Installing Kokoro TTS (this downloads ~500MB model)...
%PIP_CMD% install --quiet --disable-pip-version-check https://github.com/hexgrad/kokoro/archive/refs/heads/main.zip

echo       Installing Whisper for captions...
%PIP_CMD% install --quiet --disable-pip-version-check openai-whisper

echo       [OK] All dependencies installed
echo.

:start_server
:: ================================================================
:: STEP 5: Start Server
:: ================================================================
echo [5/5] Starting Vidzeo Local...
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║  Server starting at: http://localhost:5000                   ║
echo  ║  Browser will open automatically                             ║
echo  ║  Press Ctrl+C to stop the server                             ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Set FFmpeg path for the server
set "PATH=%FFMPEG_DIR%;%PATH%"

:: Open browser after 3 seconds
start /b cmd /c "timeout /t 3 >nul && start http://localhost:5000"

:: Start server
cd /d "%APP_DIR%server"
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" app.py
) else (
    "%PYTHON_EXE%" app.py
)

:: If server exits, show message
echo.
echo Server stopped. Press any key to exit...
pause >nul
