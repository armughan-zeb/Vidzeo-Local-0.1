@echo off
title Create Deployment Zip for RDP/Data Center
color 0B

echo ========================================================
echo  Creating Vidzeo Local Portable Package
echo  (Excluding heavy files like venv, models, output)
echo ========================================================
echo.

set "ZIP_NAME=Vidzeo_Local_Deploy.zip"

:: Use PowerShell to zip specific files/folders
powershell -Command "Compress-Archive -Path 'server', 'public', 'fonts', 'tools', 'INSTALL_AND_RUN.bat', 'start.bat', 'QUICK_START.txt', 'README.md' -DestinationPath '%ZIP_NAME%' -Force"

echo.
echo [OK] Created %ZIP_NAME%
echo.
echo ========================================================
echo  INSTRUCTIONS FOR RDP / DATA CENTER:
echo ========================================================
echo.
echo 1. Copy "%ZIP_NAME%" to your RDP / New PC.
echo    (It's small because we excluded the heavy AI models)
echo.
echo 2. Extract it on the RDP.
echo.
echo 3. Double-click "INSTALL_AND_RUN.bat"
echo.
echo    The script will automatically:
echo    - Download Python (if missing)
echo    - Download FFmpeg
echo    - Download all AI models using the RDP's fast internet
echo.
pause
