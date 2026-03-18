@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================================
echo Localhost Full Pipeline + Dashboard (Port 5000)
echo ============================================================
echo.
python -m qeu_bundling.cli run full
if errorlevel 1 (
  echo Full pipeline failed. Check logs above.
  pause
  exit /b 1
)
echo.
echo Opening http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python -m qeu_bundling.cli serve --host 127.0.0.1 --port 5000
pause
