@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo QEU New Results - Phases 6, 7, 8, 9 + Presentation
echo ============================================================
echo.
echo Uses cached data from data/ - run run_pipeline.bat first if needed.
echo.
python src\run_new_results.py
if errorlevel 1 (
  echo New results failed. Check logs above.
  pause
  exit /b 1
)
echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)
echo.
echo Starting Flask presentation at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause
