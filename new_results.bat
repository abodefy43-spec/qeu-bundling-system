@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo QEU New Results - Phases 6, 7, 8, 9 only (fast)
echo ============================================================
echo.
echo Uses cached data from data/ - run run_pipeline.bat first if needed.
echo.
python src\run_new_results.py
echo.
echo Done. Check output/top_10_bundles.csv
pause
