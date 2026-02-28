@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo Usage Events Summary
echo ============================================================
echo.

set "CSV_PATH=%~1"
if "%CSV_PATH%"=="" set "CSV_PATH=d:\usage-events-2026-02-28.csv"

python src\summarize_usage.py --csv "%CSV_PATH%" --save output\usage_summary_latest.txt
echo.
pause
