@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo QEU Bundling Pipeline - Full Run
echo ============================================================
python src\run_pipeline.py
echo.
echo Done. Check output/top_10_bundles.csv
pause
