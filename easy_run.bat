@echo off
setlocal
cd /d "%~dp0"

title QEU Bundling - Easy Run
echo ============================================================
echo QEU Bundling System - Easy Run
echo ============================================================
echo.
echo [1] Quick refresh + open dashboard (Phases 6-9)
echo [2] Full rebuild + open dashboard (Phases 1-9)
echo [3] Open dashboard only
echo [4] Exit
echo.
set /p choice=Choose an option (1-4): 

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto full
if "%choice%"=="3" goto dashboard
if "%choice%"=="4" goto done

echo Invalid option.
goto done

:quick
echo.
echo Running quick refresh (new results)...
call new_results.bat
if errorlevel 1 goto fail
goto dashboard

:full
echo.
echo Running full pipeline...
call run_pipeline.bat
if errorlevel 1 goto fail
goto dashboard

:dashboard
echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo Starting dashboard at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
goto done

:fail
echo.
echo Something failed. Check logs above and try again.

:done
echo.
pause
endlocal
