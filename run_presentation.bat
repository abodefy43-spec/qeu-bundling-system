@echo off
setlocal
cd /d "%~dp0"
echo.
echo Starting Flask presentation at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause
