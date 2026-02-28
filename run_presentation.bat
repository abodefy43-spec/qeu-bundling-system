@echo off
setlocal
cd /d "%~dp0"

echo Installing required dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  exit /b 1
)

echo Starting Flask presentation app at http://127.0.0.1:5000
python app.py

endlocal

