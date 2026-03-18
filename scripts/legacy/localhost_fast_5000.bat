@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================================
echo Localhost Fast Refresh + Dashboard (Port 5000)
echo ============================================================
echo.
python -m qeu_bundling.cli run quick
if errorlevel 1 (
  echo Quick refresh failed quality gates. Using latest available outputs to launch dashboard anyway.
)
echo.
echo Starting dashboard server in this window...
set QEU_LOCAL_FAST_MODE=1
set QEU_DASHBOARD_DEFAULT_PERSON_COUNT=5
echo Browser will open automatically once the local dashboard is ready...
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "$deadline=(Get-Date).AddMinutes(2);" ^
  "while((Get-Date)-lt $deadline){" ^
  "  try{$resp=Invoke-WebRequest 'http://127.0.0.1:5000/healthz' -UseBasicParsing -TimeoutSec 5; if($resp.StatusCode -eq 200){Start-Process 'http://127.0.0.1:5000'; exit 0}}catch{}; Start-Sleep -Milliseconds 500" ^
  "}" ^
  "exit 1"
python -m qeu_bundling.cli serve --host 127.0.0.1 --port 5000
