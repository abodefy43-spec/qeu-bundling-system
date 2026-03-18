@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

set "PY_CMD="
set "PY_ARGS="
if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=py"
    set "PY_ARGS=-3"
  ) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
  )
)

if not defined PY_CMD (
  echo Python not found. Install Python or create .venv first.
  pause
  exit /b 1
)

set "PORT=5000"
call :is_port_busy %PORT%
if "!PORT_BUSY!"=="1" (
  set "PORT=5001"
  call :is_port_busy %PORT%
  if "!PORT_BUSY!"=="1" (
    echo Ports 5000 and 5001 are busy. Stop other servers and try again.
    pause
    exit /b 1
  )
)

set "LAN_IP="
for /f "usebackq delims=" %%I in (`"%PY_CMD%" %PY_ARGS% -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2^>nul`) do (
  set "LAN_IP=%%I"
)
if not defined LAN_IP set "LAN_IP=YOUR-LAN-IP"
if not defined QEU_DASHBOARD_DEFAULT_PERSON_COUNT set "QEU_DASHBOARD_DEFAULT_PERSON_COUNT=1"

echo ============================================================
echo Network Share Mode
echo ============================================================
echo.
echo This starts the dashboard on this device and shares it on LAN.
echo Dashboard startup mode: QEU_DASHBOARD_DEFAULT_PERSON_COUNT=%QEU_DASHBOARD_DEFAULT_PERSON_COUNT%
echo Local URL: http://127.0.0.1:%PORT%
echo LAN URL:   http://%LAN_IP%:%PORT%
echo.

start "" cmd /c "ping -n 3 127.0.0.1 >nul && start \"\" http://127.0.0.1:%PORT%"
"%PY_CMD%" %PY_ARGS% -m qeu_bundling.cli serve --host 0.0.0.0 --port %PORT%
pause
exit /b 0

:is_port_busy
set "PORT_BUSY=0"
for /f "tokens=5" %%A in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do (
  set "PORT_BUSY=1"
)
exit /b 0
