@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================================
echo Network Share Mode (Port 5000)
echo ============================================================
echo.

echo This will expose the dashboard on your LAN.
echo Share: http://YOUR-LAN-IP:5000
echo.
echo Tip: run "ipconfig" and use your IPv4 Address.
echo.
start "" http://127.0.0.1:5000
python -m qeu_bundling.cli serve --host 0.0.0.0 --port 5000
pause
