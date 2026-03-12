@echo off
chcp 65001 >nul
cd /d "%~dp0.."
python -m qeu_bundling.cli serve --host 127.0.0.1 --port 5000
echo Opening http://127.0.0.1:5000
start "" http://127.0.0.1:5000
