@echo off
rem Yakutan Remote Inference - open Windows Firewall for LAN access (ASCII-only to avoid codepage issues)
rem Only TCP 18775 (WebSocket service) needs inbound access.
rem Port 18776 (llama-server) binds 127.0.0.1 only and must NOT be opened inbound.
rem The actual firewall rules live in allow_lan.ps1 next to this script; this bat only elevates it.
setlocal
echo Configuring Windows Firewall for Yakutan Remote Inference (port 18775)...
echo Requesting administrator privileges...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Start-Process powershell -Verb RunAs -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%~dp0allow_lan.ps1')"

echo.
echo [OK] Elevation request sent (please accept the UAC prompt and follow the elevated window).
pause
exit /b 0
