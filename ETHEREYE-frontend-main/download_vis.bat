@echo off
echo Downloading vis.js network library...

if not exist "js\vis" mkdir "js\vis"

echo Downloading vis-network.min.js...
powershell -Command "Invoke-WebRequest -Uri 'https://unpkg.com/vis-network/standalone/umd/vis-network.min.js' -OutFile 'js\vis\vis-network.min.js'"

echo Downloading vis-network.min.css...
powershell -Command "Invoke-WebRequest -Uri 'https://unpkg.com/vis-network/styles/vis-network.min.css' -OutFile 'js\vis\vis-network.min.css'"

echo ✅ vis.js network library downloaded successfully!
pause
