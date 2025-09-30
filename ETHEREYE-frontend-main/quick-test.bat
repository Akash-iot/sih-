@echo off
title ETHEREYE Quick Test
color 0B

echo.
echo ==============================
echo   🧪 ETHEREYE Quick Test  
echo ==============================
echo.

echo 🔍 Testing ETHEREYE Platform...
echo.

REM Test if servers are running
echo 📡 Checking Backend (Port 8000)...
netstat -an | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo    ✅ Backend server detected on port 8000
) else (
    echo    ❌ Backend server not detected on port 8000
    echo    💡 Start it with: cd ../backend ^&^& python -m uvicorn simple_main:app --reload --host 0.0.0.0 --port 8000
)

echo.
echo 🎨 Checking Frontend (Port 3000)...
netstat -an | findstr :3000 >nul
if %errorlevel% equ 0 (
    echo    ✅ Frontend server detected on port 3000
) else (
    echo    ❌ Frontend server not detected on port 3000
    echo    💡 Start it with: python -m http.server 3000
)

echo.
echo 📁 Checking Required Files...
if exist "js\api-client.js" (echo    ✅ js\api-client.js - Found) else (echo    ❌ js\api-client.js - Missing)
if exist "js\vis\vis-network.min.js" (echo    ✅ js\vis\vis-network.min.js - Found) else (echo    ❌ js\vis\vis-network.min.js - Missing)
if exist "traces.html" (echo    ✅ traces.html - Found) else (echo    ❌ traces.html - Missing)
if exist "test-api.html" (echo    ✅ test-api.html - Found) else (echo    ❌ test-api.html - Missing)

echo.
echo ==============================
echo   🎯 Quick Access Links
echo ==============================
echo.
echo 🏠 Homepage:         http://localhost:3000/index.html
echo 🕸️  Transaction Traces: http://localhost:3000/traces.html
echo 🧪 API Test Page:    http://localhost:3000/test-api.html
echo 📡 Backend API:      http://localhost:8000
echo 📚 API Docs:         http://localhost:8000/docs
echo 🕸️  Spider Map:       http://localhost:3000/spider-map.html
echo.

echo 🎮 Quick Test Steps:
echo    1. Open: http://localhost:3000/traces.html
echo    2. Enter: 0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e
echo    3. Click 'Trace' or 'Demo' button
echo    4. Enjoy the interactive visualization!
echo.

echo 💡 For detailed testing, run: powershell -ExecutionPolicy Bypass -File test-ethereye.ps1
echo.

pause
