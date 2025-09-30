@echo off
title ETHEREYE Platform - Complete Startup
color 0A

echo.
echo =======================================
echo    🚀 ETHEREYE Platform Startup
echo =======================================
echo.

echo 📋 Starting ETHEREYE blockchain analytics platform...
echo.

REM Check if ports are available
echo 🔍 Checking port availability...
netstat -an | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 8000 is already in use. Backend may already be running.
    echo    Continue anyway? [Y/N]
    set /p continue=
    if /i not "%continue%"=="Y" goto :eof
)

netstat -an | findstr :3000 >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 3000 is already in use. Frontend may already be running.
    echo    Continue anyway? [Y/N]
    set /p continue=
    if /i not "%continue%"=="Y" goto :eof
)

echo.
echo 🚀 Starting Backend API Server (Port 8000)...
echo =======================================

REM Start backend server in new window
cd /d "%~dp0..\backend"
start "ETHEREYE Backend Server" cmd /k "echo 🚀 ETHEREYE BACKEND SERVER & echo ========================= & echo Backend API: http://localhost:8000 & echo Docs: http://localhost:8000/docs & echo. & python -m uvicorn simple_main:app --host 0.0.0.0 --port 8000 --reload"

echo    ✅ Backend server starting...
echo    📡 API will be available at: http://localhost:8000
echo    📖 Documentation at: http://localhost:8000/docs

REM Wait for backend to start
echo.
echo ⏳ Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

REM Test backend health
echo 🔍 Testing backend health...
for /l %%i in (1,1,5) do (
    powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 3; if($response.status -eq 'healthy') { Write-Host '   ✅ Backend is healthy!'; exit 0 } } catch { Write-Host '   ⏳ Attempt %%i/5 - Backend still starting...' }"
    if %errorlevel% equ 0 goto backend_ready
    timeout /t 2 /nobreak >nul
)

echo    ⚠️  Backend health check failed, but continuing...

:backend_ready
echo.
echo 🎨 Starting Frontend Server (Port 3000)...
echo =======================================

REM Go back to frontend directory and start frontend
cd /d "%~dp0"
start "ETHEREYE Frontend Server" cmd /k "echo 🎨 ETHEREYE FRONTEND SERVER & echo ========================== & echo Frontend: http://localhost:3000 & echo Enhanced Traces: http://localhost:3000/traces.html & echo API Test Page: http://localhost:3000/test-api.html & echo. & python -m http.server 3000"

echo    ✅ Frontend server starting...
echo    🌐 Frontend available at: http://localhost:3000

REM Wait for frontend
echo.
echo ⏳ Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

REM Test frontend
echo 🔍 Testing frontend availability...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:3000' -TimeoutSec 5 -UseBasicParsing; if($response.StatusCode -eq 200) { Write-Host '   ✅ Frontend is accessible!' } } catch { Write-Host '   ❌ Frontend test failed' }"

echo.
echo =======================================
echo    🎉 ETHEREYE Platform Ready!
echo =======================================
echo.
echo 📊 Access Points:
echo    🏠 Homepage:        http://localhost:3000/index.html
echo    🕸️  Enhanced Traces: http://localhost:3000/traces.html
echo    🗺️  Spider Map:      http://localhost:3000/spider-map.html
echo    🧪 API Test Page:   http://localhost:3000/test-api.html
echo    📡 Backend API:     http://localhost:8000
echo    📚 API Docs:        http://localhost:8000/docs
echo.
echo 🎯 Quick Test:
echo    Try the Enhanced Traces page and enter: 0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e
echo    Or click "Demo" to see sample network visualization
echo.
echo ⚙️  Both servers are running in separate windows.
echo    Close those windows to stop the servers.
echo.
echo 🔧 Troubleshooting:
echo    - If API fails, check backend window for errors
echo    - Both servers auto-reload on file changes
echo    - Backend logs show in the backend terminal
echo.

REM Final health check
echo 🔍 Final system health check...
powershell -Command "
try {
    Write-Host '   Testing Backend...' -NoNewline
    $backend = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 3
    Write-Host ' ✅ OK' -ForegroundColor Green
    
    Write-Host '   Testing Frontend...' -NoNewline  
    $frontend = Invoke-WebRequest -Uri 'http://localhost:3000' -TimeoutSec 3 -UseBasicParsing
    Write-Host ' ✅ OK' -ForegroundColor Green
    
    Write-Host '   Testing Spider Map API...' -NoNewline
    $spider = Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/spider-map/network/0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e' -TimeoutSec 5
    Write-Host ' ✅ OK ($($spider.network.nodes.Count) nodes)' -ForegroundColor Green
    
    Write-Host ''
    Write-Host '🎉 All systems operational!' -ForegroundColor Green
    Write-Host ''
} catch {
    Write-Host '' 
    Write-Host '⚠️  Some components may still be starting...' -ForegroundColor Yellow
    Write-Host 'Give it another 30 seconds and try accessing the URLs above.' -ForegroundColor Yellow
    Write-Host ''
}"

echo Ready to explore your blockchain analytics platform! 🚀
pause
