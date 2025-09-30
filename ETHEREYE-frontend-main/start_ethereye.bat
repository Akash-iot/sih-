@echo off
title ETHEREYE - Complete Platform Startup
color 0A

echo.
echo  ███████╗████████╗██╗  ██╗███████╗██████╗ ███████╗██╗   ██╗███████╗
echo  ██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔════╝╚██╗ ██╔╝██╔════╝
echo  █████╗     ██║   ███████║█████╗  ██████╔╝█████╗   ╚████╔╝ █████╗  
echo  ██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██╔══╝    ╚██╔╝  ██╔══╝  
echo  ███████╗   ██║   ██║  ██║███████╗██║  ██║███████╗   ██║   ███████╗
echo  ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝
echo.
echo                    🚀 Blockchain Analytics Platform
echo                    Complete Startup Script v1.0
echo.
echo ========================================================================

echo.
echo 🔧 STARTING ETHEREYE COMPLETE PLATFORM...
echo ========================================================================
echo.
echo This will start both:
echo   🔙 Backend Server  (API + ML Services) on http://localhost:8000
echo   🎨 Frontend Server (Web Interface)     on http://localhost:3000
echo.
echo ⚠️  Important: Keep both windows open while using ETHEREYE
echo.

pause

echo.
echo 📋 Pre-flight checks...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! 
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Python installation found
)

REM Check backend directory
if not exist "..\backend\simple_main.py" (
    echo ❌ Backend files not found!
    echo Make sure you're in the ETHEREYE frontend directory
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Backend files found
)

REM Check frontend files
if not exist "dashboard.html" (
    echo ❌ Frontend files not found!
    echo Make sure you're in the ETHEREYE frontend directory
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Frontend files found
)

echo.
echo ✅ All pre-flight checks passed!
echo.
echo 🚀 Starting servers in 3 seconds...
timeout /t 3 >nul

echo.
echo ========================================================================
echo 🔙 STARTING BACKEND SERVER...
echo ========================================================================
start "ETHEREYE Backend" cmd /c "start_backend.bat"

echo Waiting for backend to initialize...
timeout /t 5 >nul

echo.
echo ========================================================================  
echo 🎨 STARTING FRONTEND SERVER...
echo ========================================================================
start "ETHEREYE Frontend" cmd /c "start_frontend.bat"

echo.
echo ========================================================================
echo 🎉 ETHEREYE PLATFORM STARTED SUCCESSFULLY!
echo ========================================================================
echo.
echo 🌐 Access Points:
echo   • Main Dashboard:     http://localhost:3000/dashboard.html
echo   • Fixed Traces:       http://localhost:3000/traces_fixed.html  
echo   • Analytics:          http://localhost:3000/analytics.html
echo   • Spider Map:         http://localhost:3000/spider-map.html
echo   • API Documentation:  http://localhost:8000/docs
echo   • ML Endpoints:       http://localhost:8000/api/v1/ml/
echo.
echo 📋 Server Status:
echo   • Backend:  Running on port 8000 (API + ML Services)
echo   • Frontend: Running on port 3000 (Web Interface)
echo.
echo ⚠️  Keep this window and both server windows open while using ETHEREYE
echo ⚠️  Press Ctrl+C in server windows to stop them
echo.

timeout /t 10 >nul

echo 🌐 Opening main dashboard...
start http://localhost:3000/dashboard.html

echo.
echo 🎯 ETHEREYE is ready! Happy analyzing! 🚀
echo.
pause
