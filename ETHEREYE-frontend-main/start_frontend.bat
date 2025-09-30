@echo off
echo ========================================
echo 🎨 STARTING ETHEREYE FRONTEND SERVER
echo ========================================

cd /d "%~dp0"
echo Current directory: %CD%

echo.
echo 📋 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Alternatively, you can open HTML files directly in browser.
    pause
    exit /b 1
)

echo.
echo 🔍 Checking frontend files...
if not exist "dashboard.html" (
    echo ❌ Frontend files not found!
    echo Make sure you're in the correct directory.
    pause
    exit /b 1
)

echo.
echo ✅ Starting ETHEREYE Frontend Server...
echo 🌐 Frontend will be available at: http://localhost:3000
echo 📊 Main Dashboard: http://localhost:3000/dashboard.html
echo 🔍 Fixed Traces: http://localhost:3000/traces_fixed.html
echo 🕸️ Spider Map: http://localhost:3000/spider-map.html
echo 📈 Analytics: http://localhost:3000/analytics.html
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python -m http.server 3000

pause
