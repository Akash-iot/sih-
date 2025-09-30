@echo off
echo ========================================
echo 🚀 STARTING ETHEREYE BACKEND SERVER
echo ========================================

cd /d "%~dp0\..\backend"
echo Current directory: %CD%

echo.
echo 📋 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo 🔍 Checking server files...
if not exist "simple_main.py" (
    echo ❌ simple_main.py not found!
    echo Make sure you're in the correct directory.
    pause
    exit /b 1
)

echo.
echo ✅ Starting ETHEREYE Backend Server...
echo 🌐 Backend will be available at: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🤖 ML Endpoints: http://localhost:8000/api/v1/ml/
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python -m uvicorn simple_main:app --reload --host 0.0.0.0 --port 8000

pause
