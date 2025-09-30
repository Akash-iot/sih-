@echo off
echo Starting ETHEREYE API Server...
echo ================================

cd /d "C:\Users\akash\Downloads\ETHEREYE-frontend-main\backend"

echo Starting FastAPI server on port 8001...
start "ETHEREYE API Server" python -m uvicorn simple_main:app --host 127.0.0.1 --port 8001 --reload

echo.
echo ✅ Server starting in background...
echo 📡 API will be available at: http://127.0.0.1:8001
echo 📊 API Documentation: http://127.0.0.1:8001/docs
echo 💊 Health Check: http://127.0.0.1:8001/health
echo.
echo Press any key to continue...
pause > nul

echo.
echo Opening demo page...
start "ETHEREYE Demo" "C:\Users\akash\Downloads\ETHEREYE-frontend-main\ETHEREYE-frontend-main\demo.html"

echo.
echo 🎉 ETHEREYE system is now running!
echo To stop the server, close the "ETHEREYE API Server" window