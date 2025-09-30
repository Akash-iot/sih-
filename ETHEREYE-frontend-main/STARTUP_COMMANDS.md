# 🚀 ETHEREYE Server Startup Commands

## Quick Manual Commands

### Option 1: PowerShell Commands
```powershell
# Start Backend (in new window)
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "cd '../backend'; python -m uvicorn simple_main:app --reload; Read-Host 'Press Enter to close'"

# Start Frontend (in new window)  
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "python -m http.server 3000; Read-Host 'Press Enter to close'"
```

### Option 2: Command Prompt
```cmd
# Terminal 1 - Backend Server
cd ..\backend
python -m uvicorn simple_main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend Server (open new cmd window)
cd ETHEREYE-frontend-main
python -m http.server 3000
```

### Option 3: Using the Batch Files

**🎯 EASIEST METHOD:**
```cmd
# Double-click start_ethereye.bat for complete startup
start_ethereye.bat
```

**Individual Servers:**
```cmd
# Backend only
start_backend.bat

# Frontend only
start_frontend.bat
```

## 🌐 Access URLs After Startup

### Frontend (Web Interface) - Port 3000
- **Main Dashboard**: http://localhost:3000/dashboard.html
- **Fixed Traces**: http://localhost:3000/traces_fixed.html
- **Analytics**: http://localhost:3000/analytics.html
- **Spider Map**: http://localhost:3000/spider-map.html
- **Wallets**: http://localhost:3000/wallets.html
- **Explorer**: http://localhost:3000/explorer.html

### Backend (API) - Port 8000
- **API Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
- **ML Endpoints**: http://localhost:8000/api/v1/ml/
- **Live Data**: http://localhost:8000/api/v1/live/

## 🛠️ Troubleshooting

### Backend Issues
```cmd
# Check if port 8000 is in use
netstat -an | findstr :8000

# Kill process on port 8000 if needed
taskkill /f /im python.exe
```

### Frontend Issues
```cmd
# Check if port 3000 is in use
netstat -an | findstr :3000

# Alternative: Open HTML files directly
start dashboard.html
```

### Python Issues
```cmd
# Check Python version
python --version

# Install/upgrade dependencies
pip install --upgrade -r ../backend/requirements.txt
```

## 📋 System Requirements
- **Python 3.8+** (required for both servers)
- **pip** (for installing dependencies)
- **Modern web browser** (Chrome, Firefox, Edge)
- **Minimum 4GB RAM** (for ML services)
- **Ports 3000 & 8000** (must be available)