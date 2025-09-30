#!/usr/bin/env python3
"""
ETHEREYE Demo Starter - Simplified version without Redis
This starts the FastAPI server with basic scraping capabilities
"""
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met"""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings requests aiohttp loguru python-dotenv")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting ETHEREYE API Server...")
    
    # Add the backend directory to Python path
    backend_dir = Path("backend").absolute()
    
    try:
        # Start the FastAPI server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "simple_main:app",  # We'll create this simplified version
            "--host", "127.0.0.1",
            "--port", "8001",
            "--reload"
        ]
        
        process = subprocess.Popen(
            cmd, 
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ FastAPI server started on http://127.0.0.1:8001")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def open_frontend():
    """Open the frontend in browser"""
    frontend_path = Path("ETHEREYE-frontend-main/dashboard.html").absolute()
    if frontend_path.exists():
        webbrowser.open(f"file:///{frontend_path}")
        print(f"🌐 Frontend opened: {frontend_path}")
    else:
        frontend_path = Path("ETHEREYE-frontend-main/index.html").absolute()
        if frontend_path.exists():
            webbrowser.open(f"file:///{frontend_path}")
            print(f"🌐 Frontend opened: {frontend_path}")
        else:
            print("❌ Frontend files not found")

def show_demo_info():
    """Show demo information"""
    print("\n" + "="*60)
    print("🎯 ETHEREYE Demo - Blockchain Analytics Platform")
    print("="*60)
    print("📡 API Server: http://127.0.0.1:8001")
    print("📊 API Documentation: http://127.0.0.1:8001/docs")
    print("💊 Health Check: http://127.0.0.1:8001/health")
    print("\n📝 Demo Features:")
    print("• Live gas price tracking")
    print("• Cryptocurrency price monitoring") 
    print("• Transaction data scraping")
    print("• Real-time dashboard updates")
    print("\n⚡ Available Endpoints:")
    print("• GET  /api/v1/live/gas - Current gas prices")
    print("• GET  /api/v1/live/prices - Live crypto prices")
    print("• GET  /health - System health")
    print("• GET  /docs - Interactive API documentation")
    print("="*60)

def main():
    print("🔬 ETHEREYE - Blockchain Analytics Demo")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Start API server
    server_process = start_api_server()
    if not server_process:
        return
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Open frontend
    open_frontend()
    
    # Show demo information
    show_demo_info()
    
    print("\n🎉 ETHEREYE Demo is running!")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Keep the server running
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()