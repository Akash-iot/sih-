#!/usr/bin/env python3
"""
ETHEREYE Web Scraping System Setup
This script sets up and runs the ETHEREYE blockchain analytics scraping system
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Please install and start Redis server")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    requirements_file = Path("backend/requirements.txt")
    
    if requirements_file.exists():
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                         check=True, capture_output=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    env_file = Path("backend/.env")
    env_example = Path("backend/.env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔧 Setting up environment file...")
        shutil.copy(env_example, env_file)
        print("✅ Environment file created from template")
        print("⚠️  Please edit backend/.env with your API keys")
        return True
    elif env_file.exists():
        print("✅ Environment file already exists")
        return True
    else:
        print("❌ Environment template file not found")
        return False

def setup_database():
    """Initialize database"""
    print("🗄️  Setting up database...")
    try:
        # Import here to avoid issues if dependencies aren't installed yet
        sys.path.append(str(Path("backend").absolute()))
        from models.database import init_database
        
        init_database()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def start_celery_worker():
    """Start Celery worker"""
    print("🔄 Starting Celery worker...")
    try:
        # Change to backend directory
        backend_dir = Path("backend").absolute()
        
        cmd = [
            sys.executable, "-m", "celery", "worker",
            "-A", "scheduler.scraping_scheduler:celery_app",
            "--loglevel=info",
            "--concurrency=4"
        ]
        
        subprocess.Popen(cmd, cwd=backend_dir)
        print("✅ Celery worker started")
        return True
    except Exception as e:
        print(f"❌ Failed to start Celery worker: {e}")
        return False

def start_celery_beat():
    """Start Celery beat scheduler"""
    print("⏰ Starting Celery beat scheduler...")
    try:
        backend_dir = Path("backend").absolute()
        
        cmd = [
            sys.executable, "-m", "celery", "beat",
            "-A", "scheduler.scraping_scheduler:celery_app",
            "--loglevel=info"
        ]
        
        subprocess.Popen(cmd, cwd=backend_dir)
        print("✅ Celery beat scheduler started")
        return True
    except Exception as e:
        print(f"❌ Failed to start Celery beat: {e}")
        return False

def start_api_server():
    """Start FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        backend_dir = Path("backend").absolute()
        
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        subprocess.Popen(cmd, cwd=backend_dir)
        print("✅ FastAPI server started on http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def open_frontend():
    """Open frontend in browser"""
    frontend_path = Path("ETHEREYE-frontend-main/index.html").absolute()
    if frontend_path.exists():
        import webbrowser
        webbrowser.open(f"file://{frontend_path}")
        print(f"🌐 Frontend opened: file://{frontend_path}")
    else:
        print("❌ Frontend file not found")

def show_status():
    """Show system status"""
    print("\n" + "="*60)
    print("🎯 ETHEREYE Web Scraping System Status")
    print("="*60)
    print("📡 API Server: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("💊 Health Check: http://localhost:8000/health")
    print("🔍 Frontend: Open index.html in browser")
    print("\n📝 Next Steps:")
    print("1. Add your API keys to backend/.env")
    print("2. Open frontend and check dashboard")
    print("3. Monitor logs for scraping activity")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="ETHEREYE Web Scraping System Setup")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't start services")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--start-services", action="store_true", help="Start all services")
    
    args = parser.parse_args()
    
    print("🔬 ETHEREYE - Blockchain Analytics Web Scraping System")
    print("="*60)
    
    # Check system requirements
    check_python_version()
    
    if not check_redis():
        print("\n💡 To install Redis:")
        print("   Windows: Download from https://redis.io/download")
        print("   macOS: brew install redis && brew services start redis")
        print("   Ubuntu: sudo apt-get install redis-server")
        return
    
    # Setup phase
    if not args.skip_deps:
        if not install_dependencies():
            return
    
    if not setup_environment():
        return
    
    if not setup_database():
        return
    
    if args.setup_only:
        print("✅ Setup completed successfully!")
        return
    
    # Start services
    if args.start_services or not args.setup_only:
        print("\n🚀 Starting services...")
        
        # Start background services
        start_celery_worker()
        start_celery_beat()
        start_api_server()
        
        # Wait a moment for services to start
        import time
        time.sleep(3)
        
        # Open frontend
        open_frontend()
        
        # Show status
        show_status()
        
        print("\n🎉 ETHEREYE is now running!")
        print("Press Ctrl+C to stop all services")
        
        try:
            # Keep script running
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")
            print("✅ ETHEREYE stopped")

if __name__ == "__main__":
    main()