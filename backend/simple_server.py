#!/usr/bin/env python3
"""
Simple test server for debugging ML endpoints
"""

import sys
import traceback

try:
    from fastapi import FastAPI
    from api.ml_endpoints import ml_router
    from datetime import datetime
    
    print("✅ All imports successful")
    
    # Create simple app
    app = FastAPI(title="ETHEREYE ML Test API")
    
    # Add ML routes
    app.include_router(ml_router, prefix="/api/v1")
    
    @app.get("/")
    def root():
        return {"message": "ETHEREYE ML API is running", "timestamp": datetime.now().isoformat()}
    
    print("✅ FastAPI app created successfully")
    
    if __name__ == "__main__":
        import uvicorn
        print("🚀 Starting server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("📍 Full traceback:")
    traceback.print_exc()
    sys.exit(1)