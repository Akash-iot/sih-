#!/usr/bin/env python3
"""
Standalone ETHEREYE FastAPI Application
======================================

Simplified FastAPI server with ML endpoints that can run directly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

# Import our ML endpoints
from api.ml_endpoints import ml_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="ETHEREYE Analytics API",
    description="Blockchain Analytics and Intelligence Platform API with ML Services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML router
app.include_router(ml_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ETHEREYE Analytics API",
        "version": "1.0.0",
        "description": "Blockchain Analytics and Intelligence Platform",
        "ml_endpoints": {
            "health": "/api/v1/ml/health",
            "clustering": "/api/v1/ml/cluster",
            "anomaly_detection": "/api/v1/ml/anomaly-detection",
            "risk_assessment": "/api/v1/ml/risk-assessment",
            "text_analysis": "/api/v1/ml/text-analysis"
        },
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_services": "available"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )