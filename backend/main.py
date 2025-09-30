from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from .config import settings
from .models.database import get_db, init_database
from .scrapers.etherscan_scraper import EtherscanScraper
from .scrapers.coingecko_scraper import CoinGeckoScraper
from .processors.data_processor import DataProcessor
from .api import transactions, addresses, blocks, prices, analytics
from .api.ml_endpoints import ml_router
from .api.spider_map import router as spider_router

# Initialize database on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    yield
    # Shutdown
    pass

# Create FastAPI app
app = FastAPI(
    title="ETHEREYE Analytics API",
    description="Blockchain Analytics and Intelligence Platform API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["transactions"])
app.include_router(addresses.router, prefix="/api/v1/addresses", tags=["addresses"])
app.include_router(blocks.router, prefix="/api/v1/blocks", tags=["blocks"])
app.include_router(prices.router, prefix="/api/v1/prices", tags=["prices"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(ml_router, prefix="/api/v1")
app.include_router(spider_router, prefix="/api/v1/spider-map", tags=["spider-map"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ETHEREYE Analytics API",
        "version": "1.0.0",
        "description": "Blockchain Analytics and Intelligence Platform",
        "endpoints": {
            "transactions": "/api/v1/transactions",
            "addresses": "/api/v1/addresses", 
            "blocks": "/api/v1/blocks",
            "prices": "/api/v1/prices",
            "analytics": "/api/v1/analytics",
            "scraping": "/api/v1/scraping",
            "ml_services": "/api/v1/ml"
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
        "database": "connected",
        "scrapers": {
            "etherscan": "ready",
            "coingecko": "ready"
        }
    }

# Scraping endpoints
@app.post("/api/v1/scraping/start")
async def start_scraping(
    background_tasks: BackgroundTasks,
    scraper: str = Query(..., description="Scraper name (etherscan, coingecko)"),
    endpoint: str = Query(..., description="Endpoint to scrape"),
    params: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    """Start a scraping job"""
    
    # Validate scraper
    if scraper not in ["etherscan", "coingecko"]:
        raise HTTPException(status_code=400, detail="Invalid scraper name")
    
    # Create data processor
    processor = DataProcessor(db)
    
    # Create scraping job
    job_id = await processor.create_scraping_job(scraper, endpoint, params or {})
    
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to create scraping job")
    
    # Start scraping in background
    background_tasks.add_task(execute_scraping_job, scraper, endpoint, params or {}, job_id)
    
    return {
        "message": "Scraping job started",
        "job_id": job_id,
        "scraper": scraper,
        "endpoint": endpoint,
        "status": "pending"
    }

@app.get("/api/v1/scraping/jobs")
async def get_scraping_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(10, description="Number of jobs to return"),
    db: Session = Depends(get_db)
):
    """Get scraping jobs"""
    from .models.database import ScrapingJob
    
    query = db.query(ScrapingJob)
    
    if status:
        query = query.filter(ScrapingJob.status == status)
    
    jobs = query.order_by(ScrapingJob.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": job.id,
            "scraper_name": job.scraper_name,
            "endpoint": job.endpoint,
            "parameters": job.parameters,
            "status": job.status,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "records_scraped": job.records_scraped,
            "error_message": job.error_message,
            "created_at": job.created_at
        }
        for job in jobs
    ]

@app.get("/api/v1/scraping/jobs/{job_id}")
async def get_scraping_job(job_id: int, db: Session = Depends(get_db)):
    """Get specific scraping job"""
    from .models.database import ScrapingJob
    
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Scraping job not found")
    
    return {
        "id": job.id,
        "scraper_name": job.scraper_name,
        "endpoint": job.endpoint,
        "parameters": job.parameters,
        "status": job.status,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "records_scraped": job.records_scraped,
        "error_message": job.error_message,
        "created_at": job.created_at
    }

@app.get("/api/v1/scrapers/info")
async def get_scrapers_info():
    """Get information about available scrapers"""
    scrapers_info = {}
    
    # Get Etherscan scraper info
    etherscan = EtherscanScraper()
    scrapers_info["etherscan"] = etherscan.get_scraper_info()
    
    # Get CoinGecko scraper info  
    coingecko = CoinGeckoScraper()
    scrapers_info["coingecko"] = coingecko.get_scraper_info()
    
    return scrapers_info

async def execute_scraping_job(scraper_name: str, endpoint: str, params: Dict, job_id: int):
    """Execute a scraping job in the background"""
    # Get database session
    from .models.database import SessionLocal
    db = SessionLocal()
    
    try:
        processor = DataProcessor(db)
        
        # Update job status to running
        await processor.update_scraping_job(job_id, "running")
        
        # Initialize scraper
        if scraper_name == "etherscan":
            scraper = EtherscanScraper()
        elif scraper_name == "coingecko":
            scraper = CoinGeckoScraper()
        else:
            raise ValueError(f"Unknown scraper: {scraper_name}")
        
        # Execute scraping
        async with scraper:
            scraped_data = await scraper.scrape(endpoint, **params)
        
        # Process and store data
        records_count = 0
        
        if endpoint in ["account_transactions", "latest_blocks"]:
            records_count = await processor.process_transactions(scraped_data)
        elif endpoint == "token_transfers":
            records_count = await processor.process_token_transfers(scraped_data)
        elif endpoint == "latest_blocks":
            records_count = await processor.process_blocks(scraped_data)
        elif endpoint == "gas_tracker":
            records_count = await processor.process_gas_data(scraped_data)
        elif endpoint in ["simple_price", "market_data"]:
            records_count = await processor.process_price_data(scraped_data)
        elif endpoint == "coin_info":
            records_count = await processor.process_coin_info(scraped_data)
        else:
            # Generic data storage
            records_count = len(scraped_data)
        
        # Update job status to completed
        await processor.update_scraping_job(job_id, "completed", records_count)
        
    except Exception as e:
        # Update job status to failed
        await processor.update_scraping_job(job_id, "failed", 0, str(e))
        
    finally:
        db.close()

@app.post("/api/v1/scraping/cleanup")
async def cleanup_old_data(
    days: int = Query(90, description="Days of data to keep"),
    db: Session = Depends(get_db)
):
    """Clean up old scraped data"""
    processor = DataProcessor(db)
    cleanup_stats = await processor.cleanup_old_data(days)
    
    return {
        "message": "Cleanup completed",
        "days_kept": days,
        "records_deleted": cleanup_stats
    }

# Real-time data endpoints
@app.get("/api/v1/live/gas")
async def get_live_gas_prices():
    """Get current gas prices"""
    scraper = EtherscanScraper()
    
    try:
        async with scraper:
            gas_data = await scraper.scrape("gas_tracker")
        
        if gas_data:
            return gas_data[0]
        else:
            raise HTTPException(status_code=503, detail="Failed to fetch gas prices")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.get("/api/v1/live/prices")
async def get_live_prices(
    coins: str = Query("ethereum,bitcoin", description="Comma-separated coin IDs"),
    currencies: str = Query("usd", description="Comma-separated currencies")
):
    """Get current cryptocurrency prices"""
    scraper = CoinGeckoScraper()
    
    try:
        coin_list = coins.split(",")
        currency_list = currencies.split(",")
        
        async with scraper:
            price_data = await scraper.scrape(
                "simple_price",
                coin_ids=coin_list,
                vs_currencies=currency_list
            )
        
        return price_data
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )