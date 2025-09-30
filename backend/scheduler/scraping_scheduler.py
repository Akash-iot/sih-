import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List
from celery import Celery
from loguru import logger

from ..config import settings, SCRAPING_CATEGORIES
from ..scrapers.etherscan_scraper import EtherscanScraper
from ..scrapers.coingecko_scraper import CoinGeckoScraper
from ..processors.data_processor import DataProcessor
from ..models.database import SessionLocal

# Initialize Celery
celery_app = Celery(
    'ethereye_scraper',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'ethereye_scraper.scrape_data': {'queue': 'scraping'},
        'ethereye_scraper.process_data': {'queue': 'processing'},
    },
    beat_schedule={
        # Schedule gas tracker every 5 minutes
        'scrape-gas-tracker': {
            'task': 'ethereye_scraper.scrape_gas_tracker',
            'schedule': 300.0,  # 5 minutes
        },
        # Schedule price data every 2 minutes
        'scrape-price-data': {
            'task': 'ethereye_scraper.scrape_price_data',
            'schedule': 120.0,  # 2 minutes
        },
        # Schedule latest blocks every 15 seconds
        'scrape-latest-blocks': {
            'task': 'ethereye_scraper.scrape_latest_blocks',
            'schedule': 15.0,  # 15 seconds
        },
        # Schedule trending coins every hour
        'scrape-trending-coins': {
            'task': 'ethereye_scraper.scrape_trending_coins',
            'schedule': 3600.0,  # 1 hour
        },
        # Cleanup old data daily at 2 AM
        'cleanup-old-data': {
            'task': 'ethereye_scraper.cleanup_old_data',
            'schedule': 86400.0,  # 24 hours
        },
    },
)

class ScrapingScheduler:
    """Manages scheduled scraping tasks"""
    
    def __init__(self):
        self.is_running = False
        self.tasks = {}
        
    def start(self):
        """Start the scheduler"""
        self.is_running = True
        logger.info("Starting scraping scheduler...")
        
        # Schedule periodic tasks
        self._schedule_tasks()
        
        # Run scheduler
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        schedule.clear()
        logger.info("Scraping scheduler stopped")
    
    def _schedule_tasks(self):
        """Schedule all scraping tasks"""
        
        # High frequency tasks
        schedule.every(15).seconds.do(self._trigger_high_frequency_scraping)
        
        # Medium frequency tasks  
        schedule.every(5).minutes.do(self._trigger_medium_frequency_scraping)
        
        # Low frequency tasks
        schedule.every().hour.do(self._trigger_low_frequency_scraping)
        
        # Daily cleanup
        schedule.every().day.at("02:00").do(self._trigger_daily_cleanup)
        
        logger.info("Scheduled all scraping tasks")
    
    def _trigger_high_frequency_scraping(self):
        """Trigger high frequency scraping tasks"""
        try:
            # Latest blocks and transactions
            scrape_latest_blocks.delay()
            logger.debug("Triggered high frequency scraping")
        except Exception as e:
            logger.error(f"Failed to trigger high frequency scraping: {e}")
    
    def _trigger_medium_frequency_scraping(self):
        """Trigger medium frequency scraping tasks"""
        try:
            # Gas tracker and prices
            scrape_gas_tracker.delay()
            scrape_price_data.delay()
            logger.debug("Triggered medium frequency scraping")
        except Exception as e:
            logger.error(f"Failed to trigger medium frequency scraping: {e}")
    
    def _trigger_low_frequency_scraping(self):
        """Trigger low frequency scraping tasks"""
        try:
            # Market data, trending coins
            scrape_market_data.delay()
            scrape_trending_coins.delay()
            logger.debug("Triggered low frequency scraping")
        except Exception as e:
            logger.error(f"Failed to trigger low frequency scraping: {e}")
    
    def _trigger_daily_cleanup(self):
        """Trigger daily cleanup tasks"""
        try:
            cleanup_old_data.delay()
            logger.info("Triggered daily cleanup")
        except Exception as e:
            logger.error(f"Failed to trigger daily cleanup: {e}")

# ===================
# CELERY TASKS
# ===================

@celery_app.task(bind=True, max_retries=3)
def scrape_gas_tracker(self):
    """Scrape current gas prices"""
    try:
        asyncio.run(_scrape_gas_tracker_async())
        logger.info("Gas tracker scraping completed")
    except Exception as e:
        logger.error(f"Gas tracker scraping failed: {e}")
        raise self.retry(countdown=60)

async def _scrape_gas_tracker_async():
    """Async gas tracker scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("etherscan", "gas_tracker", {})
        
        async with EtherscanScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            gas_data = await scraper.scrape("gas_tracker")
            records_count = await processor.process_gas_data(gas_data)
            
            await processor.update_scraping_job(job_id, "completed", records_count)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def scrape_price_data(self):
    """Scrape cryptocurrency prices"""
    try:
        asyncio.run(_scrape_price_data_async())
        logger.info("Price data scraping completed")
    except Exception as e:
        logger.error(f"Price data scraping failed: {e}")
        raise self.retry(countdown=120)

async def _scrape_price_data_async():
    """Async price data scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("coingecko", "simple_price", {
            "coin_ids": ["ethereum", "bitcoin", "cardano", "solana", "polygon"]
        })
        
        async with CoinGeckoScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            price_data = await scraper.scrape("simple_price", 
                coin_ids=["ethereum", "bitcoin", "cardano", "solana", "polygon"],
                vs_currencies=["usd"]
            )
            records_count = await processor.process_price_data(price_data)
            
            await processor.update_scraping_job(job_id, "completed", records_count)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def scrape_latest_blocks(self):
    """Scrape latest blocks and transactions"""
    try:
        asyncio.run(_scrape_latest_blocks_async())
        logger.debug("Latest blocks scraping completed")
    except Exception as e:
        logger.error(f"Latest blocks scraping failed: {e}")
        raise self.retry(countdown=30)

async def _scrape_latest_blocks_async():
    """Async latest blocks scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("etherscan", "latest_blocks", {"count": 5})
        
        async with EtherscanScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            # Scrape latest blocks
            blocks_data = await scraper.scrape("latest_blocks", count=5)
            blocks_count = await processor.process_blocks(blocks_data)
            
            await processor.update_scraping_job(job_id, "completed", blocks_count)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def scrape_market_data(self):
    """Scrape cryptocurrency market data"""
    try:
        asyncio.run(_scrape_market_data_async())
        logger.info("Market data scraping completed")
    except Exception as e:
        logger.error(f"Market data scraping failed: {e}")
        raise self.retry(countdown=300)

async def _scrape_market_data_async():
    """Async market data scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("coingecko", "market_data", {"per_page": 50})
        
        async with CoinGeckoScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            market_data = await scraper.scrape("market_data", per_page=50)
            # Process as price data for simplicity
            records_count = await processor.process_price_data(market_data)
            
            await processor.update_scraping_job(job_id, "completed", records_count)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def scrape_trending_coins(self):
    """Scrape trending cryptocurrency coins"""
    try:
        asyncio.run(_scrape_trending_coins_async())
        logger.info("Trending coins scraping completed")
    except Exception as e:
        logger.error(f"Trending coins scraping failed: {e}")
        raise self.retry(countdown=600)

async def _scrape_trending_coins_async():
    """Async trending coins scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("coingecko", "trending_coins", {})
        
        async with CoinGeckoScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            trending_data = await scraper.scrape("trending_coins")
            # Store trending coins as coin info
            records_count = len(trending_data) if trending_data else 0
            
            await processor.update_scraping_job(job_id, "completed", records_count)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

@celery_app.task(bind=True)
def cleanup_old_data(self):
    """Clean up old scraped data"""
    try:
        asyncio.run(_cleanup_old_data_async())
        logger.info("Data cleanup completed")
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")

async def _cleanup_old_data_async():
    """Async data cleanup"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        cleanup_stats = await processor.cleanup_old_data(settings.DATA_RETENTION_DAYS)
        logger.info(f"Cleanup stats: {cleanup_stats}")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def scrape_address_data(self, address: str):
    """Scrape data for a specific address"""
    try:
        asyncio.run(_scrape_address_data_async(address))
        logger.info(f"Address scraping completed for {address}")
    except Exception as e:
        logger.error(f"Address scraping failed for {address}: {e}")
        raise self.retry(countdown=60)

async def _scrape_address_data_async(address: str):
    """Async address data scraping"""
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        
        # Create scraping job
        job_id = await processor.create_scraping_job("etherscan", "account_transactions", {"address": address})
        
        async with EtherscanScraper() as scraper:
            await processor.update_scraping_job(job_id, "running")
            
            # Scrape account transactions
            transactions = await scraper.scrape("account_transactions", address=address, offset=100)
            tx_count = await processor.process_transactions(transactions)
            
            # Scrape token transfers
            token_transfers = await scraper.scrape("token_transfers", address=address, offset=100)
            transfer_count = await processor.process_token_transfers(token_transfers)
            
            total_records = tx_count + transfer_count
            await processor.update_scraping_job(job_id, "completed", total_records)
            
    except Exception as e:
        if 'job_id' in locals():
            await processor.update_scraping_job(job_id, "failed", 0, str(e))
        raise
    finally:
        db.close()

# ===================
# SCHEDULER MANAGEMENT
# ===================

def start_scheduler():
    """Start the scraping scheduler"""
    scheduler = ScrapingScheduler()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping scheduler...")
        scheduler.stop()

if __name__ == "__main__":
    start_scheduler()