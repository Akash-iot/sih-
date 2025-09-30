import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import json

from ..config import settings

class BaseScraper(ABC):
    """Base class for all scrapers with common functionality"""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = self._create_rate_limiter()
        self.last_request_time = 0
        
        # Setup logging
        logger.add(
            f"logs/{self.name}_scraper.log",
            rotation="10 MB",
            level=settings.LOG_LEVEL
        )
        
    def _create_rate_limiter(self):
        """Create rate limiter based on settings"""
        @sleep_and_retry
        @limits(calls=settings.REQUESTS_PER_SECOND, period=1)
        def rate_limited_request():
            pass
        return rate_limited_request
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
    
    async def start_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': settings.USER_AGENT,
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def make_request(
        self, 
        url: str, 
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        method: str = 'GET'
    ) -> Dict:
        """Make HTTP request with retry logic and rate limiting"""
        
        # Rate limiting
        self.rate_limiter()
        
        # Ensure minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < settings.REQUEST_DELAY:
            await asyncio.sleep(settings.REQUEST_DELAY - time_since_last)
        
        try:
            if not self.session:
                await self.start_session()
            
            request_headers = {}
            if headers:
                request_headers.update(headers)
            
            logger.debug(f"Making {method} request to: {url}")
            
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers
            ) as response:
                self.last_request_time = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Successful response from {url}")
                    return data
                elif response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message="Rate limited"
                    )
                else:
                    response.raise_for_status()
                    
        except Exception as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise
    
    def validate_data(self, data: Dict, required_fields: List[str]) -> bool:
        """Validate scraped data has required fields"""
        for field in required_fields:
            if field not in data or data[field] is None:
                logger.warning(f"Missing required field: {field}")
                return False
        return True
    
    def clean_data(self, data: Dict) -> Dict:
        """Clean and normalize scraped data"""
        # Remove null values
        cleaned = {k: v for k, v in data.items() if v is not None}
        
        # Convert timestamps to datetime objects
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'time']
        for field in timestamp_fields:
            if field in cleaned:
                try:
                    if isinstance(cleaned[field], (int, float)):
                        cleaned[field] = datetime.fromtimestamp(cleaned[field])
                    elif isinstance(cleaned[field], str):
                        cleaned[field] = datetime.fromisoformat(cleaned[field].replace('Z', '+00:00'))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse timestamp {field}: {e}")
        
        return cleaned
    
    def format_address(self, address: str) -> str:
        """Format blockchain address to lowercase with 0x prefix"""
        if not address:
            return ""
        
        address = address.lower()
        if not address.startswith('0x'):
            address = '0x' + address
        
        return address
    
    def format_amount(self, amount: str, decimals: int = 18) -> float:
        """Format blockchain amount from wei/smallest unit to human readable"""
        try:
            if isinstance(amount, str):
                amount = int(amount)
            return amount / (10 ** decimals)
        except (ValueError, TypeError):
            logger.warning(f"Failed to format amount: {amount}")
            return 0.0
    
    async def batch_requests(self, urls: List[str], batch_size: int = 10) -> List[Dict]:
        """Make multiple requests in batches"""
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_tasks = [self.make_request(url) for url in batch]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch request failed: {result}")
                        results.append(None)
                    else:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.extend([None] * len(batch))
            
            # Small delay between batches
            if i + batch_size < len(urls):
                await asyncio.sleep(1)
        
        return results
    
    @abstractmethod
    async def scrape(self, **kwargs) -> List[Dict]:
        """Main scraping method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_supported_endpoints(self) -> List[str]:
        """Return list of supported endpoints for this scraper"""
        pass
    
    def get_scraper_info(self) -> Dict:
        """Get information about this scraper"""
        return {
            'name': self.name,
            'base_url': self.base_url,
            'supported_endpoints': self.get_supported_endpoints(),
            'rate_limit': {
                'requests_per_second': settings.REQUESTS_PER_SECOND,
                'requests_per_minute': settings.REQUESTS_PER_MINUTE,
                'requests_per_hour': settings.REQUESTS_PER_HOUR
            }
        }