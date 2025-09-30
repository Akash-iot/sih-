import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    ETHERSCAN_API_KEY: Optional[str] = os.getenv("ETHERSCAN_API_KEY")
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    INFURA_PROJECT_ID: Optional[str] = os.getenv("INFURA_PROJECT_ID")
    ALCHEMY_API_KEY: Optional[str] = os.getenv("ALCHEMY_API_KEY")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ethereye.db")
    
    # Redis (for caching and task queue)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Scraping Settings
    REQUEST_DELAY: float = 1.0  # Seconds between requests
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    USER_AGENT: str = "ETHEREYE-Analytics/1.0 (Blockchain Analytics Platform)"
    
    # Rate Limiting
    REQUESTS_PER_SECOND: int = 5
    REQUESTS_PER_MINUTE: int = 100
    REQUESTS_PER_HOUR: int = 1000
    
    # Data Sources URLs
    ETHERSCAN_BASE_URL: str = "https://api.etherscan.io/api"
    COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
    BLOCKCHAIN_INFO_URL: str = "https://blockchain.info"
    DEXGURU_API_URL: str = "https://api.dex.guru/v1"
    
    # Web3 RPC URLs
    ETHEREUM_RPC_URL: str = os.getenv(
        "ETHEREUM_RPC_URL", 
        f"https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}" if INFURA_PROJECT_ID else ""
    )
    
    # Scraping Intervals (in minutes)
    PRICE_SCRAPING_INTERVAL: int = 5
    TRANSACTION_SCRAPING_INTERVAL: int = 10
    ADDRESS_SCRAPING_INTERVAL: int = 15
    TOKEN_SCRAPING_INTERVAL: int = 30
    
    # Data Retention (in days)
    DATA_RETENTION_DAYS: int = 90
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/ethereye.log"
    
    # Selenium WebDriver Settings
    WEBDRIVER_HEADLESS: bool = True
    WEBDRIVER_TIMEOUT: int = 30
    WEBDRIVER_IMPLICIT_WAIT: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Blockchain Networks Configuration
SUPPORTED_NETWORKS = {
    "ethereum": {
        "chain_id": 1,
        "name": "Ethereum Mainnet",
        "rpc_url": settings.ETHEREUM_RPC_URL,
        "explorer_api": settings.ETHERSCAN_BASE_URL,
        "native_token": "ETH",
        "block_time": 12  # seconds
    },
    "polygon": {
        "chain_id": 137,
        "name": "Polygon",
        "rpc_url": "https://polygon-rpc.com",
        "explorer_api": "https://api.polygonscan.com/api",
        "native_token": "MATIC",
        "block_time": 2
    },
    "bsc": {
        "chain_id": 56,
        "name": "Binance Smart Chain",
        "rpc_url": "https://bsc-dataseed.binance.org",
        "explorer_api": "https://api.bscscan.com/api",
        "native_token": "BNB",
        "block_time": 3
    }
}

# Data Categories for Scraping
SCRAPING_CATEGORIES = {
    "transactions": {
        "priority": 1,
        "interval": settings.TRANSACTION_SCRAPING_INTERVAL,
        "endpoints": ["pending_txs", "latest_txs", "token_transfers"]
    },
    "addresses": {
        "priority": 2,
        "interval": settings.ADDRESS_SCRAPING_INTERVAL,
        "endpoints": ["balance", "transactions", "tokens"]
    },
    "tokens": {
        "priority": 3,
        "interval": settings.TOKEN_SCRAPING_INTERVAL,
        "endpoints": ["token_info", "holders", "transfers"]
    },
    "prices": {
        "priority": 4,
        "interval": settings.PRICE_SCRAPING_INTERVAL,
        "endpoints": ["current_price", "historical_price", "market_data"]
    },
    "defi": {
        "priority": 5,
        "interval": 60,  # 1 hour
        "endpoints": ["pools", "liquidity", "yields"]
    }
}