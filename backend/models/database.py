from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

from ..config import settings

Base = declarative_base()

class Transaction(Base):
    """Ethereum transaction data"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(String(66), unique=True, index=True)  # 0x + 64 hex chars
    block_number = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    from_address = Column(String(42), index=True)  # 0x + 40 hex chars
    to_address = Column(String(42), index=True)
    value_wei = Column(String(100))  # Store as string to handle large numbers
    value_eth = Column(Float)
    gas_price = Column(String(100))
    gas_used = Column(Integer)
    gas_limit = Column(Integer)
    tx_fee_wei = Column(String(100))
    tx_fee_eth = Column(Float)
    is_error = Column(Boolean, default=False)
    transaction_index = Column(Integer)
    input_data = Column(Text)
    method_id = Column(String(10))
    source = Column(String(50), default="etherscan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_transactions_from_timestamp', from_address, timestamp),
        Index('idx_transactions_to_timestamp', to_address, timestamp),
        Index('idx_transactions_block_index', block_number, transaction_index),
    )

class TokenTransfer(Base):
    """ERC-20 token transfer data"""
    __tablename__ = "token_transfers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(String(66), index=True)
    block_number = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    from_address = Column(String(42), index=True)
    to_address = Column(String(42), index=True)
    contract_address = Column(String(42), index=True)
    value = Column(String(100))  # Raw value
    value_formatted = Column(Float)  # Human readable value
    token_name = Column(String(100))
    token_symbol = Column(String(20), index=True)
    token_decimal = Column(Integer)
    transaction_index = Column(Integer)
    gas = Column(Integer)
    gas_price = Column(String(100))
    gas_used = Column(Integer)
    source = Column(String(50), default="etherscan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_token_transfers_contract_timestamp', contract_address, timestamp),
        Index('idx_token_transfers_from_token', from_address, token_symbol),
        Index('idx_token_transfers_to_token', to_address, token_symbol),
    )

class Address(Base):
    """Ethereum address data"""
    __tablename__ = "addresses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), unique=True, index=True)
    balance_wei = Column(String(100))
    balance_eth = Column(Float)
    transaction_count = Column(Integer, default=0)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    is_contract = Column(Boolean, default=False)
    contract_name = Column(String(100))
    labels = Column(JSON)  # Store address labels/tags as JSON
    source = Column(String(50), default="etherscan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Block(Base):
    """Ethereum block data"""
    __tablename__ = "blocks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_number = Column(Integer, unique=True, index=True)
    block_hash = Column(String(66), unique=True, index=True)
    parent_hash = Column(String(66))
    timestamp = Column(DateTime, index=True)
    gas_limit = Column(Integer)
    gas_used = Column(Integer)
    miner = Column(String(42), index=True)
    difficulty = Column(String(100))
    total_difficulty = Column(String(100))
    size = Column(Integer)
    transaction_count = Column(Integer)
    base_fee_per_gas = Column(String(100))
    source = Column(String(50), default="etherscan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GasTracker(Base):
    """Gas price tracking data"""
    __tablename__ = "gas_tracker"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    safe_gas_price = Column(Integer)  # in Gwei
    standard_gas_price = Column(Integer)  # in Gwei
    fast_gas_price = Column(Integer)  # in Gwei
    suggested_base_fee = Column(Float)
    gas_used_ratio = Column(String(200))  # Store as comma-separated values
    timestamp = Column(DateTime, index=True)
    source = Column(String(50), default="etherscan")
    created_at = Column(DateTime, default=datetime.utcnow)

class PriceData(Base):
    """Cryptocurrency price data"""
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(50), index=True)
    currency = Column(String(10), index=True)
    price = Column(Float)
    market_cap = Column(Float)
    volume_24h = Column(Float)
    change_24h = Column(Float)
    timestamp = Column(DateTime, index=True)
    source = Column(String(50), default="coingecko")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_price_data_coin_currency', coin_id, currency, timestamp),
    )

class CoinInfo(Base):
    """Cryptocurrency information"""
    __tablename__ = "coin_info"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    symbol = Column(String(20), index=True)
    description = Column(Text)
    homepage = Column(String(500))
    blockchain_site = Column(JSON)  # Store as JSON array
    genesis_date = Column(String(20))
    market_cap_rank = Column(Integer)
    current_price_usd = Column(Float)
    market_cap_usd = Column(Float)
    total_volume_usd = Column(Float)
    high_24h_usd = Column(Float)
    low_24h_usd = Column(Float)
    price_change_24h = Column(Float)
    price_change_percentage_24h = Column(Float)
    price_change_percentage_7d = Column(Float)
    price_change_percentage_30d = Column(Float)
    circulating_supply = Column(Float)
    total_supply = Column(Float)
    max_supply = Column(Float)
    ath_usd = Column(Float)
    ath_date = Column(String(50))
    atl_usd = Column(Float)
    atl_date = Column(String(50))
    last_updated = Column(String(50))
    source = Column(String(50), default="coingecko")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MarketData(Base):
    """Market data for cryptocurrencies"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(50), index=True)
    name = Column(String(100))
    symbol = Column(String(20), index=True)
    market_cap_rank = Column(Integer)
    current_price = Column(Float)
    market_cap = Column(Float)
    total_volume = Column(Float)
    high_24h = Column(Float)
    low_24h = Column(Float)
    price_change_24h = Column(Float)
    price_change_percentage_24h = Column(Float)
    price_change_percentage_7d = Column(Float)
    price_change_percentage_30d = Column(Float)
    market_cap_change_24h = Column(Float)
    market_cap_change_percentage_24h = Column(Float)
    circulating_supply = Column(Float)
    total_supply = Column(Float)
    max_supply = Column(Float)
    ath = Column(Float)
    ath_change_percentage = Column(Float)
    ath_date = Column(String(50))
    atl = Column(Float)
    atl_change_percentage = Column(Float)
    atl_date = Column(String(50))
    last_updated = Column(String(50))
    vs_currency = Column(String(10))
    timestamp = Column(DateTime, index=True)
    source = Column(String(50), default="coingecko")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_market_data_symbol_timestamp', symbol, timestamp),
        Index('idx_market_data_rank_timestamp', market_cap_rank, timestamp),
    )

class ScrapingJob(Base):
    """Track scraping jobs and their status"""
    __tablename__ = "scraping_jobs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scraper_name = Column(String(50), index=True)
    endpoint = Column(String(100), index=True)
    parameters = Column(JSON)  # Store job parameters as JSON
    status = Column(String(20), index=True)  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    records_scraped = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_scraping_jobs_scraper_status', scraper_name, status),
    )

# Database connection and session management
def create_database_engine():
    """Create database engine"""
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,  # Set to True for SQL query logging
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine

def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def get_session_maker(engine):
    """Get session maker"""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global database setup
engine = create_database_engine()
SessionLocal = get_session_maker(engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with tables"""
    create_tables(engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_database()