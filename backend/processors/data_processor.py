import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from loguru import logger

from ..models.database import (
    Transaction, TokenTransfer, Address, Block, GasTracker, 
    PriceData, CoinInfo, MarketData, ScrapingJob, get_db
)

class DataProcessor:
    """Process and store scraped blockchain data"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
    async def process_transactions(self, transactions: List[Dict]) -> int:
        """Process and store transaction data"""
        stored_count = 0
        
        for tx_data in transactions:
            try:
                # Check if transaction already exists
                existing_tx = self.db.query(Transaction).filter(
                    Transaction.hash == tx_data.get("hash")
                ).first()
                
                if existing_tx:
                    logger.debug(f"Transaction {tx_data.get('hash')} already exists, updating...")
                    # Update existing transaction
                    for key, value in tx_data.items():
                        if hasattr(existing_tx, key) and value is not None:
                            setattr(existing_tx, key, value)
                    existing_tx.updated_at = datetime.utcnow()
                else:
                    # Create new transaction
                    transaction = Transaction(
                        hash=tx_data.get("hash"),
                        block_number=tx_data.get("block_number"),
                        timestamp=tx_data.get("timestamp"),
                        from_address=tx_data.get("from_address"),
                        to_address=tx_data.get("to_address"),
                        value_wei=str(tx_data.get("value_wei", "0")),
                        value_eth=tx_data.get("value_eth", 0.0),
                        gas_price=str(tx_data.get("gas_price", "0")),
                        gas_used=tx_data.get("gas_used", 0),
                        gas_limit=tx_data.get("gas_limit", 0),
                        tx_fee_wei=str(tx_data.get("tx_fee_wei", "0")),
                        tx_fee_eth=tx_data.get("tx_fee_eth", 0.0),
                        is_error=tx_data.get("is_error", False),
                        transaction_index=tx_data.get("transaction_index", 0),
                        input_data=tx_data.get("input", ""),
                        method_id=tx_data.get("method_id", ""),
                        source=tx_data.get("source", "unknown")
                    )
                    
                    self.db.add(transaction)
                    stored_count += 1
                
                # Update address information
                await self._update_address_info(tx_data.get("from_address"), tx_data.get("timestamp"))
                await self._update_address_info(tx_data.get("to_address"), tx_data.get("timestamp"))
                
            except Exception as e:
                logger.error(f"Error processing transaction {tx_data.get('hash', 'unknown')}: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} transactions")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing transactions: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def process_token_transfers(self, transfers: List[Dict]) -> int:
        """Process and store token transfer data"""
        stored_count = 0
        
        for transfer_data in transfers:
            try:
                # Check if transfer already exists
                existing_transfer = self.db.query(TokenTransfer).filter(
                    TokenTransfer.hash == transfer_data.get("hash"),
                    TokenTransfer.contract_address == transfer_data.get("contract_address"),
                    TokenTransfer.transaction_index == transfer_data.get("transaction_index", 0)
                ).first()
                
                if existing_transfer:
                    logger.debug(f"Token transfer already exists, skipping...")
                    continue
                
                transfer = TokenTransfer(
                    hash=transfer_data.get("hash"),
                    block_number=transfer_data.get("block_number"),
                    timestamp=transfer_data.get("timestamp"),
                    from_address=transfer_data.get("from_address"),
                    to_address=transfer_data.get("to_address"),
                    contract_address=transfer_data.get("contract_address"),
                    value=str(transfer_data.get("value", "0")),
                    value_formatted=transfer_data.get("value_formatted", 0.0),
                    token_name=transfer_data.get("token_name", ""),
                    token_symbol=transfer_data.get("token_symbol", ""),
                    token_decimal=transfer_data.get("token_decimal", 18),
                    transaction_index=transfer_data.get("transaction_index", 0),
                    gas=transfer_data.get("gas", 0),
                    gas_price=str(transfer_data.get("gas_price", "0")),
                    gas_used=transfer_data.get("gas_used", 0),
                    source=transfer_data.get("source", "unknown")
                )
                
                self.db.add(transfer)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error processing token transfer: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} token transfers")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing token transfers: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def process_blocks(self, blocks: List[Dict]) -> int:
        """Process and store block data"""
        stored_count = 0
        
        for block_data in blocks:
            try:
                # Check if block already exists
                existing_block = self.db.query(Block).filter(
                    Block.block_number == block_data.get("block_number")
                ).first()
                
                if existing_block:
                    logger.debug(f"Block {block_data.get('block_number')} already exists, updating...")
                    # Update existing block
                    for key, value in block_data.items():
                        if hasattr(existing_block, key) and value is not None:
                            setattr(existing_block, key, value)
                    existing_block.updated_at = datetime.utcnow()
                else:
                    block = Block(
                        block_number=block_data.get("block_number"),
                        block_hash=block_data.get("block_hash"),
                        parent_hash=block_data.get("parent_hash"),
                        timestamp=block_data.get("timestamp"),
                        gas_limit=block_data.get("gas_limit", 0),
                        gas_used=block_data.get("gas_used", 0),
                        miner=block_data.get("miner"),
                        difficulty=str(block_data.get("difficulty", "0")),
                        total_difficulty=str(block_data.get("total_difficulty", "0")),
                        size=block_data.get("size", 0),
                        transaction_count=block_data.get("transaction_count", 0),
                        base_fee_per_gas=str(block_data.get("base_fee_per_gas", "0")),
                        source=block_data.get("source", "unknown")
                    )
                    
                    self.db.add(block)
                    stored_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing block {block_data.get('block_number', 'unknown')}: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} blocks")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing blocks: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def process_gas_data(self, gas_data_list: List[Dict]) -> int:
        """Process and store gas tracker data"""
        stored_count = 0
        
        for gas_data in gas_data_list:
            try:
                gas_tracker = GasTracker(
                    safe_gas_price=gas_data.get("safe_gas_price", 0),
                    standard_gas_price=gas_data.get("standard_gas_price", 0),
                    fast_gas_price=gas_data.get("fast_gas_price", 0),
                    suggested_base_fee=gas_data.get("suggested_base_fee", 0.0),
                    gas_used_ratio=gas_data.get("gas_used_ratio", ""),
                    timestamp=gas_data.get("timestamp", datetime.utcnow()),
                    source=gas_data.get("source", "unknown")
                )
                
                self.db.add(gas_tracker)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error processing gas data: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} gas records")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing gas data: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def process_price_data(self, price_data_list: List[Dict]) -> int:
        """Process and store price data"""
        stored_count = 0
        
        for price_data in price_data_list:
            try:
                price_record = PriceData(
                    coin_id=price_data.get("coin_id"),
                    currency=price_data.get("currency"),
                    price=price_data.get("price", 0.0),
                    market_cap=price_data.get("market_cap", 0.0),
                    volume_24h=price_data.get("volume_24h", 0.0),
                    change_24h=price_data.get("change_24h", 0.0),
                    timestamp=price_data.get("timestamp", datetime.utcnow()),
                    source=price_data.get("source", "unknown")
                )
                
                self.db.add(price_record)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error processing price data for {price_data.get('coin_id', 'unknown')}: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} price records")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing price data: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def process_coin_info(self, coin_info_list: List[Dict]) -> int:
        """Process and store coin information"""
        stored_count = 0
        
        for coin_data in coin_info_list:
            try:
                # Check if coin info already exists
                existing_coin = self.db.query(CoinInfo).filter(
                    CoinInfo.coin_id == coin_data.get("coin_id")
                ).first()
                
                if existing_coin:
                    # Update existing coin info
                    for key, value in coin_data.items():
                        if hasattr(existing_coin, key) and value is not None:
                            setattr(existing_coin, key, value)
                    existing_coin.updated_at = datetime.utcnow()
                else:
                    coin_info = CoinInfo(
                        coin_id=coin_data.get("coin_id"),
                        name=coin_data.get("name"),
                        symbol=coin_data.get("symbol"),
                        description=coin_data.get("description"),
                        homepage=coin_data.get("homepage"),
                        blockchain_site=coin_data.get("blockchain_site"),
                        genesis_date=coin_data.get("genesis_date"),
                        market_cap_rank=coin_data.get("market_cap_rank"),
                        current_price_usd=coin_data.get("current_price_usd"),
                        market_cap_usd=coin_data.get("market_cap_usd"),
                        total_volume_usd=coin_data.get("total_volume_usd"),
                        high_24h_usd=coin_data.get("high_24h_usd"),
                        low_24h_usd=coin_data.get("low_24h_usd"),
                        price_change_24h=coin_data.get("price_change_24h"),
                        price_change_percentage_24h=coin_data.get("price_change_percentage_24h"),
                        price_change_percentage_7d=coin_data.get("price_change_percentage_7d"),
                        price_change_percentage_30d=coin_data.get("price_change_percentage_30d"),
                        circulating_supply=coin_data.get("circulating_supply"),
                        total_supply=coin_data.get("total_supply"),
                        max_supply=coin_data.get("max_supply"),
                        ath_usd=coin_data.get("ath_usd"),
                        ath_date=coin_data.get("ath_date"),
                        atl_usd=coin_data.get("atl_usd"),
                        atl_date=coin_data.get("atl_date"),
                        last_updated=coin_data.get("last_updated"),
                        source=coin_data.get("source", "unknown")
                    )
                    
                    self.db.add(coin_info)
                    stored_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing coin info for {coin_data.get('coin_id', 'unknown')}: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully processed {stored_count} coin info records")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing coin info: {str(e)}")
            stored_count = 0
        
        return stored_count
    
    async def _update_address_info(self, address: str, timestamp: datetime) -> None:
        """Update address information"""
        if not address:
            return
        
        try:
            existing_address = self.db.query(Address).filter(
                Address.address == address
            ).first()
            
            if existing_address:
                # Update last seen timestamp
                if not existing_address.last_seen or timestamp > existing_address.last_seen:
                    existing_address.last_seen = timestamp
                if not existing_address.first_seen or timestamp < existing_address.first_seen:
                    existing_address.first_seen = timestamp
                existing_address.updated_at = datetime.utcnow()
            else:
                # Create new address record
                new_address = Address(
                    address=address,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                self.db.add(new_address)
                
        except Exception as e:
            logger.error(f"Error updating address info for {address}: {str(e)}")
    
    async def create_scraping_job(
        self, 
        scraper_name: str, 
        endpoint: str, 
        parameters: Dict
    ) -> int:
        """Create a new scraping job record"""
        try:
            job = ScrapingJob(
                scraper_name=scraper_name,
                endpoint=endpoint,
                parameters=parameters,
                status="pending"
            )
            
            self.db.add(job)
            self.db.commit()
            
            logger.info(f"Created scraping job {job.id} for {scraper_name}:{endpoint}")
            return job.id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating scraping job: {str(e)}")
            return None
    
    async def update_scraping_job(
        self, 
        job_id: int, 
        status: str, 
        records_scraped: int = 0,
        error_message: str = None
    ) -> bool:
        """Update scraping job status"""
        try:
            job = self.db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
            
            if job:
                job.status = status
                job.records_scraped = records_scraped
                
                if status == "running" and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    job.completed_at = datetime.utcnow()
                
                if error_message:
                    job.error_message = error_message
                
                self.db.commit()
                logger.info(f"Updated scraping job {job_id} status to {status}")
                return True
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating scraping job {job_id}: {str(e)}")
        
        return False
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleanup_stats = {}
        
        try:
            # Clean up old price data
            price_deleted = self.db.query(PriceData).filter(
                PriceData.timestamp < cutoff_date
            ).count()
            self.db.query(PriceData).filter(
                PriceData.timestamp < cutoff_date
            ).delete()
            cleanup_stats["price_data"] = price_deleted
            
            # Clean up old gas tracker data
            gas_deleted = self.db.query(GasTracker).filter(
                GasTracker.timestamp < cutoff_date
            ).count()
            self.db.query(GasTracker).filter(
                GasTracker.timestamp < cutoff_date
            ).delete()
            cleanup_stats["gas_tracker"] = gas_deleted
            
            # Clean up old scraping jobs
            jobs_deleted = self.db.query(ScrapingJob).filter(
                ScrapingJob.created_at < cutoff_date,
                ScrapingJob.status.in_(["completed", "failed"])
            ).count()
            self.db.query(ScrapingJob).filter(
                ScrapingJob.created_at < cutoff_date,
                ScrapingJob.status.in_(["completed", "failed"])
            ).delete()
            cleanup_stats["scraping_jobs"] = jobs_deleted
            
            self.db.commit()
            logger.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during cleanup: {str(e)}")
        
        return cleanup_stats