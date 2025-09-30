from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..models.database import get_db, Transaction
from ..scrapers.etherscan_scraper import EtherscanScraper

router = APIRouter()

@router.get("/")
async def get_transactions(
    limit: int = Query(50, description="Number of transactions to return", le=1000),
    offset: int = Query(0, description="Number of transactions to skip"),
    from_address: Optional[str] = Query(None, description="Filter by from address"),
    to_address: Optional[str] = Query(None, description="Filter by to address"),
    block_number: Optional[int] = Query(None, description="Filter by block number"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    db: Session = Depends(get_db)
):
    """Get transactions with optional filters"""
    query = db.query(Transaction)
    
    # Apply filters
    if from_address:
        query = query.filter(Transaction.from_address == from_address.lower())
    
    if to_address:
        query = query.filter(Transaction.to_address == to_address.lower())
    
    if block_number:
        query = query.filter(Transaction.block_number == block_number)
    
    if start_date:
        query = query.filter(Transaction.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Transaction.timestamp <= end_date)
    
    # Get total count
    total = query.count()
    
    # Get paginated results
    transactions = query.order_by(Transaction.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "transactions": [
            {
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_wei": tx.value_wei,
                "value_eth": tx.value_eth,
                "gas_price": tx.gas_price,
                "gas_used": tx.gas_used,
                "gas_limit": tx.gas_limit,
                "tx_fee_wei": tx.tx_fee_wei,
                "tx_fee_eth": tx.tx_fee_eth,
                "is_error": tx.is_error,
                "transaction_index": tx.transaction_index,
                "method_id": tx.method_id,
                "source": tx.source
            }
            for tx in transactions
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    }

@router.get("/{tx_hash}")
async def get_transaction(
    tx_hash: str,
    db: Session = Depends(get_db)
):
    """Get specific transaction by hash"""
    transaction = db.query(Transaction).filter(Transaction.hash == tx_hash.lower()).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {
        "hash": transaction.hash,
        "block_number": transaction.block_number,
        "timestamp": transaction.timestamp,
        "from_address": transaction.from_address,
        "to_address": transaction.to_address,
        "value_wei": transaction.value_wei,
        "value_eth": transaction.value_eth,
        "gas_price": transaction.gas_price,
        "gas_used": transaction.gas_used,
        "gas_limit": transaction.gas_limit,
        "tx_fee_wei": transaction.tx_fee_wei,
        "tx_fee_eth": transaction.tx_fee_eth,
        "is_error": transaction.is_error,
        "transaction_index": transaction.transaction_index,
        "input_data": transaction.input_data,
        "method_id": transaction.method_id,
        "source": transaction.source,
        "created_at": transaction.created_at,
        "updated_at": transaction.updated_at
    }

@router.get("/live/latest")
async def get_latest_transactions(
    count: int = Query(10, description="Number of latest transactions", le=50)
):
    """Get latest transactions from Etherscan"""
    scraper = EtherscanScraper()
    
    try:
        async with scraper:
            # Get latest blocks and their transactions
            blocks = await scraper.scrape("latest_blocks", count=5)
            
        # Extract transactions from blocks
        latest_txs = []
        for block in blocks[:count]:
            if 'transactions' in block:
                latest_txs.extend(block['transactions'][:2])  # Get 2 txs per block
        
        return {
            "transactions": latest_txs[:count],
            "timestamp": datetime.utcnow(),
            "source": "etherscan"
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch latest transactions: {str(e)}")

@router.get("/stats/daily")
async def get_daily_transaction_stats(
    days: int = Query(30, description="Number of days", le=365),
    db: Session = Depends(get_db)
):
    """Get daily transaction statistics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Query for daily stats
    from sqlalchemy import func, text
    
    daily_stats = db.query(
        func.date(Transaction.timestamp).label('date'),
        func.count(Transaction.id).label('transaction_count'),
        func.sum(Transaction.value_eth).label('total_volume_eth'),
        func.avg(Transaction.tx_fee_eth).label('avg_fee_eth'),
        func.sum(Transaction.tx_fee_eth).label('total_fees_eth')
    ).filter(
        Transaction.timestamp >= start_date
    ).group_by(
        func.date(Transaction.timestamp)
    ).order_by(text('date')).all()
    
    return {
        "stats": [
            {
                "date": stat.date.isoformat(),
                "transaction_count": stat.transaction_count,
                "total_volume_eth": float(stat.total_volume_eth or 0),
                "avg_fee_eth": float(stat.avg_fee_eth or 0),
                "total_fees_eth": float(stat.total_fees_eth or 0)
            }
            for stat in daily_stats
        ],
        "period": f"{days} days",
        "start_date": start_date.date().isoformat()
    }

@router.get("/search")
async def search_transactions(
    q: str = Query(..., description="Search query (address, hash, or block number)"),
    db: Session = Depends(get_db)
):
    """Search transactions by various criteria"""
    results = {"transactions": [], "total": 0}
    
    # Search by transaction hash
    if q.startswith("0x") and len(q) == 66:
        tx = db.query(Transaction).filter(Transaction.hash == q.lower()).first()
        if tx:
            results["transactions"].append({
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_eth": tx.value_eth,
                "type": "transaction"
            })
            results["total"] = 1
    
    # Search by address
    elif q.startswith("0x") and len(q) == 42:
        address = q.lower()
        txs = db.query(Transaction).filter(
            (Transaction.from_address == address) | (Transaction.to_address == address)
        ).order_by(Transaction.timestamp.desc()).limit(20).all()
        
        results["transactions"] = [
            {
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_eth": tx.value_eth,
                "type": "transaction"
            }
            for tx in txs
        ]
        results["total"] = len(txs)
    
    # Search by block number
    elif q.isdigit():
        block_number = int(q)
        txs = db.query(Transaction).filter(
            Transaction.block_number == block_number
        ).order_by(Transaction.transaction_index).all()
        
        results["transactions"] = [
            {
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_eth": tx.value_eth,
                "type": "transaction"
            }
            for tx in txs
        ]
        results["total"] = len(txs)
    
    return results