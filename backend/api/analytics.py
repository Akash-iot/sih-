from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime, timedelta
from ..models.database import get_db, Transaction, Block, PriceData, GasTracker

router = APIRouter()

@router.get("/overview")
async def get_analytics_overview(db: Session = Depends(get_db)):
    """Get analytics overview dashboard data"""
    
    # Get total transactions
    total_transactions = db.query(func.count(Transaction.id)).scalar() or 0
    
    # Get total blocks
    total_blocks = db.query(func.count(Block.id)).scalar() or 0
    
    # Get 24h transaction count
    yesterday = datetime.utcnow() - timedelta(days=1)
    transactions_24h = db.query(func.count(Transaction.id)).filter(
        Transaction.timestamp >= yesterday
    ).scalar() or 0
    
    # Get average gas price (last 24h)
    avg_gas = db.query(func.avg(GasTracker.standard_gas_price)).filter(
        GasTracker.timestamp >= yesterday
    ).scalar() or 0
    
    # Get latest ETH price
    latest_eth_price = db.query(PriceData).filter(
        PriceData.coin_id == "ethereum",
        PriceData.currency == "usd"
    ).order_by(PriceData.timestamp.desc()).first()
    
    eth_price = latest_eth_price.price if latest_eth_price else 0
    
    return {
        "overview": {
            "total_transactions": total_transactions,
            "total_blocks": total_blocks,
            "transactions_24h": transactions_24h,
            "avg_gas_price_gwei": float(avg_gas),
            "eth_price_usd": eth_price,
            "last_updated": datetime.utcnow()
        }
    }

@router.get("/gas-tracker")
async def get_gas_analytics(
    days: int = Query(7, description="Number of days", le=30),
    db: Session = Depends(get_db)
):
    """Get gas price analytics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    gas_data = db.query(GasTracker).filter(
        GasTracker.timestamp >= start_date
    ).order_by(GasTracker.timestamp).all()
    
    return {
        "gas_tracker": [
            {
                "timestamp": gas.timestamp,
                "safe_gas_price": gas.safe_gas_price,
                "standard_gas_price": gas.standard_gas_price,
                "fast_gas_price": gas.fast_gas_price,
                "suggested_base_fee": gas.suggested_base_fee
            }
            for gas in gas_data
        ],
        "period": f"{days} days"
    }

@router.get("/network-stats")
async def get_network_stats(
    days: int = Query(30, description="Number of days", le=365),
    db: Session = Depends(get_db)
):
    """Get network statistics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Daily transaction stats
    daily_stats = db.query(
        func.date(Transaction.timestamp).label('date'),
        func.count(Transaction.id).label('tx_count'),
        func.sum(Transaction.value_eth).label('volume'),
        func.sum(Transaction.tx_fee_eth).label('fees'),
        func.avg(Transaction.tx_fee_eth).label('avg_fee')
    ).filter(
        Transaction.timestamp >= start_date
    ).group_by(
        func.date(Transaction.timestamp)
    ).order_by(text('date')).all()
    
    return {
        "network_stats": [
            {
                "date": stat.date.isoformat(),
                "transaction_count": stat.tx_count,
                "total_volume_eth": float(stat.volume or 0),
                "total_fees_eth": float(stat.fees or 0),
                "avg_fee_eth": float(stat.avg_fee or 0)
            }
            for stat in daily_stats
        ],
        "period": f"{days} days"
    }