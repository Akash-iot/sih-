from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..models.database import get_db, PriceData, CoinInfo
from ..scrapers.coingecko_scraper import CoinGeckoScraper

router = APIRouter()

@router.get("/")
async def get_prices(
    coin_ids: Optional[str] = Query(None, description="Comma-separated coin IDs"),
    currencies: str = Query("usd", description="Comma-separated currencies"),
    limit: int = Query(100, description="Number of records to return", le=1000),
    db: Session = Depends(get_db)
):
    """Get current prices for cryptocurrencies"""
    query = db.query(PriceData).order_by(PriceData.timestamp.desc())
    
    if coin_ids:
        coin_list = [coin.strip() for coin in coin_ids.split(",")]
        query = query.filter(PriceData.coin_id.in_(coin_list))
    
    prices = query.limit(limit).all()
    
    return {
        "prices": [
            {
                "coin_id": price.coin_id,
                "currency": price.currency,
                "price": price.price,
                "market_cap": price.market_cap,
                "volume_24h": price.volume_24h,
                "change_24h": price.change_24h,
                "timestamp": price.timestamp,
                "source": price.source
            }
            for price in prices
        ],
        "total": len(prices)
    }

@router.get("/live")
async def get_live_prices(
    coins: str = Query("ethereum,bitcoin", description="Comma-separated coin IDs"),
    currencies: str = Query("usd", description="Comma-separated currencies")
):
    """Get live prices from CoinGecko"""
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
        
        return {
            "prices": price_data,
            "timestamp": datetime.utcnow(),
            "source": "coingecko"
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch prices: {str(e)}")
