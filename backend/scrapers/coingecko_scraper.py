from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from .base_scraper import BaseScraper
from ..config import settings

class CoinGeckoScraper(BaseScraper):
    """Scraper for CoinGecko API to collect cryptocurrency price and market data"""
    
    def __init__(self):
        super().__init__("coingecko", settings.COINGECKO_BASE_URL)
        self.api_key = settings.COINGECKO_API_KEY
    
    def get_supported_endpoints(self) -> List[str]:
        return [
            "simple_price",
            "coin_info",
            "market_data",
            "historical_price",
            "trending_coins",
            "global_market",
            "exchanges",
            "coin_list",
            "price_change",
            "market_chart"
        ]
    
    async def scrape_simple_price(
        self, 
        coin_ids: List[str], 
        vs_currencies: List[str] = ["usd"],
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
        include_24hr_change: bool = True
    ) -> List[Dict]:
        """Get current prices for specified coins"""
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
            "include_market_cap": str(include_market_cap).lower(),
            "include_24hr_vol": str(include_24hr_vol).lower(),
            "include_24hr_change": str(include_24hr_change).lower()
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/simple/price"
        response = await self.make_request(url, params=params)
        
        results = []
        for coin_id, price_data in response.items():
            for currency in vs_currencies:
                result = {
                    "coin_id": coin_id,
                    "currency": currency,
                    "price": price_data.get(currency),
                    "market_cap": price_data.get(f"{currency}_market_cap"),
                    "volume_24h": price_data.get(f"{currency}_24h_vol"),
                    "change_24h": price_data.get(f"{currency}_24h_change"),
                    "timestamp": datetime.now(),
                    "source": "coingecko"
                }
                
                if self.validate_data(result, ["coin_id", "currency", "price"]):
                    results.append(result)
        
        return results
    
    async def scrape_coin_info(self, coin_id: str) -> Dict:
        """Get detailed information about a specific coin"""
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/coins/{coin_id}"
        response = await self.make_request(url, params=params)
        
        if response:
            market_data = response.get("market_data", {})
            
            return {
                "coin_id": response.get("id"),
                "name": response.get("name"),
                "symbol": response.get("symbol", "").upper(),
                "description": response.get("description", {}).get("en", "")[:500],  # Truncate description
                "homepage": response.get("links", {}).get("homepage", [None])[0],
                "blockchain_site": response.get("links", {}).get("blockchain_site", []),
                "genesis_date": response.get("genesis_date"),
                "market_cap_rank": response.get("market_cap_rank"),
                "current_price_usd": market_data.get("current_price", {}).get("usd"),
                "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
                "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
                "high_24h_usd": market_data.get("high_24h", {}).get("usd"),
                "low_24h_usd": market_data.get("low_24h", {}).get("usd"),
                "price_change_24h": market_data.get("price_change_24h"),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
                "price_change_percentage_7d": market_data.get("price_change_percentage_7d"),
                "price_change_percentage_30d": market_data.get("price_change_percentage_30d"),
                "circulating_supply": market_data.get("circulating_supply"),
                "total_supply": market_data.get("total_supply"),
                "max_supply": market_data.get("max_supply"),
                "ath_usd": market_data.get("ath", {}).get("usd"),
                "ath_date": market_data.get("ath_date", {}).get("usd"),
                "atl_usd": market_data.get("atl", {}).get("usd"),
                "atl_date": market_data.get("atl_date", {}).get("usd"),
                "last_updated": response.get("last_updated"),
                "timestamp": datetime.now(),
                "source": "coingecko"
            }
        
        return {}
    
    async def scrape_market_data(
        self, 
        vs_currency: str = "usd",
        per_page: int = 100,
        page: int = 1,
        order: str = "market_cap_desc"
    ) -> List[Dict]:
        """Get market data for top cryptocurrencies"""
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d"
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/coins/markets"
        response = await self.make_request(url, params=params)
        
        results = []
        for coin in response:
            result = {
                "coin_id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "market_cap_rank": coin.get("market_cap_rank"),
                "current_price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "total_volume": coin.get("total_volume"),
                "high_24h": coin.get("high_24h"),
                "low_24h": coin.get("low_24h"),
                "price_change_24h": coin.get("price_change_24h"),
                "price_change_percentage_24h": coin.get("price_change_percentage_24h"),
                "price_change_percentage_7d": coin.get("price_change_percentage_7d_in_currency"),
                "price_change_percentage_30d": coin.get("price_change_percentage_30d_in_currency"),
                "market_cap_change_24h": coin.get("market_cap_change_24h"),
                "market_cap_change_percentage_24h": coin.get("market_cap_change_percentage_24h"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply"),
                "max_supply": coin.get("max_supply"),
                "ath": coin.get("ath"),
                "ath_change_percentage": coin.get("ath_change_percentage"),
                "ath_date": coin.get("ath_date"),
                "atl": coin.get("atl"),
                "atl_change_percentage": coin.get("atl_change_percentage"),
                "atl_date": coin.get("atl_date"),
                "last_updated": coin.get("last_updated"),
                "vs_currency": vs_currency,
                "timestamp": datetime.now(),
                "source": "coingecko"
            }
            
            if self.validate_data(result, ["coin_id", "current_price"]):
                results.append(result)
        
        return results
    
    async def scrape_historical_price(
        self, 
        coin_id: str, 
        vs_currency: str = "usd",
        days: int = 30
    ) -> List[Dict]:
        """Get historical price data for a coin"""
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
            "interval": "daily"
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        response = await self.make_request(url, params=params)
        
        results = []
        if response and "prices" in response:
            prices = response["prices"]
            volumes = response.get("total_volumes", [])
            market_caps = response.get("market_caps", [])
            
            for i, price_data in enumerate(prices):
                timestamp = datetime.fromtimestamp(price_data[0] / 1000)
                price = price_data[1]
                
                result = {
                    "coin_id": coin_id,
                    "vs_currency": vs_currency,
                    "timestamp": timestamp,
                    "price": price,
                    "volume": volumes[i][1] if i < len(volumes) else None,
                    "market_cap": market_caps[i][1] if i < len(market_caps) else None,
                    "source": "coingecko"
                }
                
                if self.validate_data(result, ["coin_id", "price", "timestamp"]):
                    results.append(result)
        
        return results
    
    async def scrape_trending_coins(self) -> List[Dict]:
        """Get trending coins"""
        params = {}
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/search/trending"
        response = await self.make_request(url, params=params)
        
        results = []
        if response and "coins" in response:
            for coin_data in response["coins"]:
                coin = coin_data.get("item", {})
                
                result = {
                    "coin_id": coin.get("id"),
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol", "").upper(),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "thumb": coin.get("thumb"),
                    "small": coin.get("small"),
                    "large": coin.get("large"),
                    "slug": coin.get("slug"),
                    "price_btc": coin.get("price_btc"),
                    "score": coin.get("score"),
                    "timestamp": datetime.now(),
                    "source": "coingecko"
                }
                
                if self.validate_data(result, ["coin_id", "name"]):
                    results.append(result)
        
        return results
    
    async def scrape_global_market(self) -> Dict:
        """Get global cryptocurrency market data"""
        params = {}
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/global"
        response = await self.make_request(url, params=params)
        
        if response and "data" in response:
            data = response["data"]
            
            return {
                "active_cryptocurrencies": data.get("active_cryptocurrencies"),
                "upcoming_icos": data.get("upcoming_icos"),
                "ongoing_icos": data.get("ongoing_icos"),
                "ended_icos": data.get("ended_icos"),
                "markets": data.get("markets"),
                "total_market_cap": data.get("total_market_cap", {}).get("usd"),
                "total_volume": data.get("total_volume", {}).get("usd"),
                "market_cap_percentage": data.get("market_cap_percentage", {}),
                "market_cap_change_percentage_24h_usd": data.get("market_cap_change_percentage_24h_usd"),
                "updated_at": data.get("updated_at"),
                "timestamp": datetime.now(),
                "source": "coingecko"
            }
        
        return {}
    
    async def scrape_exchanges(self, per_page: int = 100) -> List[Dict]:
        """Get cryptocurrency exchanges data"""
        params = {
            "per_page": per_page
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/exchanges"
        response = await self.make_request(url, params=params)
        
        results = []
        for exchange in response:
            result = {
                "exchange_id": exchange.get("id"),
                "name": exchange.get("name"),
                "year_established": exchange.get("year_established"),
                "country": exchange.get("country"),
                "description": exchange.get("description", "")[:200] if exchange.get("description") else "",
                "url": exchange.get("url"),
                "image": exchange.get("image"),
                "has_trading_incentive": exchange.get("has_trading_incentive"),
                "trust_score": exchange.get("trust_score"),
                "trust_score_rank": exchange.get("trust_score_rank"),
                "trade_volume_24h_btc": exchange.get("trade_volume_24h_btc"),
                "trade_volume_24h_btc_normalized": exchange.get("trade_volume_24h_btc_normalized"),
                "timestamp": datetime.now(),
                "source": "coingecko"
            }
            
            if self.validate_data(result, ["exchange_id", "name"]):
                results.append(result)
        
        return results
    
    async def scrape_coin_list(self, include_platform: bool = True) -> List[Dict]:
        """Get list of all supported coins"""
        params = {
            "include_platform": str(include_platform).lower()
        }
        
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.base_url}/coins/list"
        response = await self.make_request(url, params=params)
        
        results = []
        for coin in response:
            result = {
                "coin_id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "platforms": coin.get("platforms", {}) if include_platform else {},
                "timestamp": datetime.now(),
                "source": "coingecko"
            }
            
            if self.validate_data(result, ["coin_id", "name", "symbol"]):
                results.append(result)
        
        return results
    
    async def scrape(self, endpoint: str, **kwargs) -> List[Dict]:
        """Main scraping method"""
        try:
            if endpoint == "simple_price":
                return await self.scrape_simple_price(
                    kwargs.get("coin_ids", ["bitcoin", "ethereum"]),
                    kwargs.get("vs_currencies", ["usd"]),
                    kwargs.get("include_market_cap", True),
                    kwargs.get("include_24hr_vol", True),
                    kwargs.get("include_24hr_change", True)
                )
            
            elif endpoint == "coin_info":
                result = await self.scrape_coin_info(kwargs.get("coin_id"))
                return [result] if result else []
            
            elif endpoint == "market_data":
                return await self.scrape_market_data(
                    kwargs.get("vs_currency", "usd"),
                    kwargs.get("per_page", 100),
                    kwargs.get("page", 1),
                    kwargs.get("order", "market_cap_desc")
                )
            
            elif endpoint == "historical_price":
                return await self.scrape_historical_price(
                    kwargs.get("coin_id"),
                    kwargs.get("vs_currency", "usd"),
                    kwargs.get("days", 30)
                )
            
            elif endpoint == "trending_coins":
                return await self.scrape_trending_coins()
            
            elif endpoint == "global_market":
                result = await self.scrape_global_market()
                return [result] if result else []
            
            elif endpoint == "exchanges":
                return await self.scrape_exchanges(kwargs.get("per_page", 100))
            
            elif endpoint == "coin_list":
                return await self.scrape_coin_list(kwargs.get("include_platform", True))
            
            else:
                logger.error(f"Unsupported endpoint: {endpoint}")
                return []
                
        except Exception as e:
            logger.error(f"Scraping failed for endpoint {endpoint}: {str(e)}")
            return []