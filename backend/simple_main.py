from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
import os
import sys
import random
import hashlib
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ML API router
try:
    from backend.api.advanced_ml_endpoints import router as ml_router
    ML_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced ML endpoints not available: {e}")
    ML_SERVICES_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="ETHEREYE Analytics API - Demo",
    description="Simplified blockchain analytics API with web scraping",
    version="1.0.0-demo"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
demo_data = {
    "gas_prices": {},
    "crypto_prices": {},
    "transactions": [],
    "last_updated": None
}

class SimpleEtherscanScraper:
    """Simplified Etherscan scraper for demo"""
    
    def __init__(self):
        self.base_url = "https://api.etherscan.io/api"
        self.api_key = os.getenv("ETHERSCAN_API_KEY", "YourApiKeyToken")  # Demo key
    
    async def get_gas_prices(self):
        """Get current gas prices from Etherscan"""
        try:
            url = f"{self.base_url}?module=gastracker&action=gasoracle&apikey={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "1":
                            result = data.get("result", {})
                            return {
                                "safe_gas_price": int(result.get("SafeGasPrice", 12)),
                                "standard_gas_price": int(result.get("StandardGasPrice", 15)),
                                "fast_gas_price": int(result.get("FastGasPrice", 18)),
                                "timestamp": datetime.now().isoformat(),
                                "source": "etherscan"
                            }
        except Exception as e:
            print(f"Error fetching gas prices: {e}")
        
        # Always return demo data if API fails or times out
        return {
            "safe_gas_price": 12,
            "standard_gas_price": 15,
            "fast_gas_price": 18,
            "timestamp": datetime.now().isoformat(),
            "source": "demo"
        }

class SimpleCoinGeckoScraper:
    """Simplified CoinGecko scraper for demo"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    async def get_prices(self, coin_ids: List[str] = None, vs_currencies: List[str] = None):
        """Get cryptocurrency prices from CoinGecko"""
        if coin_ids is None:
            coin_ids = ["ethereum", "bitcoin"]
        if vs_currencies is None:
            vs_currencies = ["usd"]
        
        try:
            ids = ",".join(coin_ids)
            currencies = ",".join(vs_currencies)
            url = f"{self.base_url}/simple/price?ids={ids}&vs_currencies={currencies}&include_24hr_change=true"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = []
                        
                        for coin_id, price_data in data.items():
                            for currency in vs_currencies:
                                result.append({
                                    "coin_id": coin_id,
                                    "currency": currency,
                                    "price": price_data.get(currency, 0),
                                    "change_24h": price_data.get(f"{currency}_24h_change", 0),
                                    "timestamp": datetime.now().isoformat(),
                                    "source": "coingecko"
                                })
                        
                        return result
                        
        except Exception as e:
            print(f"Error fetching crypto prices: {e}")
            # Return demo data if API fails
            return [
                {
                    "coin_id": "ethereum",
                    "currency": "usd", 
                    "price": 2450.50,
                    "change_24h": 2.34,
                    "timestamp": datetime.now().isoformat(),
                    "source": "demo"
                },
                {
                    "coin_id": "bitcoin",
                    "currency": "usd",
                    "price": 67800.25,
                    "change_24h": 1.82,
                    "timestamp": datetime.now().isoformat(),
                    "source": "demo"
                }
            ]
        
        return []

# Initialize scrapers
etherscan_scraper = SimpleEtherscanScraper()
coingecko_scraper = SimpleCoinGeckoScraper()

# Include advanced ML router if available
if ML_SERVICES_AVAILABLE:
    app.include_router(ml_router, prefix="/api/v1/ml", tags=["Advanced ML"])
    print("✅ Advanced ML endpoints registered successfully")
else:
    print("⚠️  Advanced ML endpoints not available")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ETHEREYE Analytics API - Demo",
        "version": "1.0.0-demo",
        "description": "Simplified blockchain analytics API with web scraping",
        "features": [
            "Live gas price tracking",
            "Cryptocurrency price monitoring",
            "Real-time data scraping",
            "Spider map network analysis",
            "Transaction relationship mapping",
            "Risk analysis and scoring",
            "DBSCAN clustering analysis",
            "IsolationForest anomaly detection",
            "NLP/PII extraction (spaCy, NLTK, HuggingFace)",
            "Custom ML risk scoring pipeline",
            "Comprehensive wallet analysis",
            "REST API endpoints"
        ],
        "endpoints": {
            "health": "/health",
            "gas_prices": "/api/v1/live/gas",
            "crypto_prices": "/api/v1/live/prices",
            "analytics": "/api/v1/analytics/overview",
            "spider_map": "/api/v1/spider-map/network/{address}",
            "risk_analysis": "/api/v1/spider-map/risk-analysis/{address}",
            "ml_clustering": "/api/v1/ml/clustering/dbscan",
            "ml_anomaly": "/api/v1/ml/clustering/anomaly-detection",
            "ml_nlp_pii": "/api/v1/ml/nlp/extract-pii",
            "ml_risk_predict": "/api/v1/ml/risk-scoring/predict",
            "ml_comprehensive": "/api/v1/ml/comprehensive-analysis",
            "ml_status": "/api/v1/ml/ml-services/status",
            "docs": "/docs"
        },
        "status": "demo"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "demo",
        "scrapers": {
            "etherscan": "ready",
            "coingecko": "ready"
        }
    }

@app.get("/api/v1/live/gas")
async def get_live_gas_prices():
    """Get current gas prices"""
    try:
        gas_data = await etherscan_scraper.get_gas_prices()
        
        if gas_data:
            demo_data["gas_prices"] = gas_data
            demo_data["last_updated"] = datetime.now().isoformat()
            return gas_data
        else:
            raise HTTPException(status_code=503, detail="Failed to fetch gas prices")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.get("/api/v1/live/prices")
async def get_live_prices(
    coins: str = "ethereum,bitcoin",
    currencies: str = "usd"
):
    """Get live cryptocurrency prices"""
    try:
        coin_list = [coin.strip() for coin in coins.split(",")]
        currency_list = [currency.strip() for currency in currencies.split(",")]
        
        price_data = await coingecko_scraper.get_prices(coin_list, currency_list)
        
        if price_data:
            demo_data["crypto_prices"] = price_data
            demo_data["last_updated"] = datetime.now().isoformat()
            return {
                "prices": price_data,
                "timestamp": datetime.now().isoformat(),
                "source": "live"
            }
        else:
            raise HTTPException(status_code=503, detail="Failed to fetch prices")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.get("/api/v1/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview for dashboard"""
    return {
        "overview": {
            "total_transactions": 1234567,
            "total_blocks": 18500000,
            "transactions_24h": 45678,
            "avg_gas_price_gwei": demo_data.get("gas_prices", {}).get("standard_gas_price", 15),
            "eth_price_usd": next(
                (p["price"] for p in demo_data.get("crypto_prices", []) 
                 if p["coin_id"] == "ethereum"), 
                2450.50
            ),
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/api/v1/demo/data")
async def get_demo_data():
    """Get all cached demo data"""
    return {
        "data": demo_data,
        "timestamp": datetime.now().isoformat(),
        "note": "This is demo data for ETHEREYE blockchain analytics platform"
    }

@app.post("/api/v1/demo/refresh")
async def refresh_demo_data():
    """Manually refresh demo data"""
    try:
        # Fetch fresh data
        gas_data = await etherscan_scraper.get_gas_prices()
        price_data = await coingecko_scraper.get_prices()
        
        # Update demo data
        if gas_data:
            demo_data["gas_prices"] = gas_data
        if price_data:
            demo_data["crypto_prices"] = price_data
        
        demo_data["last_updated"] = datetime.now().isoformat()
        
        return {
            "message": "Demo data refreshed successfully",
            "timestamp": datetime.now().isoformat(),
            "data": demo_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh data: {str(e)}")

# Background task to refresh data periodically
async def refresh_data_periodically():
    """Background task to refresh data every 5 minutes"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            # Refresh gas prices
            gas_data = await etherscan_scraper.get_gas_prices()
            if gas_data:
                demo_data["gas_prices"] = gas_data
            
            # Refresh crypto prices
            price_data = await coingecko_scraper.get_prices()
            if price_data:
                demo_data["crypto_prices"] = price_data
            
            demo_data["last_updated"] = datetime.now().isoformat()
            print(f"✅ Demo data refreshed at {datetime.now()}")
            
        except Exception as e:
            print(f"Error in background refresh: {e}")

# Spider Map endpoints
@app.get("/api/v1/spider-map/network/{address}")
async def get_spider_network(address: str, depth: int = 2, limit: int = 20, risk_analysis: bool = True):
    """Generate spider map network data"""
    
    # Validate address format
    if not address.startswith('0x') or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format")
    
    # Generate network data
    network = await generate_spider_network(address, depth, limit, risk_analysis)
    
    return {
        "center_address": address,
        "network": network,
        "metadata": {
            "depth": depth,
            "total_nodes": len(network["nodes"]),
            "total_edges": len(network["links"]),
            "analysis_timestamp": datetime.now().isoformat(),
            "risk_level": calculate_risk_level(network["nodes"])
        }
    }

@app.get("/api/v1/spider-map/risk-analysis/{address}")
async def get_risk_analysis(address: str):
    """Get risk analysis for address"""
    return {
        "address": address,
        "risk_score": random.uniform(0.1, 0.9),
        "risk_level": random.choice(["low", "medium", "high"]),
        "factors": {
            "transaction_frequency": random.uniform(0, 1),
            "large_transactions": random.uniform(0, 1),
            "mixer_interaction": random.uniform(0, 1),
            "blacklist_connections": random.uniform(0, 1)
        },
        "confidence": random.uniform(0.7, 0.95)
    }

async def generate_spider_network(address: str, depth: int, limit: int, risk_analysis: bool):
    """Generate spider network data"""
    
    # Center node
    center_node = {
        "id": address,
        "type": "center",
        "balance": f"{random.uniform(0.1, 100):.4f} ETH",
        "transactions": random.randint(50, 1000),
        "risk": random.choice(["low", "medium", "high"]) if risk_analysis else "unknown"
    }
    
    nodes = [center_node]
    links = []
    
    # Generate connections
    connection_types = ["incoming", "outgoing", "contract", "risky", "exchange"]
    num_connections = min(limit, random.randint(8, 25))
    
    for i in range(num_connections):
        node_address = generate_related_address(address, i)
        conn_type = random.choice(connection_types)
        
        node = {
            "id": node_address,
            "type": conn_type,
            "balance": f"{random.uniform(0.01, 50):.4f} ETH",
            "transactions": random.randint(1, 500),
            "risk": assign_risk_level(conn_type) if risk_analysis else "unknown",
            "value": random.uniform(0.1, 5)
        }
        
        nodes.append(node)
        
        link = {
            "source": address,
            "target": node_address,
            "value": random.uniform(0.01, 20),
            "type": conn_type,
            "transaction_count": random.randint(1, 50)
        }
        
        links.append(link)
    
    return {"nodes": nodes[:limit], "links": links}

def generate_related_address(base_address: str, seed: int) -> str:
    """Generate related address"""
    hash_input = f"{base_address}{seed}".encode()
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    return "0x" + hash_hex[:40]

def assign_risk_level(connection_type: str) -> str:
    """Assign risk level"""
    risk_mapping = {
        "risky": "high",
        "exchange": "medium", 
        "contract": "low",
        "incoming": "low",
        "outgoing": "low"
    }
    return risk_mapping.get(connection_type, "medium")

def calculate_risk_level(nodes: List[Dict]) -> str:
    """Calculate network risk"""
    risk_scores = {"low": 1, "medium": 2, "high": 3, "unknown": 1}
    total_risk = sum(risk_scores.get(node.get("risk", "unknown"), 1) for node in nodes)
    avg_risk = total_risk / len(nodes) if nodes else 1
    
    if avg_risk >= 2.5:
        return "high"
    elif avg_risk >= 1.8:
        return "medium"
    else:
        return "low"

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    print("🚀 ETHEREYE Demo API starting...")
    
    # Initial data load
    try:
        gas_data = await etherscan_scraper.get_gas_prices()
        if gas_data:
            demo_data["gas_prices"] = gas_data
        
        price_data = await coingecko_scraper.get_prices()
        if price_data:
            demo_data["crypto_prices"] = price_data
        
        demo_data["last_updated"] = datetime.now().isoformat()
        print("✅ Initial demo data loaded")
    except Exception as e:
        print(f"⚠️  Initial data load failed: {e}")
    
    # Start background refresh task
    asyncio.create_task(refresh_data_periodically())
    print("✅ Background refresh task started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)