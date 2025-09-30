# ETHEREYE Web Scraping System

🔬 A comprehensive blockchain analytics platform with real-time web scraping capabilities for Ethereum data.

## 🌟 Features

### Web Scraping Capabilities
- **Real-time Data Collection**: Automated scraping from Etherscan and CoinGecko
- **Multi-source Integration**: Supports multiple blockchain data providers
- **Scheduled Tasks**: Background job processing with Celery
- **Rate Limiting**: Respects API rate limits with intelligent throttling
- **Data Validation**: Comprehensive data cleaning and validation
- **Error Recovery**: Automatic retries and error handling

### Supported Data Sources
- **Etherscan API**: Transactions, blocks, addresses, gas prices, token transfers
- **CoinGecko API**: Cryptocurrency prices, market data, trending coins
- **Direct RPC**: Ethereum node interactions (optional)

### Data Types
- ✅ Ethereum transactions and internal transactions
- ✅ Block data and mining information  
- ✅ Address balances and transaction history
- ✅ ERC-20 token transfers and token information
- ✅ Gas price tracking and analytics
- ✅ Cryptocurrency prices and market data
- ✅ DeFi protocol data (planned)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Redis server
- API keys (Etherscan, CoinGecko, etc.)

### Installation

1. **Clone and Setup**
   ```bash
   cd ETHEREYE-frontend-main
   python setup_scraping.py
   ```

2. **Configure API Keys**
   Edit `backend/.env` with your API keys:
   ```env
   ETHERSCAN_API_KEY=your_etherscan_api_key
   COINGECKO_API_KEY=your_coingecko_api_key
   INFURA_PROJECT_ID=your_infura_project_id
   ```

3. **Start Services**
   ```bash
   python setup_scraping.py --start-services
   ```

### Manual Setup (Advanced)

1. **Install Dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Setup Database**
   ```bash
   cd backend
   python -c "from models.database import init_database; init_database()"
   ```

3. **Start Services**
   ```bash
   # Terminal 1: API Server
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

   # Terminal 2: Celery Worker
   cd backend
   celery worker -A scheduler.scraping_scheduler:celery_app --loglevel=info

   # Terminal 3: Celery Beat (Scheduler)
   cd backend
   celery beat -A scheduler.scraping_scheduler:celery_app --loglevel=info
   ```

## 📊 API Endpoints

### Core Data APIs
```
GET  /api/v1/transactions     - Get transactions with filters
GET  /api/v1/addresses        - Get address information
GET  /api/v1/blocks          - Get block data
GET  /api/v1/prices          - Get price data
GET  /api/v1/analytics       - Get analytics and statistics
```

### Live Data APIs
```
GET  /api/v1/live/gas        - Current gas prices
GET  /api/v1/live/prices     - Live cryptocurrency prices
GET  /api/v1/transactions/live/latest - Latest transactions
```

### Scraping Control APIs
```
POST /api/v1/scraping/start  - Start manual scraping job
GET  /api/v1/scraping/jobs   - Get scraping job status
GET  /api/v1/scrapers/info   - Get available scrapers info
```

### Health and Status
```
GET  /health                 - API health check
GET  /                       - API information
GET  /docs                   - Interactive API documentation
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Required API Keys
ETHERSCAN_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here
INFURA_PROJECT_ID=your_project_id

# Database
DATABASE_URL=sqlite:///./ethereye.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Scraping Configuration
REQUEST_DELAY=1.0
MAX_RETRIES=3
TIMEOUT=30
REQUESTS_PER_SECOND=5

# Data Retention
DATA_RETENTION_DAYS=90
```

### Scraping Schedule
- **Latest Blocks**: Every 15 seconds
- **Gas Prices**: Every 5 minutes  
- **Cryptocurrency Prices**: Every 2 minutes
- **Market Data**: Every hour
- **Data Cleanup**: Daily at 2 AM

## 🎯 Frontend Integration

### Real-time Dashboard
The frontend automatically connects to the scraping backend:

```javascript
// API Client automatically available
const prices = await window.ethereyeApi.getLivePrices();
const transactions = await window.ethereyeApi.getLatestTransactions();
const gasData = await window.ethereyeApi.getLiveGas();
```

### Data Attributes for Live Updates
```html
<!-- Stats cards -->
<span data-stat="total-transactions">Loading...</span>
<span data-stat="eth-price">Loading...</span>

<!-- Live data containers -->
<div data-live="latest-transactions"></div>
<div data-live="gas-tracker"></div>

<!-- Charts -->
<canvas data-chart="transactions"></canvas>
```

## 📈 Monitoring & Analytics

### Built-in Analytics
- Transaction volume and count trends
- Gas price analysis and predictions
- Address activity patterns
- Token transfer analytics
- Network health metrics

### Logs and Monitoring
- Structured logging with Loguru
- Scraping job tracking and status
- Performance metrics collection
- Error rate monitoring

### Database Schema
```sql
-- Key tables created automatically
transactions       -- Ethereum transactions
token_transfers    -- ERC-20 transfers  
addresses          -- Address information
blocks            -- Block data
gas_tracker       -- Gas price history
price_data        -- Cryptocurrency prices
scraping_jobs     -- Job tracking
```

## 🔧 Advanced Usage

### Custom Scrapers
```python
from backend.scrapers.base_scraper import BaseScraper

class CustomScraper(BaseScraper):
    def __init__(self):
        super().__init__("custom", "https://api.example.com")
    
    async def scrape(self, endpoint, **kwargs):
        # Implement custom scraping logic
        return await self.make_request(f"/api/{endpoint}")
```

### Manual Scraping Jobs
```python
# Start specific scraping job
response = await ethereyeApi.startScraping(
    "etherscan", 
    "account_transactions", 
    {"address": "0x123...", "limit": 100}
)

# Monitor job progress  
job = await ethereyeApi.getScrapingJob(response.job_id)
```

### Data Processing Pipeline
```python
from backend.processors.data_processor import DataProcessor

processor = DataProcessor(db_session)

# Process scraped data
await processor.process_transactions(transaction_data)
await processor.process_token_transfers(transfer_data)
await processor.process_price_data(price_data)
```

## 🚨 Rate Limiting & Ethics

### Built-in Rate Limiting
- Configurable requests per second/minute/hour
- Automatic backoff on rate limit errors
- Respect for API provider limits
- Intelligent request scheduling

### Ethical Scraping
- Follows robots.txt guidelines
- Respects API terms of service
- Implements proper delays between requests
- Uses official APIs when available

## 🛠 Troubleshooting

### Common Issues

**Redis Connection Error**
```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis

# Install Redis (macOS)
brew install redis
brew services start redis
```

**API Key Issues**
- Verify API keys in `.env` file
- Check API key permissions and quotas
- Ensure keys are not rate limited

**Database Issues**
```bash
# Reinitialize database
cd backend
python -c "from models.database import init_database; init_database()"
```

**Import Errors**
```bash
# Check Python path
cd backend
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r requirements.txt
```

### Performance Tuning
- Adjust `REQUEST_DELAY` for faster/slower scraping
- Modify Celery worker concurrency
- Tune database connection pooling
- Configure Redis memory settings

## 📝 Development

### Project Structure
```
backend/
├── scrapers/           # Web scraping modules
├── processors/         # Data processing pipeline
├── models/            # Database models
├── api/               # FastAPI route handlers
├── scheduler/         # Celery task definitions
└── config.py          # Configuration management

ETHEREYE-frontend-main/
├── js/
│   ├── api-client.js      # API communication
│   └── dashboard-manager.js # Real-time updates
└── [existing frontend files]
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new scrapers
4. Update documentation
5. Submit pull request

## 📜 License

This web scraping extension is part of the ETHEREYE blockchain analytics platform. 

## 🔗 Resources

- [Etherscan API Documentation](https://docs.etherscan.io/)
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Redis Documentation](https://redis.io/documentation)

---

🎯 **ETHEREYE** - Powering blockchain analytics with intelligent web scraping