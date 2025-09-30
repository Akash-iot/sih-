/**
 * Dashboard Manager
 * Handles real-time data updates for the ETHEREYE dashboard
 */

class DashboardManager {
    constructor() {
        this.updateInterval = 30000; // 30 seconds
        this.intervals = {};
        this.cache = new Map();
        this.isInitialized = false;
        
        // Dashboard elements
        this.elements = {
            // Stats cards
            totalTransactions: document.querySelector('[data-stat="total-transactions"]'),
            totalBlocks: document.querySelector('[data-stat="total-blocks"]'),
            transactions24h: document.querySelector('[data-stat="transactions-24h"]'),
            avgGasPrice: document.querySelector('[data-stat="avg-gas-price"]'),
            ethPrice: document.querySelector('[data-stat="eth-price"]'),
            
            // Live data containers
            latestTransactions: document.querySelector('[data-live="latest-transactions"]'),
            latestBlocks: document.querySelector('[data-live="latest-blocks"]'),
            gasTracker: document.querySelector('[data-live="gas-tracker"]'),
            
            // Charts
            transactionChart: document.querySelector('[data-chart="transactions"]'),
            gasChart: document.querySelector('[data-chart="gas-prices"]'),
            
            // Status indicators
            connectionStatus: document.querySelector('[data-status="connection"]'),
            lastUpdated: document.querySelector('[data-status="last-updated"]')
        };
    }

    /**
     * Initialize dashboard with data
     */
    async initialize() {
        try {
            console.log('Initializing ETHEREYE Dashboard...');
            
            // Check API connection
            await this.checkConnection();
            
            // Load initial data
            await this.loadInitialData();
            
            // Start real-time updates
            this.startRealTimeUpdates();
            
            // Set up event listeners
            this.setupEventListeners();
            
            this.isInitialized = true;
            console.log('Dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to connect to ETHEREYE API');
        }
    }

    /**
     * Check API connection
     */
    async checkConnection() {
        try {
            const health = await window.ethereyeApi.healthCheck();
            this.updateConnectionStatus(true);
            console.log('API Health:', health);
        } catch (error) {
            this.updateConnectionStatus(false);
            throw new Error('API connection failed');
        }
    }

    /**
     * Load initial dashboard data
     */
    async loadInitialData() {
        try {
            // Load overview data
            const overview = await window.ethereyeApi.getAnalyticsOverview();
            this.updateOverviewStats(overview.overview);
            
            // Load latest transactions
            const transactions = await window.ethereyeApi.getLatestTransactions(10);
            this.updateLatestTransactions(transactions.transactions || []);
            
            // Load latest blocks
            const blocks = await window.ethereyeApi.getBlocks({ limit: 5 });
            this.updateLatestBlocks(blocks.blocks || []);
            
            // Load gas data
            const gasData = await window.ethereyeApi.getLiveGas();
            this.updateGasTracker(gasData);
            
            // Load price data
            const prices = await window.ethereyeApi.getLiveCryptoPrices();
            this.updatePrices(prices.prices || []);
            
            this.updateLastUpdated();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        // Update overview stats every minute
        this.intervals.overview = setInterval(() => {
            this.updateOverviewData();
        }, 60000);
        
        // Update live data every 30 seconds
        this.intervals.liveData = setInterval(() => {
            this.updateLiveData();
        }, this.updateInterval);
        
        // Update prices every 2 minutes
        this.intervals.prices = setInterval(() => {
            this.updatePriceData();
        }, 120000);
    }

    /**
     * Update overview statistics
     */
    async updateOverviewData() {
        try {
            const overview = await window.ethereyeApi.getAnalyticsOverview();
            this.updateOverviewStats(overview.overview);
        } catch (error) {
            console.error('Failed to update overview data:', error);
        }
    }

    /**
     * Update live transaction and block data
     */
    async updateLiveData() {
        try {
            // Update latest transactions
            const transactions = await window.ethereyeApi.getLatestTransactions(10);
            this.updateLatestTransactions(transactions.transactions || []);
            
            // Update gas data
            const gasData = await window.ethereyeApi.getLiveGas();
            this.updateGasTracker(gasData);
            
            this.updateLastUpdated();
            
        } catch (error) {
            console.error('Failed to update live data:', error);
        }
    }

    /**
     * Update price data
     */
    async updatePriceData() {
        try {
            const prices = await window.ethereyeApi.getLiveCryptoPrices();
            this.updatePrices(prices.prices || []);
        } catch (error) {
            console.error('Failed to update price data:', error);
        }
    }

    /**
     * Update overview statistics in UI
     */
    updateOverviewStats(stats) {
        if (!stats) return;
        
        if (this.elements.totalTransactions) {
            this.elements.totalTransactions.textContent = this.formatNumber(stats.total_transactions);
        }
        
        if (this.elements.totalBlocks) {
            this.elements.totalBlocks.textContent = this.formatNumber(stats.total_blocks);
        }
        
        if (this.elements.transactions24h) {
            this.elements.transactions24h.textContent = this.formatNumber(stats.transactions_24h);
        }
        
        if (this.elements.avgGasPrice) {
            this.elements.avgGasPrice.textContent = `${stats.avg_gas_price_gwei.toFixed(1)} Gwei`;
        }
        
        if (this.elements.ethPrice) {
            this.elements.ethPrice.textContent = `$${stats.eth_price_usd.toFixed(2)}`;
        }
    }

    /**
     * Update latest transactions in UI
     */
    updateLatestTransactions(transactions) {
        if (!this.elements.latestTransactions || !transactions) return;
        
        const html = transactions.slice(0, 5).map(tx => `
            <div class="transaction-item" data-hash="${tx.hash}">
                <div class="tx-hash">
                    <a href="#" onclick="viewTransaction('${tx.hash}')">${this.truncateHash(tx.hash)}</a>
                </div>
                <div class="tx-details">
                    <span class="from">${this.truncateHash(tx.from_address)}</span>
                    <span class="arrow">→</span>
                    <span class="to">${this.truncateHash(tx.to_address)}</span>
                </div>
                <div class="tx-value">
                    ${tx.value_eth ? tx.value_eth.toFixed(4) : '0'} ETH
                </div>
                <div class="tx-time">
                    ${this.timeAgo(tx.timestamp)}
                </div>
            </div>
        `).join('');
        
        this.elements.latestTransactions.innerHTML = html;
    }

    /**
     * Update latest blocks in UI
     */
    updateLatestBlocks(blocks) {
        if (!this.elements.latestBlocks || !blocks) return;
        
        const html = blocks.slice(0, 5).map(block => `
            <div class="block-item" data-number="${block.block_number}">
                <div class="block-number">
                    <a href="#" onclick="viewBlock(${block.block_number})">#${block.block_number}</a>
                </div>
                <div class="block-details">
                    <span>Txs: ${block.transaction_count || 0}</span>
                    <span>Gas: ${this.formatPercentage(block.gas_used, block.gas_limit)}%</span>
                </div>
                <div class="block-time">
                    ${this.timeAgo(block.timestamp)}
                </div>
            </div>
        `).join('');
        
        this.elements.latestBlocks.innerHTML = html;
    }

    /**
     * Update gas tracker in UI
     */
    updateGasTracker(gasData) {
        if (!this.elements.gasTracker || !gasData) return;
        
        const html = `
            <div class="gas-prices">
                <div class="gas-price safe">
                    <label>Safe</label>
                    <value>${gasData.safe_gas_price || 0} Gwei</value>
                </div>
                <div class="gas-price standard">
                    <label>Standard</label>
                    <value>${gasData.standard_gas_price || 0} Gwei</value>
                </div>
                <div class="gas-price fast">
                    <label>Fast</label>
                    <value>${gasData.fast_gas_price || 0} Gwei</value>
                </div>
            </div>
        `;
        
        this.elements.gasTracker.innerHTML = html;
    }

    /**
     * Update cryptocurrency prices
     */
    updatePrices(prices) {
        prices.forEach(priceData => {
            const element = document.querySelector(`[data-price="${priceData.coin_id}"]`);
            if (element) {
                const price = priceData.price ? `$${priceData.price.toFixed(2)}` : 'N/A';
                const change = priceData.change_24h ? `${priceData.change_24h.toFixed(2)}%` : '';
                const changeClass = priceData.change_24h >= 0 ? 'positive' : 'negative';
                
                element.innerHTML = `
                    <span class="price">${price}</span>
                    <span class="change ${changeClass}">${change}</span>
                `;
            }
        });
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(connected) {
        if (this.elements.connectionStatus) {
            this.elements.connectionStatus.className = connected ? 'status-online' : 'status-offline';
            this.elements.connectionStatus.textContent = connected ? 'Online' : 'Offline';
        }
    }

    /**
     * Update last updated timestamp
     */
    updateLastUpdated() {
        if (this.elements.lastUpdated) {
            this.elements.lastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Search functionality
        const searchInput = document.querySelector('[data-search="input"]');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.performSearch(e.target.value);
                }
            });
        }
        
        // Refresh button
        const refreshButton = document.querySelector('[data-action="refresh"]');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.loadInitialData();
            });
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
    }

    /**
     * Perform search
     */
    async performSearch(query) {
        if (!query.trim()) return;
        
        try {
            const results = await window.ethereyeApi.searchTransactions(query);
            this.displaySearchResults(results);
        } catch (error) {
            console.error('Search failed:', error);
            this.showError('Search failed');
        }
    }

    /**
     * Display search results
     */
    displaySearchResults(results) {
        // Implementation depends on your UI structure
        console.log('Search results:', results);
    }

    /**
     * Show error message
     */
    showError(message) {
        // Create or update error notification
        const errorElement = document.querySelector('.error-notification') || this.createErrorElement();
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    }

    /**
     * Create error element
     */
    createErrorElement() {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4444;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 9999;
            display: none;
        `;
        document.body.appendChild(errorDiv);
        return errorDiv;
    }

    /**
     * Pause updates when tab is not visible
     */
    pauseUpdates() {
        Object.values(this.intervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
    }

    /**
     * Resume updates when tab becomes visible
     */
    resumeUpdates() {
        if (this.isInitialized) {
            this.startRealTimeUpdates();
        }
    }

    /**
     * Cleanup intervals
     */
    destroy() {
        this.pauseUpdates();
        this.cache.clear();
        this.isInitialized = false;
    }

    // ===================
    // UTILITY METHODS
    // ===================

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    formatPercentage(used, total) {
        if (!used || !total) return 0;
        return ((used / total) * 100).toFixed(1);
    }

    truncateHash(hash, length = 8) {
        if (!hash) return '';
        return `${hash.slice(0, length)}...${hash.slice(-length)}`;
    }

    timeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = now - time;
        
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return `${seconds}s ago`;
    }
}

// Global functions for onclick handlers
window.viewTransaction = (hash) => {
    console.log('Viewing transaction:', hash);
    // Implement navigation to transaction detail
};

window.viewBlock = (blockNumber) => {
    console.log('Viewing block:', blockNumber);
    // Implement navigation to block detail
};

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardManager = new DashboardManager();
    window.dashboardManager.initialize();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardManager;
}