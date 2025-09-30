/**
 * ETHEREYE API Client
 * Handles all communication with the backend scraping API
 */

class EthereyeApiClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.apiVersion = 'v1';
        this.endpoints = {
            transactions: '/api/v1/transactions',
            addresses: '/api/v1/addresses',
            blocks: '/api/v1/blocks',
            prices: '/api/v1/prices',
            analytics: '/api/v1/analytics',
            scraping: '/api/v1/scraping',
            live: '/api/v1/live',
            traces: '/api/v1/traces',
            spiderMap: '/api/v1/spider-map'
        };
    }

    /**
     * Make HTTP request to API
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    /**
     * Get query string from parameters
     */
    buildQueryString(params) {
        if (!params || Object.keys(params).length === 0) return '';
        
        const queryParams = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                queryParams.append(key, value);
            }
        });
        
        return `?${queryParams.toString()}`;
    }

    // ===================
    // TRANSACTION METHODS
    // ===================

    /**
     * Get transactions with optional filters
     */
    async getTransactions(params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.transactions}${queryString}`);
    }

    /**
     * Get specific transaction by hash
     */
    async getTransaction(txHash) {
        return await this.makeRequest(`${this.endpoints.transactions}/${txHash}`);
    }

    /**
     * Get latest transactions
     */
    async getLatestTransactions(count = 10) {
        return await this.makeRequest(`${this.endpoints.transactions}/live/latest?count=${count}`);
    }

    /**
     * Get daily transaction statistics
     */
    async getTransactionStats(days = 30) {
        return await this.makeRequest(`${this.endpoints.transactions}/stats/daily?days=${days}`);
    }

    /**
     * Search transactions
     */
    async searchTransactions(query) {
        return await this.makeRequest(`${this.endpoints.transactions}/search?q=${encodeURIComponent(query)}`);
    }

    // ===================
    // ADDRESS METHODS
    // ===================

    /**
     * Get addresses
     */
    async getAddresses(params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.addresses}${queryString}`);
    }

    /**
     * Get address details
     */
    async getAddress(address) {
        return await this.makeRequest(`${this.endpoints.addresses}/${address}`);
    }

    /**
     * Get transactions for an address
     */
    async getAddressTransactions(address, params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.addresses}/${address}/transactions${queryString}`);
    }

    // ===================
    // BLOCK METHODS
    // ===================

    /**
     * Get blocks
     */
    async getBlocks(params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.blocks}${queryString}`);
    }

    /**
     * Get specific block
     */
    async getBlock(blockNumber) {
        return await this.makeRequest(`${this.endpoints.blocks}/${blockNumber}`);
    }

    // ===================
    // PRICE METHODS
    // ===================

    /**
     * Get stored price data
     */
    async getPrices(params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.prices}${queryString}`);
    }

    /**
     * Get live prices
     */
    async getLivePrices(coins = 'ethereum,bitcoin', currencies = 'usd') {
        return await this.makeRequest(`${this.endpoints.prices}/live?coins=${coins}&currencies=${currencies}`);
    }

    // ===================
    // ANALYTICS METHODS
    // ===================

    /**
     * Get analytics overview
     */
    async getAnalyticsOverview() {
        return await this.makeRequest(`${this.endpoints.analytics}/overview`);
    }

    /**
     * Get gas tracker analytics
     */
    async getGasAnalytics(days = 7) {
        return await this.makeRequest(`${this.endpoints.analytics}/gas-tracker?days=${days}`);
    }

    /**
     * Get network statistics
     */
    async getNetworkStats(days = 30) {
        return await this.makeRequest(`${this.endpoints.analytics}/network-stats?days=${days}`);
    }

    // ===================
    // LIVE DATA METHODS
    // ===================

    /**
     * Get live gas prices
     */
    async getLiveGas() {
        return await this.makeRequest(`${this.endpoints.live}/gas`);
    }

    /**
     * Get live cryptocurrency prices
     */
    async getLiveCryptoPrices(coins = 'ethereum,bitcoin', currencies = 'usd') {
        return await this.makeRequest(`${this.endpoints.live}/prices?coins=${coins}&currencies=${currencies}`);
    }

    // ===================
    // SCRAPING METHODS
    // ===================

    /**
     * Start a scraping job
     */
    async startScraping(scraper, endpoint, params = {}) {
        return await this.makeRequest(`${this.endpoints.scraping}/start`, {
            method: 'POST',
            body: JSON.stringify({
                scraper,
                endpoint,
                params
            })
        });
    }

    /**
     * Get scraping jobs
     */
    async getScrapingJobs(params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.scraping}/jobs${queryString}`);
    }

    /**
     * Get specific scraping job
     */
    async getScrapingJob(jobId) {
        return await this.makeRequest(`${this.endpoints.scraping}/jobs/${jobId}`);
    }

    /**
     * Get scrapers information
     */
    async getScrapersInfo() {
        return await this.makeRequest('/api/v1/scrapers/info');
    }

    /**
     * Cleanup old data
     */
    async cleanupData(days = 90) {
        return await this.makeRequest(`${this.endpoints.scraping}/cleanup?days=${days}`, {
            method: 'POST'
        });
    }

    // ===================
    // HEALTH CHECK
    // ===================

    /**
     * Check API health
     */
    async healthCheck() {
        return await this.makeRequest('/health');
    }

    /**
     * Get API info
     */
    async getApiInfo() {
        return await this.makeRequest('/');
    }

    // ===================
    // TRACE METHODS
    // ===================

    /**
     * Get transaction trace data
     */
    async getTransactionTrace(hash) {
        return await this.makeRequest(`${this.endpoints.traces}/${hash}`);
    }

    /**
     * Get address trace data
     */
    async getAddressTrace(address, params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.traces}/address/${address}${queryString}`);
    }

    // ===================
    // SPIDER MAP METHODS
    // ===================

    /**
     * Get spider network for address
     */
    async getSpiderNetwork(address, params = {}) {
        const queryString = this.buildQueryString(params);
        return await this.makeRequest(`${this.endpoints.spiderMap}/network/${address}${queryString}`);
    }

    /**
     * Get risk analysis for address
     */
    async getRiskAnalysis(address) {
        return await this.makeRequest(`${this.endpoints.spiderMap}/risk-analysis/${address}`);
    }
}

// Create global instance
window.ethereyeApi = new EthereyeApiClient();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EthereyeApiClient;
}