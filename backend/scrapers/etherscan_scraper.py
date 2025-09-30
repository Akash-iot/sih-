from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from .base_scraper import BaseScraper
from ..config import settings

class EtherscanScraper(BaseScraper):
    """Scraper for Etherscan API to collect Ethereum blockchain data"""
    
    def __init__(self):
        super().__init__("etherscan", settings.ETHERSCAN_BASE_URL)
        self.api_key = settings.ETHERSCAN_API_KEY
        
        if not self.api_key:
            logger.warning("Etherscan API key not provided. Some features may be limited.")
    
    def get_supported_endpoints(self) -> List[str]:
        return [
            "account_balance",
            "account_transactions",
            "token_transfers",
            "contract_source",
            "gas_tracker",
            "latest_blocks",
            "pending_transactions",
            "token_info",
            "transaction_receipt",
            "internal_transactions"
        ]
    
    async def scrape_account_balance(self, address: str) -> Dict:
        """Get ETH balance for an address"""
        params = {
            "module": "account",
            "action": "balance",
            "address": self.format_address(address),
            "tag": "latest",
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("status") == "1":
            balance_wei = int(response.get("result", "0"))
            balance_eth = self.format_amount(balance_wei)
            
            return {
                "address": self.format_address(address),
                "balance_wei": balance_wei,
                "balance_eth": balance_eth,
                "timestamp": datetime.now(),
                "source": "etherscan"
            }
        
        logger.error(f"Failed to get balance for {address}: {response.get('message', 'Unknown error')}")
        return {}
    
    async def scrape_account_transactions(
        self, 
        address: str, 
        start_block: int = 0, 
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """Get transaction history for an address"""
        params = {
            "module": "account",
            "action": "txlist",
            "address": self.format_address(address),
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "desc",
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("status") == "1":
            transactions = response.get("result", [])
            processed_txs = []
            
            for tx in transactions:
                processed_tx = {
                    "hash": tx.get("hash"),
                    "block_number": int(tx.get("blockNumber", 0)),
                    "timestamp": datetime.fromtimestamp(int(tx.get("timeStamp", 0))),
                    "from_address": self.format_address(tx.get("from", "")),
                    "to_address": self.format_address(tx.get("to", "")),
                    "value_wei": int(tx.get("value", "0")),
                    "value_eth": self.format_amount(tx.get("value", "0")),
                    "gas_price": int(tx.get("gasPrice", "0")),
                    "gas_used": int(tx.get("gasUsed", "0")),
                    "gas_limit": int(tx.get("gas", "0")),
                    "tx_fee_wei": int(tx.get("gasUsed", "0")) * int(tx.get("gasPrice", "0")),
                    "tx_fee_eth": self.format_amount(str(int(tx.get("gasUsed", "0")) * int(tx.get("gasPrice", "0")))),
                    "is_error": tx.get("isError") == "1",
                    "transaction_index": int(tx.get("transactionIndex", 0)),
                    "input": tx.get("input", ""),
                    "method_id": tx.get("input", "")[:10] if tx.get("input", "").startswith("0x") else "",
                    "source": "etherscan"
                }
                
                if self.validate_data(processed_tx, ["hash", "from_address", "to_address"]):
                    processed_txs.append(processed_tx)
            
            return processed_txs
        
        logger.error(f"Failed to get transactions for {address}: {response.get('message', 'Unknown error')}")
        return []
    
    async def scrape_token_transfers(
        self, 
        address: str, 
        contract_address: Optional[str] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """Get ERC-20 token transfer events for an address"""
        action = "tokentx" if not contract_address else "tokentx"
        
        params = {
            "module": "account",
            "action": action,
            "address": self.format_address(address),
            "page": page,
            "offset": offset,
            "sort": "desc",
            "apikey": self.api_key
        }
        
        if contract_address:
            params["contractaddress"] = self.format_address(contract_address)
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("status") == "1":
            transfers = response.get("result", [])
            processed_transfers = []
            
            for transfer in transfers:
                processed_transfer = {
                    "hash": transfer.get("hash"),
                    "block_number": int(transfer.get("blockNumber", 0)),
                    "timestamp": datetime.fromtimestamp(int(transfer.get("timeStamp", 0))),
                    "from_address": self.format_address(transfer.get("from", "")),
                    "to_address": self.format_address(transfer.get("to", "")),
                    "contract_address": self.format_address(transfer.get("contractAddress", "")),
                    "value": int(transfer.get("value", "0")),
                    "token_name": transfer.get("tokenName", ""),
                    "token_symbol": transfer.get("tokenSymbol", ""),
                    "token_decimal": int(transfer.get("tokenDecimal", "18")),
                    "transaction_index": int(transfer.get("transactionIndex", 0)),
                    "gas": int(transfer.get("gas", "0")),
                    "gas_price": int(transfer.get("gasPrice", "0")),
                    "gas_used": int(transfer.get("gasUsed", "0")),
                    "source": "etherscan"
                }
                
                # Calculate human-readable value
                processed_transfer["value_formatted"] = self.format_amount(
                    transfer.get("value", "0"), 
                    int(transfer.get("tokenDecimal", "18"))
                )
                
                if self.validate_data(processed_transfer, ["hash", "contract_address"]):
                    processed_transfers.append(processed_transfer)
            
            return processed_transfers
        
        logger.error(f"Failed to get token transfers for {address}: {response.get('message', 'Unknown error')}")
        return []
    
    async def scrape_gas_tracker(self) -> Dict:
        """Get current gas prices"""
        params = {
            "module": "gastracker",
            "action": "gasoracle",
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("status") == "1":
            gas_data = response.get("result", {})
            return {
                "safe_gas_price": int(gas_data.get("SafeGasPrice", 0)),
                "standard_gas_price": int(gas_data.get("StandardGasPrice", 0)),
                "fast_gas_price": int(gas_data.get("FastGasPrice", 0)),
                "suggested_base_fee": float(gas_data.get("suggestBaseFee", 0)),
                "gas_used_ratio": gas_data.get("gasUsedRatio", ""),
                "timestamp": datetime.now(),
                "source": "etherscan"
            }
        
        return {}
    
    async def scrape_latest_blocks(self, count: int = 10) -> List[Dict]:
        """Get latest blocks"""
        # Note: Etherscan doesn't have a direct API for latest blocks
        # We'll get latest block number and then fetch individual blocks
        params = {
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("result"):
            latest_block = int(response["result"], 16)
            blocks = []
            
            # Get last 'count' blocks
            for i in range(count):
                block_number = latest_block - i
                block_data = await self.scrape_block_by_number(block_number)
                if block_data:
                    blocks.append(block_data)
            
            return blocks
        
        return []
    
    async def scrape_block_by_number(self, block_number: int) -> Dict:
        """Get block data by number"""
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "true",
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("result"):
            block = response["result"]
            return {
                "block_number": int(block.get("number", "0"), 16),
                "block_hash": block.get("hash"),
                "parent_hash": block.get("parentHash"),
                "timestamp": datetime.fromtimestamp(int(block.get("timestamp", "0"), 16)),
                "gas_limit": int(block.get("gasLimit", "0"), 16),
                "gas_used": int(block.get("gasUsed", "0"), 16),
                "miner": self.format_address(block.get("miner", "")),
                "difficulty": int(block.get("difficulty", "0"), 16),
                "total_difficulty": block.get("totalDifficulty"),
                "size": int(block.get("size", "0"), 16),
                "transaction_count": len(block.get("transactions", [])),
                "base_fee_per_gas": int(block.get("baseFeePerGas", "0"), 16) if block.get("baseFeePerGas") else 0,
                "source": "etherscan"
            }
        
        return {}
    
    async def scrape_transaction_receipt(self, tx_hash: str) -> Dict:
        """Get transaction receipt"""
        params = {
            "module": "proxy",
            "action": "eth_getTransactionReceipt",
            "txhash": tx_hash,
            "apikey": self.api_key
        }
        
        response = await self.make_request(self.base_url, params=params)
        
        if response.get("result"):
            receipt = response["result"]
            return {
                "transaction_hash": receipt.get("transactionHash"),
                "block_number": int(receipt.get("blockNumber", "0"), 16),
                "block_hash": receipt.get("blockHash"),
                "transaction_index": int(receipt.get("transactionIndex", "0"), 16),
                "from_address": self.format_address(receipt.get("from", "")),
                "to_address": self.format_address(receipt.get("to", "")),
                "gas_used": int(receipt.get("gasUsed", "0"), 16),
                "cumulative_gas_used": int(receipt.get("cumulativeGasUsed", "0"), 16),
                "effective_gas_price": int(receipt.get("effectiveGasPrice", "0"), 16) if receipt.get("effectiveGasPrice") else 0,
                "status": receipt.get("status") == "0x1",
                "logs": receipt.get("logs", []),
                "logs_bloom": receipt.get("logsBloom"),
                "contract_address": self.format_address(receipt.get("contractAddress", "")) if receipt.get("contractAddress") else None,
                "source": "etherscan"
            }
        
        return {}
    
    async def scrape(self, endpoint: str, **kwargs) -> List[Dict]:
        """Main scraping method"""
        try:
            if endpoint == "account_balance":
                result = await self.scrape_account_balance(kwargs.get("address"))
                return [result] if result else []
            
            elif endpoint == "account_transactions":
                return await self.scrape_account_transactions(
                    kwargs.get("address"),
                    kwargs.get("start_block", 0),
                    kwargs.get("end_block", 99999999),
                    kwargs.get("page", 1),
                    kwargs.get("offset", 100)
                )
            
            elif endpoint == "token_transfers":
                return await self.scrape_token_transfers(
                    kwargs.get("address"),
                    kwargs.get("contract_address"),
                    kwargs.get("page", 1),
                    kwargs.get("offset", 100)
                )
            
            elif endpoint == "gas_tracker":
                result = await self.scrape_gas_tracker()
                return [result] if result else []
            
            elif endpoint == "latest_blocks":
                return await self.scrape_latest_blocks(kwargs.get("count", 10))
            
            elif endpoint == "transaction_receipt":
                result = await self.scrape_transaction_receipt(kwargs.get("tx_hash"))
                return [result] if result else []
            
            else:
                logger.error(f"Unsupported endpoint: {endpoint}")
                return []
                
        except Exception as e:
            logger.error(f"Scraping failed for endpoint {endpoint}: {str(e)}")
            return []