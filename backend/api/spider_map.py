"""
Spider Map API Endpoints
Provides network analysis and transaction relationship mapping
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import random
import hashlib
from datetime import datetime, timedelta

from ..models.database import get_db

router = APIRouter()

@router.get("/network/{address}")
async def get_transaction_network(
    address: str,
    depth: int = Query(2, description="Network depth (1-3)"),
    limit: int = Query(20, description="Maximum nodes to return"),
    risk_analysis: bool = Query(True, description="Include risk analysis")
):
    """
    Generate transaction network map for a given address
    Returns nodes and edges for spider map visualization
    """
    
    # Validate Ethereum address format
    if not address.startswith('0x') or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format")
    
    # Generate network data (in real implementation, this would query blockchain APIs)
    network_data = await generate_transaction_network(address, depth, limit, risk_analysis)
    
    return {
        "center_address": address,
        "network": network_data,
        "metadata": {
            "depth": depth,
            "total_nodes": len(network_data["nodes"]),
            "total_edges": len(network_data["links"]),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "risk_level": calculate_network_risk(network_data["nodes"])
        }
    }

@router.get("/risk-analysis/{address}")
async def get_risk_analysis(address: str):
    """Get detailed risk analysis for an address"""
    
    # Generate risk factors (in real implementation, use ML models)
    risk_data = {
        "address": address,
        "risk_score": random.uniform(0.1, 0.9),
        "risk_level": random.choice(["low", "medium", "high"]),
        "factors": {
            "transaction_frequency": random.uniform(0, 1),
            "large_transactions": random.uniform(0, 1),
            "mixer_interaction": random.uniform(0, 1),
            "blacklist_connections": random.uniform(0, 1),
            "contract_interactions": random.uniform(0, 1)
        },
        "flags": generate_risk_flags(),
        "confidence": random.uniform(0.7, 0.95)
    }
    
    return risk_data

@router.get("/clusters/{address}")
async def get_address_clusters(address: str):
    """Identify address clusters (addresses likely controlled by same entity)"""
    
    clusters = []
    cluster_size = random.randint(2, 8)
    
    for i in range(cluster_size):
        cluster_addr = generate_related_address(address, i)
        clusters.append({
            "address": cluster_addr,
            "confidence": random.uniform(0.6, 0.95),
            "connection_type": random.choice(["change_address", "funding_address", "business_address"]),
            "first_seen": (datetime.utcnow() - timedelta(days=random.randint(1, 365))).isoformat(),
            "transaction_count": random.randint(1, 500)
        })
    
    return {
        "primary_address": address,
        "cluster_id": hashlib.md5(address.encode()).hexdigest()[:16],
        "cluster_addresses": clusters,
        "cluster_confidence": sum(c["confidence"] for c in clusters) / len(clusters) if clusters else 0
    }

@router.get("/flow-analysis/{address}")
async def get_flow_analysis(
    address: str,
    time_window: int = Query(30, description="Time window in days"),
    min_value: float = Query(0.01, description="Minimum transaction value in ETH")
):
    """Analyze transaction flows (money laundering patterns, etc.)"""
    
    flows = []
    flow_count = random.randint(5, 15)
    
    for i in range(flow_count):
        flows.append({
            "flow_id": f"flow_{i}",
            "source": generate_related_address(address, i),
            "destination": generate_related_address(address, i + 100),
            "value_eth": random.uniform(min_value, 100),
            "hops": random.randint(1, 8),
            "timestamps": [
                (datetime.utcnow() - timedelta(days=random.randint(0, time_window))).isoformat()
                for _ in range(random.randint(2, 6))
            ],
            "pattern": random.choice(["direct", "layered", "integrated", "suspicious"]),
            "risk_score": random.uniform(0, 1)
        })
    
    return {
        "address": address,
        "analysis_period": f"{time_window} days",
        "total_flows": len(flows),
        "flows": flows,
        "summary": {
            "total_volume_eth": sum(f["value_eth"] for f in flows),
            "avg_hops": sum(f["hops"] for f in flows) / len(flows) if flows else 0,
            "suspicious_flows": len([f for f in flows if f["pattern"] == "suspicious"]),
            "risk_level": calculate_flow_risk(flows)
        }
    }

# Helper functions

async def generate_transaction_network(address: str, depth: int, limit: int, risk_analysis: bool):
    """Generate realistic transaction network data"""
    
    # Center node
    center_node = {
        "id": address,
        "type": "center",
        "balance": f"{random.uniform(0.1, 100):.4f} ETH",
        "transactions": random.randint(50, 1000),
        "risk": random.choice(["low", "medium", "high"]) if risk_analysis else "unknown",
        "first_seen": (datetime.utcnow() - timedelta(days=random.randint(30, 1000))).isoformat(),
        "last_seen": (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat()
    }
    
    nodes = [center_node]
    links = []
    
    # Connection types with weights
    connection_types = {
        "incoming": 0.3,
        "outgoing": 0.3,
        "contract": 0.2,
        "risky": 0.1,
        "exchange": 0.1
    }
    
    # Generate first level connections
    num_connections = min(limit, random.randint(8, 25))
    
    for i in range(num_connections):
        node_address = generate_related_address(address, i)
        conn_type = random.choices(
            list(connection_types.keys()), 
            weights=list(connection_types.values())
        )[0]
        
        node = {
            "id": node_address,
            "type": conn_type,
            "balance": f"{random.uniform(0.01, 50):.4f} ETH",
            "transactions": random.randint(1, 500),
            "risk": assign_risk_level(conn_type) if risk_analysis else "unknown",
            "value": random.uniform(0.1, 5),
            "first_seen": (datetime.utcnow() - timedelta(days=random.randint(1, 800))).isoformat(),
            "last_seen": (datetime.utcnow() - timedelta(days=random.randint(0, 100))).isoformat()
        }
        
        nodes.append(node)
        
        # Create link
        link = {
            "source": address,
            "target": node_address,
            "value": random.uniform(0.01, 20),
            "type": conn_type,
            "transaction_count": random.randint(1, 50),
            "first_transaction": (datetime.utcnow() - timedelta(days=random.randint(1, 600))).isoformat(),
            "last_transaction": (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat(),
            "total_volume_eth": random.uniform(0.1, 100)
        }
        
        links.append(link)
    
    # Generate second level connections if depth > 1
    if depth > 1 and len(nodes) < limit:
        for i, node in enumerate(nodes[1:min(6, len(nodes))]):  # Connect first few nodes
            if len(nodes) >= limit:
                break
                
            second_level_count = random.randint(1, 4)
            for j in range(second_level_count):
                if len(nodes) >= limit:
                    break
                    
                second_addr = generate_related_address(node["id"], j)
                second_type = random.choice(["secondary", "related", "contract"])
                
                second_node = {
                    "id": second_addr,
                    "type": second_type,
                    "balance": f"{random.uniform(0.001, 10):.4f} ETH",
                    "transactions": random.randint(1, 100),
                    "risk": "low" if risk_analysis else "unknown",
                    "value": random.uniform(0.05, 2)
                }
                
                nodes.append(second_node)
                
                # Link to first level node
                second_link = {
                    "source": node["id"],
                    "target": second_addr,
                    "value": random.uniform(0.01, 5),
                    "type": second_type,
                    "transaction_count": random.randint(1, 20),
                    "total_volume_eth": random.uniform(0.01, 20)
                }
                
                links.append(second_link)
    
    return {
        "nodes": nodes[:limit],
        "links": links
    }

def generate_related_address(base_address: str, seed: int) -> str:
    """Generate a related address based on base address and seed"""
    hash_input = f"{base_address}{seed}".encode()
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    return "0x" + hash_hex[:40]

def assign_risk_level(connection_type: str) -> str:
    """Assign risk level based on connection type"""
    risk_mapping = {
        "risky": "high",
        "exchange": "medium",
        "contract": "low",
        "incoming": "low",
        "outgoing": "low"
    }
    return risk_mapping.get(connection_type, "medium")

def calculate_network_risk(nodes: List[Dict]) -> str:
    """Calculate overall network risk level"""
    risk_scores = {"low": 1, "medium": 2, "high": 3, "unknown": 1}
    total_risk = sum(risk_scores.get(node.get("risk", "unknown"), 1) for node in nodes)
    avg_risk = total_risk / len(nodes) if nodes else 1
    
    if avg_risk >= 2.5:
        return "high"
    elif avg_risk >= 1.8:
        return "medium"
    else:
        return "low"

def generate_risk_flags() -> List[str]:
    """Generate risk flags for address"""
    possible_flags = [
        "high_volume_transactions",
        "mixer_usage",
        "darknet_connections",
        "exchange_clustering",
        "suspicious_timing",
        "round_number_transactions",
        "rapid_succession_txs"
    ]
    
    num_flags = random.randint(0, 3)
    return random.sample(possible_flags, num_flags)

def calculate_flow_risk(flows: List[Dict]) -> str:
    """Calculate risk level for transaction flows"""
    suspicious_flows = len([f for f in flows if f["pattern"] == "suspicious"])
    total_flows = len(flows)
    
    if total_flows == 0:
        return "unknown"
    
    risk_ratio = suspicious_flows / total_flows
    
    if risk_ratio > 0.3:
        return "high"
    elif risk_ratio > 0.1:
        return "medium"
    else:
        return "low"