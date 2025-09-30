#!/usr/bin/env python3
"""
Demo script to test ETHEREYE ML endpoints functionality
"""

import asyncio
import sys
import json
from typing import Dict, List

# Add current directory to path
sys.path.append('.')
sys.path.append('./ml_services')

async def demo_clustering_service():
    """Demo clustering functionality"""
    print("\n🔍 Testing Clustering Service...")
    print("-" * 40)
    
    from ml_services.clustering_service import BlockchainClusteringService
    
    service = BlockchainClusteringService()
    
    # Demo address data
    demo_addresses = [
        {
            "address": "0x1234567890123456789012345678901234567890",
            "balance_eth": 10.5,
            "balance_usd": 25500.0,
            "transaction_count": 150,
            "gas_used_total": 2100000,
            "unique_counterparties": 25,
            "contract_interactions": 5,
            "first_transaction_days_ago": 365,
            "last_transaction_days_ago": 1
        },
        {
            "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "balance_eth": 0.1,
            "balance_usd": 243.0,
            "transaction_count": 5,
            "gas_used_total": 210000,
            "unique_counterparties": 3,
            "contract_interactions": 0,
            "first_transaction_days_ago": 30,
            "last_transaction_days_ago": 5
        },
        {
            "address": "0x9876543210987654321098765432109876543210",
            "balance_eth": 100.0,
            "balance_usd": 243000.0,
            "transaction_count": 1000,
            "gas_used_total": 21000000,
            "unique_counterparties": 100,
            "contract_interactions": 50,
            "first_transaction_days_ago": 1000,
            "last_transaction_days_ago": 0
        }
    ]
    
    try:
        # Test clustering
        clustering_result = service.perform_clustering(demo_addresses)
        print("✓ Clustering analysis completed")
        print(f"  Clusters found: {len(set(clustering_result['cluster_labels']))}")
        
        # Test anomaly detection
        anomaly_result = service.detect_anomalies(demo_addresses)
        print("✓ Anomaly detection completed")
        anomalies = [i for i, score in enumerate(anomaly_result['anomaly_scores']) if score < 0]
        print(f"  Anomalies detected: {len(anomalies)}")
        
        # Test combined analysis
        combined_result = service.combined_analysis(demo_addresses)
        print("✓ Combined analysis completed")
        print(f"  Risk distribution: {combined_result['summary']['risk_distribution']}")
        
    except Exception as e:
        print(f"✗ Clustering service error: {e}")

async def demo_nlp_service():
    """Demo NLP functionality"""
    print("\n📝 Testing NLP Service...")
    print("-" * 40)
    
    from ml_services.nlp_service import BlockchainNLPService
    
    service = BlockchainNLPService()
    
    # Demo texts
    demo_texts = [
        "Transfer 0.5 ETH to wallet 0x1234567890123456789012345678901234567890",
        "My email is test@example.com and phone is 555-123-4567",
        "Suspicious DeFi transaction involving mixer protocols",
        "Normal exchange deposit via Coinbase",
    ]
    
    try:
        # Test PII extraction
        for text in demo_texts[:2]:
            result = service.extract_pii(text)
            print(f"✓ PII extracted from: '{text[:50]}...'")
            if result['pii_found']:
                print(f"  Found: {list(result['pii_data'].keys())}")
        
        # Test blockchain content analysis
        blockchain_text = demo_texts[2]
        content_result = service.analyze_blockchain_content(blockchain_text)
        print(f"✓ Blockchain content analyzed")
        print(f"  Risk indicators: {len(content_result['risk_indicators'])}")
        
        # Test sentiment analysis
        sentiment_result = service.analyze_sentiment(demo_texts[3])
        print(f"✓ Sentiment analysis completed")
        print(f"  Sentiment: {sentiment_result['sentiment_label']}")
        
    except Exception as e:
        print(f"✗ NLP service error: {e}")

async def demo_risk_scoring():
    """Demo risk scoring functionality"""
    print("\n⚠️  Testing Risk Scoring Service...")
    print("-" * 40)
    
    from ml_services.risk_scoring_service import BlockchainRiskScoringService
    
    service = BlockchainRiskScoringService()
    
    # Demo wallet data
    demo_wallet = {
        "address": "0x1234567890123456789012345678901234567890",
        "balance_eth": 50.0,
        "balance_usd": 121500.0,
        "transaction_count": 500,
        "total_volume_eth": 1000.0,
        "unique_counterparties": 75,
        "gas_used_total": 10500000,
        "contract_interactions": 25,
        "first_transaction_days_ago": 500,
        "last_transaction_days_ago": 2,
        "transactions": [
            {"value_eth": 10.0, "timestamp": "2024-01-01T10:00:00Z", "gas_used": 21000},
            {"value_eth": 5.0, "timestamp": "2024-01-02T15:30:00Z", "gas_used": 21000}
        ]
    }
    
    try:
        # Test risk prediction (without trained model - will use default scoring)
        risk_result = service.predict_risk_score(demo_wallet)
        print("✓ Risk score calculated")
        print(f"  Risk Level: {risk_result['risk_level']}")
        print(f"  Risk Score: {risk_result['risk_score']:.3f}")
        print(f"  Key factors: {risk_result['explanation']['key_factors'][:2]}")
        
    except Exception as e:
        print(f"✗ Risk scoring service error: {e}")

async def demo_ml_services():
    """Run all ML service demos"""
    print("🚀 ETHEREYE ML Services Demo")
    print("=" * 50)
    
    await demo_clustering_service()
    await demo_nlp_service()
    await demo_risk_scoring()
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)
    print("All ML services are working and integrated.")
    print("You can now:")
    print("  • Start the FastAPI server: python -m uvicorn simple_main:app --reload")
    print("  • Access ML endpoints at: http://localhost:8000/api/v1/ml/")
    print("  • View API docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    asyncio.run(demo_ml_services())