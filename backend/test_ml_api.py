#!/usr/bin/env python3
"""
Test script for ETHEREYE ML API endpoints
=========================================

This script demonstrates how to use the ML API endpoints for:
- Clustering wallets
- Detecting anomalous transactions
- Risk assessment
- Text analysis
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp

# API base URL
API_BASE = "http://localhost:8000/api/v1"

def test_health_check():
    """Test ML service health check"""
    print("🔍 Testing ML Health Check...")
    
    try:
        response = requests.get(f"{API_BASE}/ml/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ ML services are healthy")
            print(f"   Status: {health_data['status']}")
            print(f"   sklearn: {health_data['ml_libraries']['sklearn']}")
            print(f"   pandas: {health_data['ml_libraries']['pandas']}")
            print(f"   numpy: {health_data['ml_libraries']['numpy']}")
            print(f"   NLP available: {health_data['nlp_available']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_clustering():
    """Test wallet clustering endpoint"""
    print("\n🔍 Testing Wallet Clustering...")
    
    # Generate sample wallet data
    np.random.seed(42)
    wallets = []
    
    for i in range(20):
        wallet = {
            "address": f"0x{'%040x' % i}",
            "total_value": float(np.random.lognormal(12, 2)),
            "tx_count": int(np.random.poisson(25)),
            "unique_recipients": int(np.random.randint(1, 30)),
            "avg_tx_value": float(np.random.lognormal(8, 1.5)),
            "days_active": int(np.random.randint(1, 365))
        }
        wallets.append(wallet)
    
    payload = {
        "wallets": wallets,
        "eps": 0.5,
        "min_samples": 3
    }
    
    try:
        response = requests.post(f"{API_BASE}/ml/cluster", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Clustering completed")
            print(f"   Clusters found: {result['n_clusters']}")
            print(f"   Noise points: {result['n_noise']}")
            print(f"   Silhouette score: {result['silhouette_score']:.3f}")
            
            # Show some cluster assignments
            print("   Sample cluster assignments:")
            for i, (address, cluster_id) in enumerate(list(result['cluster_assignments'].items())[:5]):
                print(f"     {address[:8]}... -> Cluster {cluster_id}")
            
            return True
        else:
            print(f"❌ Clustering failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Clustering error: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection endpoint"""
    print("\n🚨 Testing Anomaly Detection...")
    
    # Generate sample transaction data
    np.random.seed(42)
    transactions = []
    
    # Normal transactions
    for i in range(30):
        tx = {
            "tx_hash": f"0x{'%064x' % i}",
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "from_address": "0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d",
            "to_address": f"0x{'%040x' % (i % 10)}",
            "value": float(np.random.lognormal(8, 1.2)),
            "gas_used": int(np.random.normal(25000, 8000)),
            "gas_price": float(np.random.normal(20, 5))
        }
        transactions.append(tx)
    
    # Add some suspicious transactions
    for i in range(30, 35):
        tx = {
            "tx_hash": f"0x{'%064x' % i}",
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "from_address": "0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d",
            "to_address": f"0x{'%040x' % i}",
            "value": float(np.random.lognormal(14, 1)),  # Very high values
            "gas_used": int(np.random.normal(150000, 20000)),  # High gas
            "gas_price": float(np.random.normal(100, 20))  # High gas price
        }
        transactions.append(tx)
    
    payload = {
        "transactions": transactions,
        "contamination": 0.15,
        "n_estimators": 100
    }
    
    try:
        response = requests.post(f"{API_BASE}/ml/anomaly-detection", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Anomaly detection completed")
            print(f"   Anomalies detected: {result['n_anomalies']}")
            print(f"   Anomaly rate: {result['anomaly_rate']:.1f}%")
            print(f"   Mean anomaly score: {result['mean_anomaly_score']:.3f}")
            
            # Show top anomalies
            if result['anomalous_transactions']:
                print("   Top anomalous transactions:")
                for tx in result['anomalous_transactions'][:3]:
                    print(f"     {tx['tx_hash'][:10]}... Score: {tx['anomaly_score']:.3f}, Value: ${tx['value']:,.2f}")
            
            return True
        else:
            print(f"❌ Anomaly detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Anomaly detection error: {e}")
        return False

def test_risk_assessment():
    """Test wallet risk assessment endpoint"""
    print("\n⚖️ Testing Risk Assessment...")
    
    # Generate sample wallet transaction history
    np.random.seed(42)
    wallet_address = "0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d"
    
    transactions = []
    for i in range(50):
        tx = {
            "tx_hash": f"0x{'%064x' % i}",
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "from_address": wallet_address,
            "to_address": f"0x{'%040x' % (i % 20)}",
            "value": float(np.random.lognormal(10, 1.5)),
            "gas_used": int(np.random.normal(50000, 15000))
        }
        transactions.append(tx)
    
    payload = {
        "wallet_address": wallet_address,
        "transactions": transactions,
        "include_nlp": False
    }
    
    try:
        response = requests.post(f"{API_BASE}/ml/risk-assessment", json=payload)
        if response.status_code == 200:
            result = response.json()
            risk_score = result['risk_score']
            
            print(f"✅ Risk assessment completed")
            print(f"   Wallet: {result['wallet_address'][:10]}...")
            print(f"   Risk Level: {risk_score['risk_level']}")
            print(f"   Overall Risk: {risk_score['overall_risk']:.3f}")
            print(f"   Confidence: {risk_score['confidence']:.3f}")
            
            print("   Risk Components:")
            print(f"     Transaction Risk: {risk_score['transaction_risk']:.3f}")
            print(f"     Behavioral Risk:  {risk_score['behavioral_risk']:.3f}")
            print(f"     Temporal Risk:    {risk_score['temporal_risk']:.3f}")
            print(f"     Network Risk:     {risk_score['network_risk']:.3f}")
            print(f"     Compliance Risk:  {risk_score['compliance_risk']:.3f}")
            
            print("   Contributing Factors:")
            for factor in result['contributing_factors']:
                print(f"     • {factor}")
            
            print("   Recommendations:")
            for rec in result['recommendations']:
                print(f"     • {rec}")
            
            return True
        else:
            print(f"❌ Risk assessment failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Risk assessment error: {e}")
        return False

def test_text_analysis():
    """Test text analysis endpoint"""
    print("\n📝 Testing Text Analysis...")
    
    # Sample texts with various content
    texts = [
        "Check out this great new DeFi protocol! Amazing profits await! 🚀",
        "SCAM ALERT: This project is a rugpull scheme. Stay away!",
        "My wallet address is 0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d for payments",
        "Contact me at john.doe@example.com for crypto trading opportunities",
        "This looks like a phishing attempt, very suspicious behavior detected",
        "Normal discussion about blockchain technology and smart contracts"
    ]
    
    payload = {
        "texts": texts,
        "include_pii": True,
        "include_sentiment": True,
        "include_risk_keywords": True
    }
    
    try:
        response = requests.post(f"{API_BASE}/ml/text-analysis", json=payload)
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Text analysis completed")
            print(f"   Texts analyzed: {len(texts)}")
            print(f"   PII entities found: {len(result['pii_detected'])}")
            print(f"   Risk keywords found: {len(result['risk_keywords'])}")
            print(f"   Crypto relevance: {result['crypto_relevance_score']:.3f}")
            
            if result['pii_detected']:
                print("   PII Detected:")
                for pii in result['pii_detected'][:3]:
                    print(f"     • {pii['entity_type']}: {pii['entity_text'][:20]}...")
            
            if result['risk_keywords']:
                print(f"   Risk Keywords: {', '.join(result['risk_keywords'])}")
            
            if result['sentiment_scores']:
                print("   Sentiment Analysis:")
                for i, sentiment in enumerate(result['sentiment_scores'][:3]):
                    sentiment_label = "Positive" if sentiment['sentiment'] > 0.6 else "Negative" if sentiment['sentiment'] < 0.4 else "Neutral"
                    print(f"     Text {i+1}: {sentiment_label} ({sentiment['sentiment']:.3f})")
            
            return True
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text analysis error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🎯 ETHEREYE ML API Tests")
    print("=" * 40)
    print("Testing ML endpoints with sample data...")
    print("Make sure your FastAPI server is running on localhost:8000")
    print()
    
    tests = [
        ("Health Check", test_health_check),
        ("Clustering", test_clustering),
        ("Anomaly Detection", test_anomaly_detection),
        ("Risk Assessment", test_risk_assessment),
        ("Text Analysis", test_text_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("🏁 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ML API is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the server logs and ensure all dependencies are installed.")
    
    print("\n🚀 Next Steps:")
    print("• Try the interactive API docs at http://localhost:8000/docs")
    print("• Install NLP dependencies for full text analysis: pip install spacy nltk transformers")
    print("• Integrate these endpoints into your frontend application")

if __name__ == "__main__":
    main()