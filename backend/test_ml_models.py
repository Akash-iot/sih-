#!/usr/bin/env python3
"""
Quick test script for ETHEREYE ML models
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_clustering():
    """Test DBSCAN clustering"""
    print("🔍 Testing DBSCAN Clustering...")
    
    try:
        from ml_models.clustering.dbscan_model import DBSCANClustering
        
        # Generate sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'value': np.random.lognormal(10, 2, 100),
            'tx_count': np.random.poisson(10, 100),
            'avg_time_between_txs': np.random.exponential(3600, 100),
            'unique_counterparties': np.random.randint(1, 50, 100)
        })
        
        # Test clustering
        clustering = DBSCANClustering(eps=0.5, min_samples=5)
        results = clustering.fit(data)
        
        print(f"   ✅ Found {results.n_clusters} clusters")
        print(f"   ✅ Noise points: {results.n_noise}")
        print(f"   ✅ Silhouette score: {results.silhouette_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Clustering test failed: {e}")
        return False

def test_anomaly_detection():
    """Test Isolation Forest anomaly detection"""
    print("🚨 Testing Isolation Forest Anomaly Detection...")
    
    try:
        from ml_models.anomaly_detection.isolation_forest_model import IsolationForestDetector
        
        # Generate sample transaction data
        np.random.seed(42)
        normal_data = pd.DataFrame({
            'value': np.random.lognormal(8, 1, 90),
            'gas_used': np.random.normal(21000, 5000, 90),
            'time_diff': np.random.exponential(300, 90)
        })
        
        # Add some anomalous data
        anomaly_data = pd.DataFrame({
            'value': np.random.lognormal(15, 1, 10),  # Much higher values
            'gas_used': np.random.normal(100000, 10000, 10),  # High gas usage
            'time_diff': np.random.exponential(10, 10)  # Very fast transactions
        })
        
        data = pd.concat([normal_data, anomaly_data]).reset_index(drop=True)
        
        # Test anomaly detection
        detector = IsolationForestDetector(contamination=0.1)
        results = detector.fit_predict(data)
        
        print(f"   ✅ Detected {results.n_anomalies} anomalies")
        print(f"   ✅ Anomaly rate: {results.anomaly_rate:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Anomaly detection test failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("⚙️ Testing Data Preprocessing...")
    
    try:
        from ml_models.preprocessing.data_processor import DataPreprocessor
        
        # Generate sample transaction data
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(100)],
            'from_address': [f'0x{i:040x}' for i in np.random.randint(1, 50, 100)],
            'to_address': [f'0x{i:040x}' for i in np.random.randint(1, 50, 100)],
            'value': np.random.lognormal(10, 2, 100),
            'gas_used': np.random.normal(21000, 5000, 100),
            'gas_price': np.random.normal(20, 5, 100)
        })
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_transaction_data(data)
        
        print(f"   ✅ Original columns: {len(data.columns)}")
        print(f"   ✅ Processed columns: {len(processed_data.columns)}")
        print(f"   ✅ Added features: {len(processed_data.columns) - len(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Preprocessing test failed: {e}")
        return False

def test_risk_scoring():
    """Test risk scoring"""
    print("⚖️ Testing Risk Scoring...")
    
    try:
        from ml_models.risk_scoring.risk_model import RiskScorer
        
        # Generate sample wallet data
        np.random.seed(42)
        wallet_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=x) for x in range(50)],
            'from_address': ['0x123' for _ in range(50)],
            'to_address': [f'0x{i:040x}' for i in np.random.randint(1, 20, 50)],
            'value': np.random.lognormal(10, 1.5, 50),
            'gas_used': np.random.normal(21000, 3000, 50)
        })
        
        # Test risk assessment (without initializing all models to save time)
        risk_scorer = RiskScorer()
        
        # Test individual risk calculations
        tx_risk = risk_scorer.calculate_transaction_risk(wallet_data)
        behavioral_risk = risk_scorer.calculate_behavioral_risk(wallet_data)
        temporal_risk = risk_scorer.calculate_temporal_risk(wallet_data)
        
        print(f"   ✅ Transaction risk: {tx_risk:.3f}")
        print(f"   ✅ Behavioral risk: {behavioral_risk:.3f}")
        print(f"   ✅ Temporal risk: {temporal_risk:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Risk scoring test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ETHEREYE ML Models Test Suite")
    print("=" * 50)
    
    tests = [
        test_preprocessing,
        test_clustering,
        test_anomaly_detection,
        test_risk_scoring,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML pipeline is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)