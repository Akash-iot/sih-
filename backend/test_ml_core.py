#!/usr/bin/env python3
"""
Core ML functionality test (without NLP dependencies)
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
        
        # Test parameter optimization
        print("   🔧 Testing parameter optimization...")
        opt_results = clustering.optimize_parameters(data, eps_range=[0.3, 0.5, 0.7])
        print(f"   ✅ Best parameters: eps={opt_results['best_params']['eps']}")
        
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
        
        # Test explanation
        print("   🔍 Testing anomaly explanations...")
        explanations = detector.get_anomaly_explanations(data, top_n=3)
        print(f"   ✅ Generated explanations for {len(explanations)} anomalies")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Anomaly detection test failed: {e}")
        return False

def test_preprocessing():
    """Test basic data preprocessing (without NLP dependencies)"""
    print("⚙️ Testing Basic Data Preprocessing...")
    
    try:
        from ml_models.preprocessing.data_processor import DataPreprocessor
        
        # Generate sample transaction data
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(100)],
            'value': np.random.lognormal(10, 2, 100),
            'gas_used': np.random.normal(21000, 5000, 100),
            'gas_price': np.random.normal(20, 5, 100)
        })
        
        # Test basic preprocessing functions
        preprocessor = DataPreprocessor()
        
        # Test feature importance calculation
        importance_scores = preprocessor.get_feature_importance_scores(data.select_dtypes(include=[np.number]))
        print(f"   ✅ Calculated importance for {len(importance_scores)} features")
        
        # Test sliding window features
        window_data = preprocessor.create_sliding_window_features(
            data, 'timestamp', ['value'], window_sizes=[7, 30]
        )
        print(f"   ✅ Created sliding window features: {len(window_data.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Preprocessing test failed: {e}")
        return False

def test_risk_scoring_core():
    """Test core risk scoring without NLP"""
    print("⚖️ Testing Core Risk Scoring...")
    
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
        
        # Test individual risk calculations
        risk_scorer = RiskScorer()
        
        tx_risk = risk_scorer.calculate_transaction_risk(wallet_data)
        behavioral_risk = risk_scorer.calculate_behavioral_risk(wallet_data)
        temporal_risk = risk_scorer.calculate_temporal_risk(wallet_data)
        network_risk = risk_scorer.calculate_network_risk(wallet_data)
        compliance_risk = risk_scorer.calculate_compliance_risk(wallet_data)
        
        print(f"   ✅ Transaction risk: {tx_risk:.3f}")
        print(f"   ✅ Behavioral risk: {behavioral_risk:.3f}")
        print(f"   ✅ Temporal risk: {temporal_risk:.3f}")
        print(f"   ✅ Network risk: {network_risk:.3f}")
        print(f"   ✅ Compliance risk: {compliance_risk:.3f}")
        
        # Test model info
        model_info = risk_scorer.get_model_info()
        print(f"   ✅ Model info retrieved: {model_info['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Risk scoring test failed: {e}")
        return False

def test_model_utils():
    """Test model utilities"""
    print("🔧 Testing Model Utilities...")
    
    try:
        from ml_models.utils.model_utils import ModelUtils
        import tempfile
        
        # Test data validation
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(10)],
            'from_address': [f'0x{i:040x}' for i in range(10)],
            'to_address': [f'0x{i:040x}' for i in range(10, 20)],
            'value': np.random.lognormal(8, 1, 10)
        })
        
        validation_result = ModelUtils.validate_data_schema(
            sample_data, 
            required_columns=['timestamp', 'from_address', 'to_address', 'value']
        )
        print(f"   ✅ Data validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
        
        # Test data profiling
        profile = ModelUtils.generate_data_profile(sample_data)
        print(f"   ✅ Data profile generated with {len(profile['columns'])} column profiles")
        
        # Test model directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = ModelUtils.create_model_directory(temp_dir, "test_model")
            print(f"   ✅ Model directory created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model utilities test failed: {e}")
        return False

def main():
    """Run core ML tests"""
    print("🧪 ETHEREYE Core ML Models Test Suite")
    print("=" * 50)
    
    tests = [
        test_preprocessing,
        test_clustering,
        test_anomaly_detection,
        test_risk_scoring_core,
        test_model_utils,
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
        print("🎉 All core ML tests passed! Pipeline is working correctly.")
        print("\n💡 Note: NLP features require additional dependencies:")
        print("   pip install spacy nltk transformers torch")
        print("   python -m spacy download en_core_web_sm")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)