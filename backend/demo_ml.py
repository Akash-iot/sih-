#!/usr/bin/env python3
"""
Demo of working ETHEREYE ML functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_clustering():
    """Demo DBSCAN clustering"""
    print("🔍 DBSCAN Clustering Demo")
    print("-" * 30)
    
    from ml_models.clustering.dbscan_model import DBSCANClustering
    
    # Generate sample wallet data
    np.random.seed(42)
    print("Generating sample wallet address data...")
    
    wallet_data = pd.DataFrame({
        'total_value': np.random.lognormal(12, 2, 50),  # Total transaction value
        'tx_count': np.random.poisson(25, 50),          # Number of transactions
        'unique_recipients': np.random.randint(1, 30, 50),  # Unique recipients
        'avg_tx_value': np.random.lognormal(8, 1.5, 50),    # Average transaction value
        'days_active': np.random.randint(1, 365, 50)        # Days active
    })
    
    print(f"Generated data for {len(wallet_data)} wallet addresses")
    print(f"Columns: {list(wallet_data.columns)}")
    
    # Cluster the wallet addresses
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCANClustering(eps=0.5, min_samples=3)
    results = clustering.fit(wallet_data)
    
    print(f"✅ Found {results.n_clusters} clusters")
    print(f"✅ {results.n_noise} addresses classified as noise/outliers")
    print(f"✅ Silhouette score: {results.silhouette_score:.3f}")
    
    # Show cluster statistics
    print("\nCluster breakdown:")
    for cluster_id, stats in results.cluster_stats.items():
        if cluster_id == -1:
            print(f"  Noise points: {stats['size']} addresses ({stats['percentage']:.1f}%)")
        else:
            print(f"  Cluster {cluster_id}: {stats['size']} addresses ({stats['percentage']:.1f}%)")
    
    return results

def demo_anomaly_detection():
    """Demo Isolation Forest anomaly detection"""
    print("\n🚨 Isolation Forest Anomaly Detection Demo")
    print("-" * 45)
    
    from ml_models.anomaly_detection.isolation_forest_model import IsolationForestDetector
    
    # Generate sample transaction data
    np.random.seed(42)
    print("Generating sample transaction data...")
    
    # Normal transactions
    normal_txs = pd.DataFrame({
        'value': np.random.lognormal(8, 1.2, 200),       # Normal values
        'gas_used': np.random.normal(25000, 8000, 200),  # Normal gas usage
        'time_interval': np.random.exponential(300, 200) # Normal timing
    })
    
    # Suspicious transactions
    suspicious_txs = pd.DataFrame({
        'value': np.random.lognormal(14, 1, 20),         # Very high values
        'gas_used': np.random.normal(150000, 20000, 20), # Excessive gas
        'time_interval': np.random.exponential(5, 20)    # Rapid fire
    })
    
    # Combine the data
    transaction_data = pd.concat([normal_txs, suspicious_txs]).reset_index(drop=True)
    print(f"Generated {len(transaction_data)} transactions")
    print(f"Expected anomalies: ~{len(suspicious_txs)} suspicious transactions")
    
    # Run anomaly detection
    print("\nRunning Isolation Forest anomaly detection...")
    detector = IsolationForestDetector(contamination=0.1, n_estimators=100)
    results = detector.fit_predict(transaction_data)
    
    print(f"✅ Detected {results.n_anomalies} anomalies")
    print(f"✅ Anomaly rate: {results.anomaly_rate:.1f}%")
    print(f"✅ Mean anomaly score: {results.statistics['mean_anomaly_score']:.3f}")
    
    # Show some anomaly explanations
    explanations = detector.get_anomaly_explanations(transaction_data, top_n=3)
    print("\nTop 3 anomaly explanations:")
    for idx, explanation in list(explanations.items())[:3]:
        print(f"  Transaction {idx}:")
        print(f"    Anomaly score: {explanation['anomaly_score']:.3f}")
        top_feature = explanation['top_contributing_features'][0]
        print(f"    Top factor: {top_feature['feature']} (deviation: {top_feature['deviation']:.2f})")
    
    return results

def demo_risk_assessment():
    """Demo basic risk assessment"""
    print("\n⚖️ Risk Assessment Demo")
    print("-" * 25)
    
    from ml_models.risk_scoring.risk_model import RiskScorer
    
    # Generate sample wallet transaction history
    np.random.seed(42)
    print("Generating sample wallet transaction history...")
    
    wallet_address = "0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d"
    
    # Create different risk scenarios
    scenarios = {
        "Low Risk": pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(30)],
            'value': np.random.lognormal(8, 0.5, 30),       # Consistent values
            'from_address': [wallet_address] * 30,
            'to_address': [f'0x{i:040x}' for i in [1, 2, 3, 4, 5] * 6],  # Regular recipients
            'gas_used': np.random.normal(21000, 2000, 30)
        }),
        "Medium Risk": pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=x*2) for x in range(50)],
            'value': np.random.lognormal(10, 1.5, 50),      # More varied values
            'from_address': [wallet_address] * 50,
            'to_address': [f'0x{i:040x}' for i in np.random.randint(1, 25, 50)],  # Many recipients
            'gas_used': np.random.normal(50000, 15000, 50)
        }),
        "High Risk": pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=x*10) for x in range(100)],
            'value': np.concatenate([
                np.random.lognormal(8, 1, 90),   # Normal values
                np.random.lognormal(15, 1, 10)   # Some very high values
            ]),
            'from_address': [wallet_address] * 100,
            'to_address': [f'0x{i:040x}' for i in np.random.randint(1, 80, 100)],  # High diversity
            'gas_used': np.random.normal(100000, 30000, 100)  # High gas usage
        })
    }
    
    risk_scorer = RiskScorer()
    
    print("\nAssessing risk for different wallet scenarios:")
    for scenario_name, data in scenarios.items():
        print(f"\n  {scenario_name} Wallet:")
        
        # Calculate individual risk components
        tx_risk = risk_scorer.calculate_transaction_risk(data)
        behavioral_risk = risk_scorer.calculate_behavioral_risk(data)
        temporal_risk = risk_scorer.calculate_temporal_risk(data)
        network_risk = risk_scorer.calculate_network_risk(data)
        compliance_risk = risk_scorer.calculate_compliance_risk(data)
        
        # Calculate weighted overall risk
        weights = risk_scorer.feature_weights
        overall_risk = (
            tx_risk * weights.get('transaction', 0.2) +
            behavioral_risk * weights.get('behavioral', 0.15) +
            temporal_risk * weights.get('temporal', 0.1) +
            network_risk * weights.get('network', 0.05) +
            compliance_risk * 0.1  # Simplified weighting
        )
        
        print(f"    Transaction Risk: {tx_risk:.3f}")
        print(f"    Behavioral Risk:  {behavioral_risk:.3f}")
        print(f"    Temporal Risk:    {temporal_risk:.3f}")
        print(f"    Network Risk:     {network_risk:.3f}")
        print(f"    Compliance Risk:  {compliance_risk:.3f}")
        print(f"    Overall Risk:     {overall_risk:.3f}")
        
        # Risk level classification
        if overall_risk >= 0.7:
            risk_level = "HIGH"
        elif overall_risk >= 0.4:
            risk_level = "MEDIUM"
        elif overall_risk >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
            
        print(f"    Risk Level:       {risk_level}")

def main():
    """Run the demo"""
    print("🎯 ETHEREYE ML Pipeline Demo")
    print("=" * 50)
    
    try:
        # Demo clustering
        clustering_results = demo_clustering()
        
        # Demo anomaly detection
        anomaly_results = demo_anomaly_detection()
        
        # Demo risk assessment
        demo_risk_assessment()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nWhat we've demonstrated:")
        print("✅ DBSCAN clustering for wallet grouping")
        print("✅ Isolation Forest for anomaly detection") 
        print("✅ Multi-factor risk assessment")
        print("✅ Configurable risk scoring pipeline")
        
        print("\n📊 Key Results:")
        print(f"• Found {clustering_results.n_clusters} distinct wallet clusters")
        print(f"• Detected {anomaly_results.n_anomalies} anomalous transactions")
        print("• Demonstrated risk assessment across different scenarios")
        
        print("\n🚀 Next Steps:")
        print("• Install NLP dependencies for full functionality:")
        print("  pip install spacy nltk transformers torch")
        print("  python -m spacy download en_core_web_sm")
        print("• Start the FastAPI server to test the API endpoints")
        print("• Use the training scripts to train models on real data")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()