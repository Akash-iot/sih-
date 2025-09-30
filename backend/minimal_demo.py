#!/usr/bin/env python3
"""
Minimal ETHEREYE ML Demo - No Custom Modules
Direct demonstration of core ML functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def demo_direct_clustering():
    """Direct DBSCAN demo without custom modules"""
    print("🔍 Direct DBSCAN Clustering Demo")
    print("-" * 35)
    
    # Generate sample wallet data
    np.random.seed(42)
    print("Generating sample wallet data...")
    
    data = {
        'total_value': np.random.lognormal(12, 2, 50),
        'tx_count': np.random.poisson(25, 50),
        'unique_recipients': np.random.randint(1, 30, 50),
        'avg_tx_value': np.random.lognormal(8, 1.5, 50),
        'days_active': np.random.randint(1, 365, 50)
    }
    
    wallet_data = pd.DataFrame(data)
    print(f"Generated {len(wallet_data)} wallet addresses")
    print(f"Features: {list(wallet_data.columns)}")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(wallet_data)
    
    # Apply DBSCAN
    print("\nRunning DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(scaled_data)
    
    # Calculate results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate silhouette score if we have clusters
    if n_clusters > 1 and n_noise < len(labels) - 1:
        silhouette = silhouette_score(scaled_data, labels)
    else:
        silhouette = -1
    
    print(f"✅ Found {n_clusters} clusters")
    print(f"✅ {n_noise} addresses classified as noise/outliers")
    print(f"✅ Silhouette score: {silhouette:.3f}")
    
    # Show cluster breakdown
    print("\nCluster breakdown:")
    unique_labels = set(labels)
    for label in unique_labels:
        count = list(labels).count(label)
        percentage = (count / len(labels)) * 100
        if label == -1:
            print(f"  Noise points: {count} addresses ({percentage:.1f}%)")
        else:
            print(f"  Cluster {label}: {count} addresses ({percentage:.1f}%)")
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette,
        'labels': labels
    }

def demo_direct_anomaly_detection():
    """Direct Isolation Forest demo without custom modules"""
    print("\n🚨 Direct Isolation Forest Anomaly Detection Demo")
    print("-" * 50)
    
    # Generate sample transaction data
    np.random.seed(42)
    print("Generating sample transaction data...")
    
    # Normal transactions
    normal_data = {
        'value': np.random.lognormal(8, 1.2, 200),
        'gas_used': np.random.normal(25000, 8000, 200),
        'time_interval': np.random.exponential(300, 200)
    }
    
    # Suspicious transactions
    suspicious_data = {
        'value': np.random.lognormal(14, 1, 20),
        'gas_used': np.random.normal(150000, 20000, 20),
        'time_interval': np.random.exponential(5, 20)
    }
    
    # Combine data
    all_data = {}
    for key in normal_data:
        all_data[key] = np.concatenate([normal_data[key], suspicious_data[key]])
    
    transaction_data = pd.DataFrame(all_data)
    print(f"Generated {len(transaction_data)} transactions")
    print(f"Expected anomalies: ~20 suspicious transactions")
    
    # Apply Isolation Forest
    print("\nRunning Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    anomaly_labels = iso_forest.fit_predict(transaction_data)
    anomaly_scores = iso_forest.decision_function(transaction_data)
    
    # Calculate results
    n_anomalies = list(anomaly_labels).count(-1)
    anomaly_rate = (n_anomalies / len(transaction_data)) * 100
    mean_score = np.mean(anomaly_scores)
    
    print(f"✅ Detected {n_anomalies} anomalies")
    print(f"✅ Anomaly rate: {anomaly_rate:.1f}%")
    print(f"✅ Mean anomaly score: {mean_score:.3f}")
    
    # Show some anomaly details
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    print(f"\nTop 3 anomalous transactions:")
    sorted_indices = sorted(anomaly_indices, key=lambda i: anomaly_scores[i])[:3]
    
    for i, idx in enumerate(sorted_indices):
        tx = transaction_data.iloc[idx]
        score = anomaly_scores[idx]
        print(f"  Transaction {idx}:")
        print(f"    Anomaly score: {score:.3f}")
        print(f"    Value: ${tx['value']:,.2f}")
        print(f"    Gas used: {tx['gas_used']:,.0f}")
        print(f"    Time interval: {tx['time_interval']:.1f} seconds")
    
    return {
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
        'mean_score': mean_score
    }

def demo_direct_risk_scoring():
    """Direct risk scoring demo without custom modules"""
    print("\n⚖️ Direct Risk Scoring Demo")
    print("-" * 30)
    
    # Generate wallet scenarios
    np.random.seed(42)
    wallet_address = "0x742d35cc1ba1e7d796d0e9b3dc4c86d4ef67bd3d"
    
    scenarios = {
        "Low Risk Wallet": {
            'tx_count': 25,
            'total_value': 10000,
            'avg_tx_value': 400,
            'unique_recipients': 5,
            'rapid_tx_ratio': 0.02,
            'large_tx_ratio': 0.01,
            'days_active': 120
        },
        "Medium Risk Wallet": {
            'tx_count': 150,
            'total_value': 500000,
            'avg_tx_value': 3333,
            'unique_recipients': 45,
            'rapid_tx_ratio': 0.15,
            'large_tx_ratio': 0.08,
            'days_active': 45
        },
        "High Risk Wallet": {
            'tx_count': 500,
            'total_value': 5000000,
            'avg_tx_value': 10000,
            'unique_recipients': 200,
            'rapid_tx_ratio': 0.35,
            'large_tx_ratio': 0.25,
            'days_active': 7
        }
    }
    
    print("Calculating risk scores for different wallet profiles...")
    
    for scenario_name, metrics in scenarios.items():
        print(f"\n  📊 {scenario_name}:")
        
        # Transaction risk (based on volume and frequency)
        tx_risk = min(1.0, (metrics['total_value'] / 1000000) * 0.4 + 
                          (metrics['tx_count'] / 100) * 0.3 +
                          (metrics['avg_tx_value'] / 50000) * 0.3)
        
        # Behavioral risk (based on patterns)
        behavioral_risk = min(1.0, metrics['rapid_tx_ratio'] * 0.6 + 
                                  metrics['large_tx_ratio'] * 0.4)
        
        # Network risk (based on recipient diversity)
        network_risk = min(1.0, (metrics['unique_recipients'] / 100) * 0.7)
        
        # Temporal risk (based on activity concentration)
        temporal_risk = max(0.0, 1.0 - (metrics['days_active'] / 365))
        
        # Overall risk (weighted combination)
        overall_risk = (tx_risk * 0.3 + 
                       behavioral_risk * 0.25 + 
                       network_risk * 0.2 + 
                       temporal_risk * 0.25)
        
        # Risk level classification
        if overall_risk >= 0.7:
            risk_level = "HIGH"
            color = "🔴"
        elif overall_risk >= 0.4:
            risk_level = "MEDIUM" 
            color = "🟡"
        elif overall_risk >= 0.2:
            risk_level = "LOW"
            color = "🟢"
        else:
            risk_level = "MINIMAL"
            color = "⚪"
            
        print(f"    Transaction Risk:  {tx_risk:.3f}")
        print(f"    Behavioral Risk:   {behavioral_risk:.3f}")
        print(f"    Network Risk:      {network_risk:.3f}")
        print(f"    Temporal Risk:     {temporal_risk:.3f}")
        print(f"    Overall Risk:      {overall_risk:.3f}")
        print(f"    Risk Level:        {color} {risk_level}")

def main():
    """Run the minimal demo"""
    print("🎯 ETHEREYE Minimal ML Demo")
    print("=" * 45)
    print("Direct scikit-learn implementation")
    print("=" * 45)
    
    try:
        # Demo clustering
        clustering_results = demo_direct_clustering()
        
        # Demo anomaly detection  
        anomaly_results = demo_direct_anomaly_detection()
        
        # Demo risk scoring
        demo_direct_risk_scoring()
        
        print("\n" + "=" * 45)
        print("🎉 Minimal demo completed successfully!")
        print("\n📋 What we demonstrated:")
        print("✅ DBSCAN clustering with scikit-learn")
        print("✅ Isolation Forest anomaly detection")
        print("✅ Multi-factor risk scoring algorithm")
        print("✅ Realistic blockchain data simulation")
        
        print(f"\n📊 Results Summary:")
        print(f"• Clustering: {clustering_results['n_clusters']} clusters found")
        print(f"• Anomaly Detection: {anomaly_results['n_anomalies']} anomalies detected")
        print(f"• Risk Assessment: 3 wallet risk profiles analyzed")
        
        print(f"\n🔧 Technical Details:")
        print(f"• Silhouette Score: {clustering_results['silhouette_score']:.3f}")
        print(f"• Anomaly Rate: {anomaly_results['anomaly_rate']:.1f}%")
        print(f"• Core ML Libraries: sklearn, pandas, numpy")
        
        print(f"\n🚀 This proves the core ML pipeline works!")
        print(f"Next: Install spacy/nltk for full NLP capabilities")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())