#!/usr/bin/env python3
"""
Test script to verify ML services integration for ETHEREYE
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')
sys.path.append('./ml_services')

def test_ml_services():
    """Test if ML services can be imported"""
    print("Testing ML Services Integration...")
    print("=" * 50)
    
    # Test clustering service
    try:
        from ml_services.clustering_service import BlockchainClusteringService
        print("✓ Clustering service imported successfully")
        clustering_service = BlockchainClusteringService()
        print("✓ Clustering service initialized")
    except Exception as e:
        print(f"✗ Clustering service failed: {e}")
    
    # Test NLP service
    try:
        from ml_services.nlp_service import BlockchainNLPService
        print("✓ NLP service imported successfully")
        nlp_service = BlockchainNLPService()
        print("✓ NLP service initialized")
    except Exception as e:
        print(f"✗ NLP service failed: {e}")
    
    # Test risk scoring service
    try:
        from ml_services.risk_scoring_service import BlockchainRiskScoringService
        print("✓ Risk scoring service imported successfully")
        risk_service = BlockchainRiskScoringService()
        print("✓ Risk scoring service initialized")
    except Exception as e:
        print(f"✗ Risk scoring service failed: {e}")
    
    # Test advanced ML endpoints
    try:
        from api.advanced_ml_endpoints import router
        print("✓ Advanced ML endpoints imported successfully")
    except Exception as e:
        print(f"✗ Advanced ML endpoints failed: {e}")
    
    print("\nTesting complete!")

def test_simple_main():
    """Test if simple_main.py can run"""
    print("\nTesting simple_main.py...")
    print("=" * 30)
    
    try:
        import simple_main
        print("✓ simple_main.py imported successfully")
        print("✓ FastAPI app created")
    except Exception as e:
        print(f"✗ simple_main.py failed: {e}")

if __name__ == "__main__":
    test_ml_services()
    test_simple_main()