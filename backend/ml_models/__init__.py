"""
ETHEREYE ML Models Package
==========================

This package contains machine learning models and algorithms for:
- Clustering (DBSCAN)
- Anomaly Detection (IsolationForest)  
- NLP/PII Extraction (spaCy, NLTK, HuggingFace Transformers)
- Risk Scoring (Custom ML Pipeline)
"""

from .clustering import DBSCANClustering
from .anomaly_detection import IsolationForestDetector
from .nlp_pii_extraction import NLPPIIExtractor
from .risk_scoring import RiskScorer
from .preprocessing import DataPreprocessor
from .utils import ModelUtils

__all__ = [
    'DBSCANClustering',
    'IsolationForestDetector', 
    'NLPPIIExtractor',
    'RiskScorer',
    'DataPreprocessor',
    'ModelUtils'
]

__version__ = '1.0.0'