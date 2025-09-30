"""
FastAPI ML Endpoints for ETHEREYE
================================

REST API endpoints for machine learning services:
- Clustering analysis
- Anomaly detection  
- Risk scoring
- NLP/PII extraction (when dependencies available)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Initialize router
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class WalletData(BaseModel):
    """Wallet data for clustering analysis"""
    address: str
    total_value: float
    tx_count: int
    unique_recipients: int
    avg_tx_value: float
    days_active: int

class TransactionData(BaseModel):
    """Transaction data for anomaly detection"""
    tx_hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: Optional[float] = None

class RiskAssessmentRequest(BaseModel):
    """Request for wallet risk assessment"""
    wallet_address: str
    transactions: List[TransactionData]
    include_nlp: bool = False

class ClusteringRequest(BaseModel):
    """Request for wallet clustering"""
    wallets: List[WalletData]
    eps: float = Field(default=0.5, ge=0.1, le=2.0)
    min_samples: int = Field(default=3, ge=2, le=20)

class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    transactions: List[TransactionData]
    contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    n_estimators: int = Field(default=100, ge=10, le=1000)

class TextAnalysisRequest(BaseModel):
    """Request for NLP/PII text analysis"""
    texts: List[str]
    include_pii: bool = True
    include_sentiment: bool = True
    include_risk_keywords: bool = True

# Response models
class ClusteringResult(BaseModel):
    """Clustering analysis results"""
    n_clusters: int
    n_noise: int
    silhouette_score: float
    cluster_assignments: Dict[str, int]  # wallet_address -> cluster_id
    cluster_stats: Dict[int, Dict[str, Any]]

class AnomalyResult(BaseModel):
    """Anomaly detection results"""
    n_anomalies: int
    anomaly_rate: float
    anomalous_transactions: List[Dict[str, Any]]
    mean_anomaly_score: float

class RiskScore(BaseModel):
    """Individual risk component scores"""
    transaction_risk: float
    behavioral_risk: float
    temporal_risk: float
    network_risk: float
    compliance_risk: float
    overall_risk: float
    risk_level: str
    confidence: float

class RiskAssessmentResult(BaseModel):
    """Complete risk assessment results"""
    wallet_address: str
    risk_score: RiskScore
    contributing_factors: List[str]
    recommendations: List[str]
    timestamp: datetime

class TextAnalysisResult(BaseModel):
    """NLP/PII analysis results"""
    pii_detected: List[Dict[str, Any]]
    sentiment_scores: List[Dict[str, float]]
    risk_keywords: List[str]
    crypto_relevance_score: float

# ==============================================================================
# CLUSTERING ENDPOINTS
# ==============================================================================

@ml_router.post("/cluster", response_model=ClusteringResult)
async def cluster_wallets(request: ClusteringRequest):
    """
    Perform DBSCAN clustering on wallet addresses
    """
    try:
        logger.info(f"Starting clustering analysis for {len(request.wallets)} wallets")
        
        # Convert to DataFrame
        wallet_df = pd.DataFrame([wallet.dict() for wallet in request.wallets])
        
        # Use direct sklearn implementation (like in minimal demo)
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # Prepare features
        feature_cols = ['total_value', 'tx_count', 'unique_recipients', 'avg_tx_value', 'days_active']
        X = wallet_df[feature_cols].fillna(wallet_df[feature_cols].median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        dbscan = DBSCAN(eps=request.eps, min_samples=request.min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(labels) - 1:
            silhouette = float(silhouette_score(X_scaled, labels))
        else:
            silhouette = -1.0
        
        # Create cluster assignments
        cluster_assignments = {
            wallet.address: int(label) 
            for wallet, label in zip(request.wallets, labels)
        }
        
        # Calculate cluster stats
        cluster_stats = {}
        unique_labels = set(labels)
        for label in unique_labels:
            count = list(labels).count(label)
            percentage = (count / len(labels)) * 100
            cluster_stats[int(label)] = {
                'size': count,
                'percentage': percentage
            }
        
        result = ClusteringResult(
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette_score=silhouette,
            cluster_assignments=cluster_assignments,
            cluster_stats=cluster_stats
        )
        
        logger.info(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points")
        return result
        
    except Exception as e:
        logger.error(f"Clustering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

# ==============================================================================
# ANOMALY DETECTION ENDPOINTS
# ==============================================================================

@ml_router.post("/anomaly-detection", response_model=AnomalyResult)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalous transactions using Isolation Forest
    """
    try:
        logger.info(f"Starting anomaly detection for {len(request.transactions)} transactions")
        
        # Convert to DataFrame
        tx_data = []
        for tx in request.transactions:
            tx_dict = tx.dict()
            # Convert timestamp to numeric (seconds since epoch)
            tx_dict['timestamp_numeric'] = tx.timestamp.timestamp()
            tx_data.append(tx_dict)
        
        tx_df = pd.DataFrame(tx_data)
        
        # Use direct sklearn implementation
        from sklearn.ensemble import IsolationForest
        
        # Prepare features
        feature_cols = ['value', 'gas_used', 'timestamp_numeric']
        if 'gas_price' in tx_df.columns:
            feature_cols.append('gas_price')
            
        X = tx_df[feature_cols].fillna(tx_df[feature_cols].median())
        
        # Calculate time intervals between transactions (if multiple)
        if len(X) > 1:
            X_sorted = X.sort_values('timestamp_numeric')
            time_intervals = X_sorted['timestamp_numeric'].diff().fillna(0)
            X.loc[X_sorted.index, 'time_interval'] = time_intervals
            feature_cols.append('time_interval')
        
        # Perform anomaly detection
        iso_forest = IsolationForest(
            contamination=request.contamination,
            n_estimators=request.n_estimators,
            random_state=42
        )
        
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Get anomalous transactions
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        anomalous_transactions = []
        
        for idx in anomaly_indices:
            tx = request.transactions[idx]
            anomalous_transactions.append({
                'tx_hash': tx.tx_hash,
                'anomaly_score': float(anomaly_scores[idx]),
                'value': tx.value,
                'gas_used': tx.gas_used,
                'timestamp': tx.timestamp.isoformat(),
                'from_address': tx.from_address,
                'to_address': tx.to_address
            })
        
        # Sort by anomaly score (most anomalous first)
        anomalous_transactions.sort(key=lambda x: x['anomaly_score'])
        
        n_anomalies = len(anomalous_transactions)
        anomaly_rate = (n_anomalies / len(request.transactions)) * 100
        mean_score = float(np.mean(anomaly_scores))
        
        result = AnomalyResult(
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            anomalous_transactions=anomalous_transactions,
            mean_anomaly_score=mean_score
        )
        
        logger.info(f"Anomaly detection completed: {n_anomalies} anomalies detected")
        return result
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

# ==============================================================================
# RISK SCORING ENDPOINTS
# ==============================================================================

@ml_router.post("/risk-assessment", response_model=RiskAssessmentResult)
async def assess_wallet_risk(request: RiskAssessmentRequest):
    """
    Perform comprehensive risk assessment for a wallet
    """
    try:
        logger.info(f"Starting risk assessment for wallet {request.wallet_address}")
        
        # Convert transactions to DataFrame
        tx_data = [tx.dict() for tx in request.transactions]
        tx_df = pd.DataFrame(tx_data)
        
        if tx_df.empty:
            raise HTTPException(status_code=400, detail="No transaction data provided")
        
        # Calculate risk components using simplified risk model
        risk_components = _calculate_risk_components(tx_df)
        
        # Calculate overall risk score
        weights = {
            'transaction': 0.25,
            'behavioral': 0.25,
            'temporal': 0.2,
            'network': 0.15,
            'compliance': 0.15
        }
        
        overall_risk = (
            risk_components['transaction_risk'] * weights['transaction'] +
            risk_components['behavioral_risk'] * weights['behavioral'] +
            risk_components['temporal_risk'] * weights['temporal'] +
            risk_components['network_risk'] * weights['network'] +
            risk_components['compliance_risk'] * weights['compliance']
        )
        
        # Determine risk level
        if overall_risk >= 0.7:
            risk_level = "HIGH"
        elif overall_risk >= 0.4:
            risk_level = "MEDIUM"
        elif overall_risk >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Generate contributing factors and recommendations
        contributing_factors = _generate_contributing_factors(risk_components)
        recommendations = _generate_recommendations(risk_level, risk_components)
        
        # Calculate confidence based on data availability
        confidence = min(1.0, len(request.transactions) / 50.0)  # More transactions = higher confidence
        
        risk_score = RiskScore(
            transaction_risk=risk_components['transaction_risk'],
            behavioral_risk=risk_components['behavioral_risk'],
            temporal_risk=risk_components['temporal_risk'],
            network_risk=risk_components['network_risk'],
            compliance_risk=risk_components['compliance_risk'],
            overall_risk=overall_risk,
            risk_level=risk_level,
            confidence=confidence
        )
        
        result = RiskAssessmentResult(
            wallet_address=request.wallet_address,
            risk_score=risk_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        logger.info(f"Risk assessment completed: {risk_level} risk ({overall_risk:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# ==============================================================================
# NLP/TEXT ANALYSIS ENDPOINTS
# ==============================================================================

@ml_router.post("/text-analysis", response_model=TextAnalysisResult)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text for PII, sentiment, and risk indicators
    Note: Requires NLP dependencies (spacy, nltk, transformers)
    """
    try:
        logger.info(f"Starting text analysis for {len(request.texts)} texts")
        
        # Check if NLP dependencies are available
        try:
            import spacy
            nlp_available = True
        except ImportError:
            nlp_available = False
        
        if not nlp_available:
            # Fallback to basic analysis without NLP libraries
            result = _basic_text_analysis(request.texts)
        else:
            # Use full NLP pipeline
            try:
                from ml_models.nlp_pii_extraction.nlp_model import NLPPIIExtractor
                extractor = NLPPIIExtractor()
                result = extractor.analyze_batch(request.texts)
                
                # Convert to response format
                result = TextAnalysisResult(
                    pii_detected=result.get('pii_entities', []),
                    sentiment_scores=result.get('sentiment_analysis', []),
                    risk_keywords=result.get('risk_keywords', []),
                    crypto_relevance_score=result.get('crypto_relevance', {}).get('avg_score', 0.0)
                )
            except Exception as e:
                logger.warning(f"Full NLP analysis failed, using basic analysis: {str(e)}")
                result = _basic_text_analysis(request.texts)
        
        logger.info("Text analysis completed")
        return result
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

# ==============================================================================
# HEALTH CHECK ENDPOINTS
# ==============================================================================

@ml_router.get("/health")
async def health_check():
    """
    Check health status of ML services
    """
    try:
        # Test basic ML libraries
        import sklearn
        import pandas
        import numpy
        
        # Check NLP availability
        nlp_available = False
        try:
            import spacy
            nlp_available = True
        except ImportError:
            pass
        
        return {
            "status": "healthy",
            "ml_libraries": {
                "sklearn": sklearn.__version__,
                "pandas": pandas.__version__, 
                "numpy": numpy.__version__
            },
            "nlp_available": nlp_available,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _calculate_risk_components(tx_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate individual risk components"""
    
    # Transaction risk
    total_value = tx_df['value'].sum()
    avg_value = tx_df['value'].mean()
    tx_count = len(tx_df)
    
    tx_risk = min(1.0, (total_value / 1000000) * 0.3 + 
                       (tx_count / 100) * 0.3 +
                       (avg_value / 50000) * 0.4)
    
    # Behavioral risk
    high_value_txs = (tx_df['value'] > tx_df['value'].quantile(0.9)).sum()
    high_gas_txs = (tx_df['gas_used'] > tx_df['gas_used'].quantile(0.9)).sum()
    
    behavioral_risk = min(1.0, (high_value_txs / max(1, tx_count)) * 0.6 +
                              (high_gas_txs / max(1, tx_count)) * 0.4)
    
    # Temporal risk
    if 'timestamp' in tx_df.columns and len(tx_df) > 1:
        tx_df_sorted = tx_df.sort_values('timestamp')
        time_diffs = tx_df_sorted['timestamp'].diff().dt.total_seconds().fillna(0)
        rapid_txs = (time_diffs < 60).sum()  # Transactions within 1 minute
        temporal_risk = min(1.0, rapid_txs / max(1, tx_count))
    else:
        temporal_risk = 0.0
    
    # Network risk
    unique_recipients = tx_df['to_address'].nunique()
    network_risk = min(1.0, unique_recipients / 50.0)  # Normalized by expected max
    
    # Compliance risk (simplified)
    compliance_risk = 0.0  # Placeholder - would check against blacklists etc.
    
    return {
        'transaction_risk': tx_risk,
        'behavioral_risk': behavioral_risk,
        'temporal_risk': temporal_risk,
        'network_risk': network_risk,
        'compliance_risk': compliance_risk
    }

def _generate_contributing_factors(risk_components: Dict[str, float]) -> List[str]:
    """Generate human-readable contributing factors"""
    factors = []
    
    if risk_components['transaction_risk'] > 0.5:
        factors.append("High transaction volumes detected")
    if risk_components['behavioral_risk'] > 0.5:
        factors.append("Unusual transaction patterns observed")
    if risk_components['temporal_risk'] > 0.5:
        factors.append("Rapid-fire transaction activity")
    if risk_components['network_risk'] > 0.5:
        factors.append("High recipient address diversity")
    
    if not factors:
        factors.append("No significant risk factors identified")
    
    return factors

def _generate_recommendations(risk_level: str, risk_components: Dict[str, float]) -> List[str]:
    """Generate recommendations based on risk level"""
    recommendations = []
    
    if risk_level == "HIGH":
        recommendations.extend([
            "Enhanced due diligence recommended",
            "Consider manual review of transaction patterns",
            "Monitor for continued suspicious activity"
        ])
    elif risk_level == "MEDIUM":
        recommendations.extend([
            "Additional monitoring recommended",
            "Review recent transaction history"
        ])
    else:
        recommendations.append("Standard monitoring protocols sufficient")
    
    return recommendations

def _basic_text_analysis(texts: List[str]) -> TextAnalysisResult:
    """Basic text analysis without NLP libraries"""
    import re
    
    # Basic PII detection using regex
    pii_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'wallet_address': r'\b0x[a-fA-F0-9]{40}\b',
        'private_key': r'\b[a-fA-F0-9]{64}\b'
    }
    
    pii_detected = []
    for i, text in enumerate(texts):
        for pii_type, pattern in pii_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                pii_detected.append({
                    'text_index': i,
                    'entity_type': pii_type,
                    'entity_text': match,
                    'confidence': 0.8  # Fixed confidence for regex
                })
    
    # Basic sentiment (placeholder)
    sentiment_scores = []
    for text in texts:
        # Simple sentiment based on positive/negative words
        positive_words = ['good', 'great', 'excellent', 'profit', 'gain', 'bullish']
        negative_words = ['bad', 'terrible', 'loss', 'scam', 'fraud', 'bearish']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count > neg_count:
            sentiment = 0.7
        elif neg_count > pos_count:
            sentiment = 0.3
        else:
            sentiment = 0.5
            
        sentiment_scores.append({
            'text_index': len(sentiment_scores),
            'sentiment': sentiment,
            'confidence': 0.6
        })
    
    # Basic risk keywords
    risk_keywords = ['scam', 'fraud', 'hack', 'phishing', 'ponzi', 'rugpull']
    found_keywords = []
    for text in texts:
        for keyword in risk_keywords:
            if keyword in text.lower():
                found_keywords.append(keyword)
    
    return TextAnalysisResult(
        pii_detected=pii_detected,
        sentiment_scores=sentiment_scores,
        risk_keywords=list(set(found_keywords)),
        crypto_relevance_score=0.5  # Placeholder
    )