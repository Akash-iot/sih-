"""
Advanced ML API Endpoints for ETHEREYE
Integrates DBSCAN clustering, NLP/PII extraction, and custom risk scoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

# Import ML services
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "ml_services"))

try:
    from clustering_service import BlockchainClusteringService
    from nlp_service import BlockchainNLPService
    from risk_scoring_service import BlockchainRiskScoringService
except ImportError as e:
    print(f"Warning: ML services not available: {e}")
    BlockchainClusteringService = None
    BlockchainNLPService = None
    BlockchainRiskScoringService = None

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize ML services (with error handling)
clustering_service = None
nlp_service = None
risk_service = None

try:
    if BlockchainClusteringService:
        clustering_service = BlockchainClusteringService()
    if BlockchainNLPService:
        nlp_service = BlockchainNLPService()
    if BlockchainRiskScoringService:
        risk_service = BlockchainRiskScoringService()
    logger.info("Advanced ML services initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize ML services: {e}")

# Pydantic models for request/response
class AddressData(BaseModel):
    address: str = Field(..., description="Blockchain address")
    balance_eth: Optional[float] = Field(default=0, description="ETH balance")
    balance_usd: Optional[float] = Field(default=0, description="USD balance")
    transaction_count: Optional[int] = Field(default=0, description="Number of transactions")
    transactions: Optional[List[Dict]] = Field(default=[], description="Transaction history")

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    analysis_type: Optional[str] = Field(default="full", description="Type of analysis to perform")

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")

class RiskTrainingRequest(BaseModel):
    training_data: List[Dict] = Field(..., description="Training data for risk model")
    labels: List[str] = Field(..., description="Risk labels for training data")
    test_size: Optional[float] = Field(default=0.2, description="Test size for training")
    cv_folds: Optional[int] = Field(default=5, description="Cross-validation folds")

# ============================================================================
# CLUSTERING ENDPOINTS (DBSCAN + IsolationForest)
# ============================================================================

@router.post("/clustering/dbscan", tags=["Clustering"])
async def perform_dbscan_clustering(
    addresses: List[AddressData],
    eps: float = Query(0.5, description="DBSCAN eps parameter"),
    min_samples: int = Query(5, description="DBSCAN min_samples parameter"),
    optimize_params: bool = Query(True, description="Auto-optimize parameters")
):
    """
    Perform DBSCAN clustering on blockchain addresses
    Identifies groups of addresses with similar transaction patterns
    """
    if not clustering_service:
        raise HTTPException(status_code=503, detail="Clustering service not available")
    
    try:
        # Convert Pydantic models to dictionaries
        addresses_data = [addr.dict() for addr in addresses]
        
        # Perform clustering
        result = clustering_service.perform_dbscan_clustering(
            addresses_data, eps, min_samples, optimize_params
        )
        
        return {
            "status": "success",
            "clustering_type": "DBSCAN",
            "result": result,
            "parameters_used": {
                "eps": eps,
                "min_samples": min_samples,
                "optimize_params": optimize_params
            }
        }
        
    except Exception as e:
        logger.error(f"DBSCAN clustering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@router.post("/clustering/anomaly-detection", tags=["Clustering"])
async def detect_anomalies(
    addresses: List[AddressData],
    contamination: float = Query(0.1, description="Expected proportion of anomalies"),
    n_estimators: int = Query(100, description="Number of trees in IsolationForest")
):
    """
    Detect anomalous blockchain addresses using Isolation Forest
    Identifies addresses with unusual transaction patterns
    """
    if not clustering_service:
        raise HTTPException(status_code=503, detail="Clustering service not available")
    
    try:
        addresses_data = [addr.dict() for addr in addresses]
        
        result = clustering_service.detect_anomalies(
            addresses_data, contamination, n_estimators
        )
        
        return {
            "status": "success",
            "analysis_type": "anomaly_detection",
            "result": result,
            "parameters_used": {
                "contamination": contamination,
                "n_estimators": n_estimators
            }
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.post("/clustering/combined-analysis", tags=["Clustering"])
async def combined_clustering_analysis(addresses: List[AddressData]):
    """
    Combine DBSCAN clustering with anomaly detection for comprehensive analysis
    Provides both cluster membership and anomaly scores
    """
    if not clustering_service:
        raise HTTPException(status_code=503, detail="Clustering service not available")
    
    try:
        addresses_data = [addr.dict() for addr in addresses]
        
        result = clustering_service.combine_clustering_and_anomaly_detection(addresses_data)
        
        return {
            "status": "success",
            "analysis_type": "combined_clustering_anomaly",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Combined clustering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Combined analysis failed: {str(e)}")

# ============================================================================
# NLP/PII EXTRACTION ENDPOINTS (spaCy + NLTK + HuggingFace)
# ============================================================================

@router.post("/nlp/extract-pii", tags=["NLP"])
async def extract_pii(request: TextAnalysisRequest):
    """
    Extract PII (Personally Identifiable Information) from text
    Uses spaCy, NLTK, and HuggingFace for comprehensive detection
    """
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    
    try:
        result = nlp_service.extract_pii_from_text(request.text)
        
        return {
            "status": "success",
            "analysis_type": "pii_extraction",
            "text_analyzed": len(request.text),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"PII extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"PII extraction failed: {str(e)}")

@router.post("/nlp/analyze-blockchain-content", tags=["NLP"])
async def analyze_blockchain_content(request: TextAnalysisRequest):
    """
    Analyze text for blockchain-related content and sentiment
    Identifies crypto-related keywords, sentiment, and risk indicators
    """
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    
    try:
        result = nlp_service.analyze_blockchain_content(request.text)
        
        return {
            "status": "success",
            "analysis_type": "blockchain_content_analysis",
            "text_analyzed": len(request.text),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Blockchain content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")

@router.post("/nlp/analyze-transaction-memo", tags=["NLP"])
async def analyze_transaction_memo(
    memo: str = Query(..., description="Transaction memo/note to analyze")
):
    """
    Specialized analysis for transaction memos and notes
    Detects suspicious patterns, PII, and memo categorization
    """
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    
    try:
        result = nlp_service.analyze_transaction_memo(memo)
        
        return {
            "status": "success",
            "analysis_type": "transaction_memo_analysis",
            "memo_length": len(memo),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Transaction memo analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memo analysis failed: {str(e)}")

@router.post("/nlp/batch-analyze", tags=["NLP"])
async def batch_analyze_text(request: BatchTextRequest):
    """
    Analyze multiple texts in batch for efficiency
    Processes multiple texts and provides summary statistics
    """
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    
    try:
        result = nlp_service.batch_analyze_text(request.texts)
        
        return {
            "status": "success",
            "analysis_type": "batch_text_analysis",
            "texts_processed": len(request.texts),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Batch text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/nlp/model-info", tags=["NLP"])
async def get_nlp_model_info():
    """Get information about loaded NLP models and capabilities"""
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    
    try:
        model_info = nlp_service.get_model_info()
        
        return {
            "status": "success",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Getting NLP model info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

# ============================================================================
# RISK SCORING ENDPOINTS (Custom ML Pipeline)
# ============================================================================

@router.post("/risk-scoring/train", tags=["Risk Scoring"])
async def train_risk_model(
    request: RiskTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train the risk scoring model on labeled data
    Uses ensemble ML methods for suspicious wallet classification
    """
    if not risk_service:
        raise HTTPException(status_code=503, detail="Risk scoring service not available")
    
    try:
        # Start training in background for large datasets
        if len(request.training_data) > 100:
            background_tasks.add_task(
                _train_risk_model_background,
                request.training_data,
                request.labels,
                request.test_size,
                request.cv_folds
            )
            
            return {
                "status": "training_started",
                "message": "Training started in background",
                "training_samples": len(request.training_data),
                "estimated_time": "5-10 minutes"
            }
        else:
            # Train synchronously for small datasets
            result = risk_service.train_risk_model(
                request.training_data,
                request.labels,
                request.test_size,
                request.cv_folds
            )
            
            return {
                "status": "training_completed",
                "result": result
            }
        
    except Exception as e:
        logger.error(f"Risk model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def _train_risk_model_background(training_data, labels, test_size, cv_folds):
    """Background task for model training"""
    try:
        result = risk_service.train_risk_model(training_data, labels, test_size, cv_folds)
        logger.info(f"Background training completed: {result.get('best_model', 'unknown')}")
    except Exception as e:
        logger.error(f"Background training failed: {e}")

@router.post("/risk-scoring/predict", tags=["Risk Scoring"])
async def predict_risk_score(
    wallet_data: AddressData,
    model_name: str = Query("ensemble", description="Model to use for prediction"),
    return_probabilities: bool = Query(True, description="Return class probabilities")
):
    """
    Predict risk score for a wallet using trained ML models
    Returns risk category, confidence, and explanatory factors
    """
    if not risk_service:
        raise HTTPException(status_code=503, detail="Risk scoring service not available")
    
    try:
        result = risk_service.predict_risk_score(
            wallet_data.dict(),
            model_name,
            return_probabilities
        )
        
        return {
            "status": "success",
            "prediction_type": "individual_risk_scoring",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {str(e)}")

@router.post("/risk-scoring/batch-predict", tags=["Risk Scoring"])
async def batch_predict_risk(
    wallets: List[AddressData],
    model_name: str = Query("ensemble", description="Model to use for predictions")
):
    """
    Predict risk scores for multiple wallets in batch
    Efficient processing of large wallet datasets
    """
    if not risk_service:
        raise HTTPException(status_code=503, detail="Risk scoring service not available")
    
    try:
        wallets_data = [wallet.dict() for wallet in wallets]
        
        result = risk_service.batch_predict_risk(wallets_data, model_name)
        
        return {
            "status": "success",
            "prediction_type": "batch_risk_scoring",
            "wallets_processed": len(wallets),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Batch risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/risk-scoring/model-info", tags=["Risk Scoring"])
async def get_risk_model_info():
    """Get information about trained risk scoring models"""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Risk scoring service not available")
    
    try:
        model_info = risk_service.get_model_info()
        
        return {
            "status": "success",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Getting risk model info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

# ============================================================================
# COMPREHENSIVE ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/comprehensive-analysis", tags=["Comprehensive"])
async def comprehensive_wallet_analysis(
    addresses: List[AddressData],
    include_clustering: bool = Query(True, description="Include clustering analysis"),
    include_risk_scoring: bool = Query(True, description="Include risk scoring"),
    memo_texts: Optional[List[str]] = Query(None, description="Transaction memos to analyze")
):
    """
    Comprehensive analysis combining all ML capabilities
    Provides clustering, anomaly detection, risk scoring, and NLP analysis
    """
    try:
        results = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "addresses_analyzed": len(addresses),
            "analysis_components": []
        }
        
        # Clustering analysis
        if include_clustering and clustering_service:
            try:
                addresses_data = [addr.dict() for addr in addresses]
                clustering_result = clustering_service.combine_clustering_and_anomaly_detection(addresses_data)
                results["clustering_analysis"] = clustering_result
                results["analysis_components"].append("clustering")
            except Exception as e:
                logger.warning(f"Clustering analysis failed: {e}")
                results["clustering_error"] = str(e)
        
        # Risk scoring analysis
        if include_risk_scoring and risk_service:
            try:
                wallets_data = [addr.dict() for addr in addresses]
                risk_result = risk_service.batch_predict_risk(wallets_data)
                results["risk_analysis"] = risk_result
                results["analysis_components"].append("risk_scoring")
            except Exception as e:
                logger.warning(f"Risk scoring analysis failed: {e}")
                results["risk_scoring_error"] = str(e)
        
        # NLP analysis on memos
        if memo_texts and nlp_service:
            try:
                nlp_result = nlp_service.batch_analyze_text(memo_texts)
                results["nlp_analysis"] = nlp_result
                results["analysis_components"].append("nlp")
            except Exception as e:
                logger.warning(f"NLP analysis failed: {e}")
                results["nlp_error"] = str(e)
        
        # Generate combined insights
        results["combined_insights"] = _generate_combined_insights(results)
        
        return {
            "status": "success",
            "analysis_type": "comprehensive",
            "result": results
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

def _generate_combined_insights(analysis_results: Dict) -> Dict[str, Any]:
    """Generate combined insights from multiple analysis components"""
    insights = {
        "summary": {},
        "risk_indicators": [],
        "recommendations": []
    }
    
    # Analyze clustering results
    if "clustering_analysis" in analysis_results:
        clustering_data = analysis_results["clustering_analysis"]
        if "summary" in clustering_data:
            summary = clustering_data["summary"]
            insights["summary"]["high_risk_addresses"] = summary.get("high_risk_addresses", 0)
            insights["summary"]["anomalous_addresses"] = summary.get("anomalous_addresses", 0)
            
            if summary.get("high_risk_addresses", 0) > 0:
                insights["risk_indicators"].append(
                    f"Found {summary['high_risk_addresses']} high-risk addresses in cluster analysis"
                )
    
    # Analyze risk scoring results  
    if "risk_analysis" in analysis_results:
        risk_data = analysis_results["risk_analysis"]
        if "summary_statistics" in risk_data:
            stats = risk_data["summary_statistics"]
            risk_dist = stats.get("risk_distribution", {})
            
            high_risk_count = risk_dist.get("high_risk", 0) + risk_dist.get("critical_risk", 0)
            if high_risk_count > 0:
                insights["risk_indicators"].append(
                    f"Risk scoring identified {high_risk_count} high-risk wallets"
                )
    
    # Analyze NLP results
    if "nlp_analysis" in analysis_results:
        nlp_data = analysis_results["nlp_analysis"]
        if "summary_statistics" in nlp_data:
            stats = nlp_data["summary_statistics"]
            
            if stats.get("high_risk_count", 0) > 0:
                insights["risk_indicators"].append(
                    f"NLP analysis found {stats['high_risk_count']} high-risk text patterns"
                )
    
    # Generate recommendations
    total_risk_indicators = len(insights["risk_indicators"])
    if total_risk_indicators == 0:
        insights["recommendations"].append("No significant risk indicators found - addresses appear legitimate")
    elif total_risk_indicators <= 2:
        insights["recommendations"].append("Low to medium risk detected - monitor for unusual activity")
    else:
        insights["recommendations"].append("High risk indicators found - detailed investigation recommended")
        insights["recommendations"].append("Consider implementing enhanced monitoring for flagged addresses")
    
    return insights

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@router.get("/ml-services/status", tags=["Status"])
async def get_ml_services_status():
    """Get status of all ML services"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "clustering_service": {
                "available": clustering_service is not None,
                "status": "ready" if clustering_service else "unavailable",
                "capabilities": ["DBSCAN clustering", "Isolation Forest anomaly detection"] if clustering_service else []
            },
            "nlp_service": {
                "available": nlp_service is not None,
                "status": "ready" if nlp_service else "unavailable",
                "capabilities": ["PII extraction", "Sentiment analysis", "Blockchain content analysis"] if nlp_service else []
            },
            "risk_service": {
                "available": risk_service is not None,
                "status": "ready" if risk_service else "unavailable",
                "capabilities": ["Risk scoring", "Suspicious wallet classification", "Feature extraction"] if risk_service else []
            }
        }
    }
    
    # Add detailed model info if services are available
    if nlp_service:
        try:
            status["services"]["nlp_service"]["model_details"] = nlp_service.get_model_info()
        except:
            pass
    
    if risk_service:
        try:
            status["services"]["risk_service"]["model_details"] = risk_service.get_model_info()
        except:
            pass
    
    return status

@router.get("/ml-services/health", tags=["Status"])
async def ml_services_health_check():
    """Health check for ML services"""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services_available": 0,
        "services_total": 3
    }
    
    if clustering_service:
        health["services_available"] += 1
    if nlp_service:
        health["services_available"] += 1
    if risk_service:
        health["services_available"] += 1
    
    if health["services_available"] == 0:
        health["status"] = "unhealthy"
    elif health["services_available"] < health["services_total"]:
        health["status"] = "degraded"
    
    return health