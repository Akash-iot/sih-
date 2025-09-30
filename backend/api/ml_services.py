"""
FastAPI Endpoints for ETHEREYE ML Services
==========================================

This module provides REST API endpoints for all machine learning services:
- DBSCAN Clustering
- Isolation Forest Anomaly Detection  
- NLP/PII Extraction
- Risk Scoring
- Data Preprocessing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
import io
import json
import logging
from sqlalchemy.orm import Session

# Import ML models
from ..ml_models.clustering.dbscan_model import DBSCANClustering
from ..ml_models.anomaly_detection.isolation_forest_model import IsolationForestDetector
from ..ml_models.nlp_pii_extraction.nlp_processor import NLPPIIExtractor
from ..ml_models.risk_scoring.risk_model import RiskScorer
from ..ml_models.preprocessing.data_processor import DataPreprocessor
from ..ml_models.utils.model_utils import ModelUtils

# Import database dependencies
from ..models.database import get_db

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["ml-services"])

# Pydantic models for request/response
class ClusteringRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of data points to cluster")
    eps: float = Field(0.5, description="DBSCAN eps parameter")
    min_samples: int = Field(5, description="DBSCAN min_samples parameter")
    feature_columns: Optional[List[str]] = Field(None, description="Specific columns to use for clustering")

class ClusteringResponse(BaseModel):
    success: bool
    n_clusters: int
    n_noise: int
    silhouette_score: float
    labels: List[int]
    cluster_stats: Dict[int, Dict[str, Any]]
    execution_time: float

class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of data points for anomaly detection")
    contamination: float = Field(0.1, description="Expected proportion of outliers")
    n_estimators: int = Field(100, description="Number of base estimators")
    feature_columns: Optional[List[str]] = Field(None, description="Specific columns to use")

class AnomalyDetectionResponse(BaseModel):
    success: bool
    n_anomalies: int
    anomaly_rate: float
    predictions: List[int]  # 1 for normal, -1 for anomaly
    anomaly_scores: List[float]
    threshold: Optional[float]
    execution_time: float

class NLPAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    extract_pii: bool = Field(True, description="Whether to extract PII")
    analyze_sentiment: bool = Field(True, description="Whether to analyze sentiment")
    detect_toxicity: bool = Field(True, description="Whether to detect toxicity")

class NLPAnalysisResponse(BaseModel):
    success: bool
    total_texts: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time: float

class RiskAssessmentRequest(BaseModel):
    wallet_address: str = Field(..., description="Wallet address to assess")
    transaction_data: List[Dict[str, Any]] = Field(..., description="Transaction data for the wallet")
    text_data: Optional[List[str]] = Field(None, description="Associated text data")
    config: Optional[Dict[str, Any]] = Field(None, description="Risk assessment configuration")

class RiskAssessmentResponse(BaseModel):
    success: bool
    wallet_address: str
    overall_risk_score: float
    risk_level: str
    confidence: float
    risk_factors: Dict[str, float]
    contributing_factors: List[str]
    recommendations: List[str]
    execution_time: float

class BatchRiskAssessmentRequest(BaseModel):
    wallet_data: Dict[str, List[Dict[str, Any]]] = Field(..., description="Dictionary mapping wallet addresses to transaction data")
    text_data: Optional[Dict[str, List[str]]] = Field(None, description="Optional text data for each wallet")
    config: Optional[Dict[str, Any]] = Field(None, description="Risk assessment configuration")

# Global model instances (initialized lazily)
clustering_model = None
anomaly_detector = None
nlp_processor = None
risk_scorer = None
data_preprocessor = None

def get_clustering_model():
    """Get or initialize clustering model"""
    global clustering_model
    if clustering_model is None:
        clustering_model = DBSCANClustering()
        logger.info("DBSCAN clustering model initialized")
    return clustering_model

def get_anomaly_detector():
    """Get or initialize anomaly detector"""
    global anomaly_detector
    if anomaly_detector is None:
        anomaly_detector = IsolationForestDetector()
        logger.info("Isolation Forest detector initialized")
    return anomaly_detector

def get_nlp_processor():
    """Get or initialize NLP processor"""
    global nlp_processor
    if nlp_processor is None:
        nlp_processor = NLPPIIExtractor()
        logger.info("NLP processor initialized")
    return nlp_processor

def get_risk_scorer():
    """Get or initialize risk scorer"""
    global risk_scorer
    if risk_scorer is None:
        risk_scorer = RiskScorer()
        risk_scorer.initialize_models()
        logger.info("Risk scorer initialized")
    return risk_scorer

def get_data_preprocessor():
    """Get or initialize data preprocessor"""
    global data_preprocessor
    if data_preprocessor is None:
        data_preprocessor = DataPreprocessor()
        logger.info("Data preprocessor initialized")
    return data_preprocessor

@router.get("/health")
async def health_check():
    """Health check endpoint for ML services"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "clustering": clustering_model is not None,
            "anomaly_detection": anomaly_detector is not None,
            "nlp_processing": nlp_processor is not None,
            "risk_scoring": risk_scorer is not None,
            "data_preprocessing": data_preprocessor is not None
        }
    }

@router.post("/clustering/analyze", response_model=ClusteringResponse)
async def analyze_clustering(request: ClusteringRequest):
    """
    Perform DBSCAN clustering analysis
    """
    start_time = datetime.now()
    
    try:
        # Get clustering model
        model = get_clustering_model()
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty")
        
        # Update model parameters
        model.eps = request.eps
        model.min_samples = request.min_samples
        
        # Perform clustering
        results = model.fit(df, request.feature_columns)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ClusteringResponse(
            success=True,
            n_clusters=results.n_clusters,
            n_noise=results.n_noise,
            silhouette_score=results.silhouette_score,
            labels=results.labels.tolist(),
            cluster_stats=results.cluster_stats,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@router.post("/anomaly-detection/analyze", response_model=AnomalyDetectionResponse)
async def analyze_anomalies(request: AnomalyDetectionRequest):
    """
    Perform anomaly detection using Isolation Forest
    """
    start_time = datetime.now()
    
    try:
        # Get anomaly detector
        detector = get_anomaly_detector()
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty")
        
        # Update detector parameters
        detector.contamination = request.contamination
        detector.n_estimators = request.n_estimators
        
        # Perform anomaly detection
        results = detector.fit_predict(df, request.feature_columns)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AnomalyDetectionResponse(
            success=True,
            n_anomalies=results.n_anomalies,
            anomaly_rate=results.anomaly_rate,
            predictions=results.predictions.tolist(),
            anomaly_scores=results.anomaly_scores.tolist(),
            threshold=results.threshold,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.post("/nlp/analyze", response_model=NLPAnalysisResponse)
async def analyze_text(request: NLPAnalysisRequest):
    """
    Perform NLP analysis including PII extraction, sentiment analysis, and toxicity detection
    """
    start_time = datetime.now()
    
    try:
        # Get NLP processor
        processor = get_nlp_processor()
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided for analysis")
        
        # Perform batch text analysis
        results = processor.batch_analyze_texts(request.texts)
        
        # Process results
        processed_results = []
        total_pii_found = 0
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        avg_toxicity = 0.0
        
        for i, result in enumerate(results):
            if result is None:
                continue
                
            processed_result = {
                "text_index": i,
                "original_text_length": len(result.original_text),
                "sentiment": result.sentiment if request.analyze_sentiment else None,
                "toxicity_score": result.toxicity_score if request.detect_toxicity else None,
                "pii_data": {
                    "entities_found": len(result.pii_data.pii_entities),
                    "risk_score": result.pii_data.risk_score,
                    "types_found": result.pii_data.pii_types_found
                } if request.extract_pii else None,
                "entities": result.entities,
                "keywords": result.keywords,
                "crypto_related": result.classification_results.get("is_crypto_related", False)
            }
            
            processed_results.append(processed_result)
            
            # Update summary statistics
            if request.extract_pii and result.pii_data.pii_entities:
                total_pii_found += len(result.pii_data.pii_entities)
            
            if request.analyze_sentiment:
                sentiment_label = result.sentiment.get("combined_label", "neutral")
                sentiment_distribution[sentiment_label] += 1
            
            if request.detect_toxicity:
                avg_toxicity += result.toxicity_score
        
        # Calculate averages
        total_processed = len([r for r in results if r is not None])
        if total_processed > 0:
            avg_toxicity /= total_processed
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "total_texts_processed": total_processed,
            "total_pii_entities_found": total_pii_found,
            "sentiment_distribution": sentiment_distribution if request.analyze_sentiment else None,
            "average_toxicity_score": avg_toxicity if request.detect_toxicity else None
        }
        
        return NLPAnalysisResponse(
            success=True,
            total_texts=len(request.texts),
            results=processed_results,
            summary=summary,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in NLP analysis: {e}")
        raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")

@router.post("/risk-assessment/analyze", response_model=RiskAssessmentResponse)
async def assess_wallet_risk(request: RiskAssessmentRequest):
    """
    Perform comprehensive risk assessment for a wallet
    """
    start_time = datetime.now()
    
    try:
        # Get risk scorer
        scorer = get_risk_scorer()
        
        # Convert transaction data to DataFrame
        transaction_df = pd.DataFrame(request.transaction_data)
        
        if transaction_df.empty:
            raise HTTPException(status_code=400, detail="No transaction data provided")
        
        # Perform risk assessment
        assessment = scorer.assess_wallet_risk(
            wallet_address=request.wallet_address,
            wallet_data=transaction_df,
            text_data=request.text_data,
            external_data=request.config
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return RiskAssessmentResponse(
            success=True,
            wallet_address=assessment.wallet_address,
            overall_risk_score=assessment.overall_risk_score,
            risk_level=assessment.risk_level,
            confidence=assessment.confidence,
            risk_factors={
                "clustering_risk": assessment.risk_factors.clustering_risk,
                "anomaly_risk": assessment.risk_factors.anomaly_risk,
                "nlp_risk": assessment.risk_factors.nlp_risk,
                "transaction_risk": assessment.risk_factors.transaction_risk,
                "behavioral_risk": assessment.risk_factors.behavioral_risk,
                "temporal_risk": assessment.risk_factors.temporal_risk,
                "network_risk": assessment.risk_factors.network_risk,
                "compliance_risk": assessment.risk_factors.compliance_risk
            },
            contributing_factors=assessment.contributing_factors,
            recommendations=assessment.recommendations,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.post("/risk-assessment/batch")
async def batch_assess_wallet_risk(request: BatchRiskAssessmentRequest, 
                                   background_tasks: BackgroundTasks):
    """
    Perform batch risk assessment for multiple wallets
    """
    try:
        # Get risk scorer
        scorer = get_risk_scorer()
        
        # Convert wallet data to DataFrames
        wallet_dataframes = {}
        for address, transactions in request.wallet_data.items():
            df = pd.DataFrame(transactions)
            if not df.empty:
                wallet_dataframes[address] = df
        
        if not wallet_dataframes:
            raise HTTPException(status_code=400, detail="No valid wallet data provided")
        
        # Start batch assessment in background
        def run_batch_assessment():
            try:
                assessments = scorer.batch_assess_wallets(
                    wallet_data=wallet_dataframes,
                    text_data=request.text_data
                )
                # Here you could save results to database or file
                logger.info(f"Batch assessment completed for {len(assessments)} wallets")
            except Exception as e:
                logger.error(f"Error in batch assessment: {e}")
        
        background_tasks.add_task(run_batch_assessment)
        
        return {
            "success": True,
            "message": "Batch risk assessment started",
            "wallets_queued": len(wallet_dataframes),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting batch risk assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Batch assessment failed: {str(e)}")

@router.post("/preprocessing/validate")
async def validate_data(file: UploadFile = File(...)):
    """
    Validate uploaded data schema and quality
    """
    try:
        # Get data preprocessor
        preprocessor = get_data_preprocessor()
        
        # Read uploaded file
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Define common blockchain data columns
        required_columns = ["timestamp", "from_address", "to_address", "value"]
        optional_columns = ["gas_used", "gas_price", "transaction_hash", "block_number"]
        
        # Validate data schema
        validation_result = ModelUtils.validate_data_schema(
            data=df,
            required_columns=required_columns,
            optional_columns=optional_columns
        )
        
        # Generate data profile
        data_profile = ModelUtils.generate_data_profile(df)
        
        return {
            "success": True,
            "filename": file.filename,
            "validation_result": validation_result,
            "data_profile": data_profile
        }
        
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise HTTPException(status_code=500, detail=f"Data validation failed: {str(e)}")

@router.post("/preprocessing/clean")
async def preprocess_data(file: UploadFile = File(...)):
    """
    Clean and preprocess uploaded blockchain data
    """
    try:
        # Get data preprocessor
        preprocessor = get_data_preprocessor()
        
        # Read uploaded file
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Preprocess the data
        if "transaction" in file.filename.lower() or any(col in df.columns for col in ["from_address", "to_address"]):
            processed_df = preprocessor.preprocess_transaction_data(df)
        else:
            processed_df = preprocessor.preprocess_address_data(df)
        
        # Convert to JSON for response
        processed_data = processed_df.to_dict(orient='records')
        
        return {
            "success": True,
            "original_shape": df.shape,
            "processed_shape": processed_df.shape,
            "processed_data": processed_data[:100],  # Limit response size
            "columns_added": list(set(processed_df.columns) - set(df.columns))
        }
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {str(e)}")

@router.get("/models/info")
async def get_models_info():
    """
    Get information about all available ML models
    """
    try:
        info = {
            "clustering": get_clustering_model().get_model_info() if clustering_model else None,
            "anomaly_detection": get_anomaly_detector().get_model_info() if anomaly_detector else None,
            "nlp_processing": get_nlp_processor().get_model_info() if nlp_processor else None,
            "risk_scoring": get_risk_scorer().get_model_info() if risk_scorer else None
        }
        
        return {
            "success": True,
            "models_info": info,
            "available_models": [k for k, v in info.items() if v is not None]
        }
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")

@router.post("/models/initialize")
async def initialize_models(models: List[str] = None):
    """
    Initialize specified ML models
    """
    try:
        if models is None:
            models = ["clustering", "anomaly_detection", "nlp_processing", "risk_scoring"]
        
        initialized = []
        
        for model_name in models:
            if model_name == "clustering":
                get_clustering_model()
                initialized.append("clustering")
            elif model_name == "anomaly_detection":
                get_anomaly_detector()
                initialized.append("anomaly_detection")
            elif model_name == "nlp_processing":
                get_nlp_processor()
                initialized.append("nlp_processing")
            elif model_name == "risk_scoring":
                get_risk_scorer()
                initialized.append("risk_scoring")
        
        return {
            "success": True,
            "message": f"Models initialized successfully",
            "initialized_models": initialized
        }
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

# Additional utility endpoints
@router.get("/statistics/risk-distribution")
async def get_risk_distribution(db: Session = Depends(get_db)):
    """
    Get risk distribution statistics from recent assessments
    """
    try:
        # This would typically query the database for recent risk assessments
        # For now, return a placeholder response
        return {
            "success": True,
            "message": "Risk distribution endpoint - implementation pending database integration",
            "data": {
                "high_risk": 15,
                "medium_risk": 35,
                "low_risk": 40,
                "minimal_risk": 10
            }
        }
    except Exception as e:
        logger.error(f"Error getting risk distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk distribution: {str(e)}")