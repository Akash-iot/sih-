"""
Risk Scoring ML Pipeline for ETHEREYE
=====================================

This module implements a comprehensive risk scoring system that combines:
- DBSCAN clustering results
- Isolation Forest anomaly detection
- NLP/PII extraction insights
- Custom blockchain-specific risk factors
- Ensemble scoring methodology

The risk scorer provides actionable intelligence for suspicious wallet classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import json

# Import other ML modules
from ..clustering.dbscan_model import DBSCANClustering, ClusteringResults
from ..anomaly_detection.isolation_forest_model import IsolationForestDetector, AnomalyResults
from ..nlp_pii_extraction.nlp_processor import NLPPIIExtractor, TextAnalysisResult
from ..preprocessing.data_processor import DataPreprocessor

logger = logging.getLogger(__name__)

@dataclass
class RiskFactors:
    """Container for individual risk factor scores"""
    clustering_risk: float = 0.0
    anomaly_risk: float = 0.0
    nlp_risk: float = 0.0
    transaction_risk: float = 0.0
    behavioral_risk: float = 0.0
    temporal_risk: float = 0.0
    network_risk: float = 0.0
    compliance_risk: float = 0.0

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    wallet_address: str
    overall_risk_score: float
    risk_level: str  # HIGH, MEDIUM, LOW
    confidence: float
    risk_factors: RiskFactors
    contributing_factors: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]
    assessment_timestamp: datetime

class RiskScorer:
    """
    Comprehensive risk scoring system for suspicious wallet classification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk scoring system
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize sub-models
        self.clustering_model = None
        self.anomaly_detector = None
        self.nlp_processor = None
        self.data_preprocessor = DataPreprocessor()
        
        # Risk scoring models
        self.ensemble_models = {}
        self.scalers = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            'high': self.config.get('high_risk_threshold', 0.7),
            'medium': self.config.get('medium_risk_threshold', 0.4),
            'low': self.config.get('low_risk_threshold', 0.2)
        }
        
        # Feature weights
        self.feature_weights = self.config.get('feature_weights', {
            'clustering': 0.15,
            'anomaly': 0.20,
            'nlp': 0.15,
            'transaction': 0.20,
            'behavioral': 0.15,
            'temporal': 0.10,
            'network': 0.05
        })
        
        self.is_trained = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'high_risk_threshold': 0.7,
            'medium_risk_threshold': 0.4,
            'low_risk_threshold': 0.2,
            'feature_weights': {
                'clustering': 0.15,
                'anomaly': 0.20,
                'nlp': 0.15,
                'transaction': 0.20,
                'behavioral': 0.15,
                'temporal': 0.10,
                'network': 0.05
            },
            'ensemble_method': 'weighted_average',
            'use_ml_models': True,
            'min_confidence_threshold': 0.6
        }
    
    def initialize_models(self):
        """Initialize all sub-models"""
        logger.info("Initializing risk scoring sub-models...")
        
        # Initialize clustering model
        self.clustering_model = DBSCANClustering(
            eps=0.5,
            min_samples=5,
            contamination=0.1
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = IsolationForestDetector(
            n_estimators=100,
            contamination=0.1
        )
        
        # Initialize NLP processor
        self.nlp_processor = NLPPIIExtractor()
        
        # Initialize ensemble models for final scoring
        if self.config.get('use_ml_models', True):
            self._initialize_ensemble_models()
        
        logger.info("All sub-models initialized successfully")
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble models for final risk scoring"""
        self.ensemble_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        self.scalers = {
            'features': StandardScaler(),
            'scores': MinMaxScaler()
        }
    
    def calculate_clustering_risk(self, wallet_data: pd.DataFrame, 
                                 clustering_results: Optional[ClusteringResults] = None) -> float:
        """
        Calculate risk score based on clustering analysis
        
        Args:
            wallet_data: Wallet transaction data
            clustering_results: Pre-computed clustering results
            
        Returns:
            Clustering risk score (0-1)
        """
        if clustering_results is None and self.clustering_model:
            # Run clustering analysis
            clustering_results = self.clustering_model.fit_predict(wallet_data)
        
        if clustering_results is None:
            return 0.5  # Default moderate risk if clustering unavailable
        
        clustering_risk = 0.0
        
        # Get cluster assignment for this wallet (assuming single wallet analysis)
        if len(clustering_results.labels) > 0:
            cluster_id = clustering_results.labels[0]
            
            # Risk factors based on cluster characteristics
            if cluster_id == -1:  # Noise/outlier
                clustering_risk += 0.4
            else:
                # Analyze cluster statistics
                cluster_stats = clustering_results.cluster_stats.get(cluster_id, {})
                cluster_size = cluster_stats.get('size', 1)
                
                # Smaller clusters might be more suspicious
                if cluster_size < 5:
                    clustering_risk += 0.3
                elif cluster_size < 20:
                    clustering_risk += 0.1
                
                # High transaction value clusters
                avg_value = cluster_stats.get('value_mean', 0)
                if avg_value > 100000:  # High value threshold
                    clustering_risk += 0.2
                
                # Frequent small transactions (potential structuring)
                tx_count = cluster_stats.get('tx_count_mean', 0)
                if tx_count > 100 and avg_value < 10000:
                    clustering_risk += 0.3
        
        return min(clustering_risk, 1.0)
    
    def calculate_anomaly_risk(self, wallet_data: pd.DataFrame,
                              anomaly_results: Optional[AnomalyResults] = None) -> float:
        """
        Calculate risk score based on anomaly detection
        
        Args:
            wallet_data: Wallet transaction data
            anomaly_results: Pre-computed anomaly results
            
        Returns:
            Anomaly risk score (0-1)
        """
        if anomaly_results is None and self.anomaly_detector:
            # Run anomaly detection
            anomaly_results = self.anomaly_detector.fit_predict(wallet_data)
        
        if anomaly_results is None:
            return 0.5  # Default moderate risk
        
        anomaly_risk = 0.0
        
        # Check if wallet is flagged as anomaly
        if len(anomaly_results.predictions) > 0 and anomaly_results.predictions[0] == -1:
            # This wallet is an anomaly
            anomaly_score = abs(anomaly_results.anomaly_scores[0])
            
            # Convert anomaly score to risk (lower anomaly scores = higher risk)
            # Normalize based on score distribution
            score_mean = anomaly_results.statistics.get('mean_anomaly_score', 0)
            score_std = anomaly_results.statistics.get('std_anomaly_score', 1)
            
            normalized_score = (score_mean - anomaly_score) / max(score_std, 0.1)
            anomaly_risk = min(max(normalized_score, 0), 1)
        
        return anomaly_risk
    
    def calculate_nlp_risk(self, text_data: List[str], 
                          nlp_results: Optional[List[TextAnalysisResult]] = None) -> float:
        """
        Calculate risk score based on NLP analysis
        
        Args:
            text_data: Text data associated with wallet (social media, etc.)
            nlp_results: Pre-computed NLP results
            
        Returns:
            NLP risk score (0-1)
        """
        if not text_data:
            return 0.0
        
        if nlp_results is None and self.nlp_processor:
            # Run NLP analysis
            nlp_results = self.nlp_processor.batch_analyze_texts(text_data)
        
        if not nlp_results:
            return 0.0
        
        nlp_risk = 0.0
        total_texts = len(nlp_results)
        
        for result in nlp_results:
            if result is None:
                continue
                
            # PII risk
            nlp_risk += result.pii_data.risk_score * 0.4
            
            # Toxicity risk
            nlp_risk += result.toxicity_score * 0.3
            
            # Suspicious content risk
            risk_indicators = len(result.classification_results.get('risk_indicators', []))
            nlp_risk += min(risk_indicators * 0.2, 0.6) * 0.2
            
            # Negative sentiment in crypto context
            if (result.classification_results.get('is_crypto_related', False) and
                result.sentiment.get('combined_label') == 'negative'):
                nlp_risk += 0.1
        
        # Average across all texts
        return min(nlp_risk / max(total_texts, 1), 1.0)
    
    def calculate_transaction_risk(self, wallet_data: pd.DataFrame) -> float:
        """
        Calculate risk based on transaction patterns
        
        Args:
            wallet_data: Wallet transaction data
            
        Returns:
            Transaction risk score (0-1)
        """
        if wallet_data.empty:
            return 0.0
        
        transaction_risk = 0.0
        
        # High-value transactions
        if 'value' in wallet_data.columns:
            max_value = wallet_data['value'].max()
            if max_value > 1000000:  # $1M threshold
                transaction_risk += 0.3
            elif max_value > 100000:  # $100K threshold
                transaction_risk += 0.2
        
        # Rapid succession of transactions (potential automation)
        if 'timestamp' in wallet_data.columns:
            wallet_data['timestamp'] = pd.to_datetime(wallet_data['timestamp'])
            time_diffs = wallet_data['timestamp'].diff().dt.total_seconds()
            rapid_txs = (time_diffs < 60).sum()  # Transactions within 1 minute
            
            if rapid_txs > 10:
                transaction_risk += 0.4
            elif rapid_txs > 5:
                transaction_risk += 0.2
        
        # Round number preference (structuring indicator)
        if 'value' in wallet_data.columns:
            round_numbers = (wallet_data['value'] % 1000 == 0).sum()
            total_txs = len(wallet_data)
            round_ratio = round_numbers / max(total_txs, 1)
            
            if round_ratio > 0.7:  # High preference for round numbers
                transaction_risk += 0.3
        
        # Mixing services indicators
        if 'to_address' in wallet_data.columns:
            # Look for known mixing service patterns
            unique_recipients = wallet_data['to_address'].nunique()
            total_txs = len(wallet_data)
            
            if unique_recipients / total_txs > 0.8:  # High diversity of recipients
                transaction_risk += 0.2
        
        return min(transaction_risk, 1.0)
    
    def calculate_behavioral_risk(self, wallet_data: pd.DataFrame) -> float:
        """
        Calculate risk based on behavioral patterns
        
        Args:
            wallet_data: Wallet transaction data
            
        Returns:
            Behavioral risk score (0-1)
        """
        if wallet_data.empty:
            return 0.0
        
        behavioral_risk = 0.0
        
        # Irregular activity patterns
        if 'timestamp' in wallet_data.columns:
            wallet_data['timestamp'] = pd.to_datetime(wallet_data['timestamp'])
            wallet_data['hour'] = wallet_data['timestamp'].dt.hour
            
            # Activity during unusual hours (2-6 AM)
            unusual_hours = ((wallet_data['hour'] >= 2) & (wallet_data['hour'] <= 6)).sum()
            unusual_ratio = unusual_hours / len(wallet_data)
            
            if unusual_ratio > 0.5:
                behavioral_risk += 0.3
        
        # Sudden change in transaction patterns
        if 'value' in wallet_data.columns and len(wallet_data) > 10:
            recent_txs = wallet_data.tail(10)
            historical_txs = wallet_data.head(len(wallet_data) - 10)
            
            if not historical_txs.empty:
                recent_avg = recent_txs['value'].mean()
                historical_avg = historical_txs['value'].mean()
                
                # Sudden increase in transaction values
                if recent_avg > historical_avg * 10:
                    behavioral_risk += 0.4
        
        # Low activity periods followed by high activity (dormancy activation)
        if 'timestamp' in wallet_data.columns and len(wallet_data) > 20:
            wallet_data = wallet_data.sort_values('timestamp')
            time_gaps = wallet_data['timestamp'].diff()
            
            # Look for gaps > 30 days followed by high activity
            long_gaps = time_gaps > timedelta(days=30)
            if long_gaps.sum() > 0:
                # Check activity after the gap
                gap_indices = wallet_data[long_gaps].index
                for idx in gap_indices:
                    if idx < len(wallet_data) - 10:
                        post_gap_activity = len(wallet_data[idx:idx+10])
                        if post_gap_activity >= 10:  # High activity after dormancy
                            behavioral_risk += 0.3
                            break
        
        return min(behavioral_risk, 1.0)
    
    def calculate_temporal_risk(self, wallet_data: pd.DataFrame) -> float:
        """
        Calculate risk based on temporal patterns
        
        Args:
            wallet_data: Wallet transaction data
            
        Returns:
            Temporal risk score (0-1)
        """
        if wallet_data.empty or 'timestamp' not in wallet_data.columns:
            return 0.0
        
        temporal_risk = 0.0
        wallet_data['timestamp'] = pd.to_datetime(wallet_data['timestamp'])
        
        # Activity during market crashes or regulatory announcements
        # (This would require external data feeds - simplified here)
        
        # Weekend activity (potentially suspicious for business accounts)
        wallet_data['weekday'] = wallet_data['timestamp'].dt.weekday
        weekend_txs = ((wallet_data['weekday'] == 5) | (wallet_data['weekday'] == 6)).sum()
        weekend_ratio = weekend_txs / len(wallet_data)
        
        if weekend_ratio > 0.8:  # Predominantly weekend activity
            temporal_risk += 0.2
        
        # Very recent creation with immediate high activity
        first_tx = wallet_data['timestamp'].min()
        days_since_first = (datetime.now() - first_tx).days
        
        if days_since_first < 7 and len(wallet_data) > 50:  # New wallet with high activity
            temporal_risk += 0.4
        
        # Seasonal patterns that don't match typical crypto behavior
        wallet_data['month'] = wallet_data['timestamp'].dt.month
        month_distribution = wallet_data['month'].value_counts()
        
        # Very concentrated activity in specific months (potential tax evasion timing)
        max_month_ratio = month_distribution.max() / len(wallet_data)
        if max_month_ratio > 0.6:
            temporal_risk += 0.2
        
        return min(temporal_risk, 1.0)
    
    def calculate_network_risk(self, wallet_data: pd.DataFrame) -> float:
        """
        Calculate risk based on network connections
        
        Args:
            wallet_data: Wallet transaction data
            
        Returns:
            Network risk score (0-1)
        """
        if wallet_data.empty:
            return 0.0
        
        network_risk = 0.0
        
        # Connections to known high-risk addresses (would require blacklist)
        # Simplified implementation
        
        # High fan-out (one-to-many transactions)
        if 'to_address' in wallet_data.columns:
            unique_recipients = wallet_data['to_address'].nunique()
            total_txs = len(wallet_data)
            
            fan_out_ratio = unique_recipients / total_txs
            if fan_out_ratio > 0.9:  # Almost all transactions to different addresses
                network_risk += 0.3
        
        # Circular transaction patterns
        if 'from_address' in wallet_data.columns and 'to_address' in wallet_data.columns:
            # Look for transactions that come back to the same address
            circular_txs = wallet_data[
                wallet_data['to_address'].isin(wallet_data['from_address'])
            ]
            
            if len(circular_txs) > len(wallet_data) * 0.3:  # >30% circular
                network_risk += 0.4
        
        return min(network_risk, 1.0)
    
    def calculate_compliance_risk(self, wallet_data: pd.DataFrame,
                                 jurisdiction: str = 'US') -> float:
        """
        Calculate compliance-related risk factors
        
        Args:
            wallet_data: Wallet transaction data
            jurisdiction: Regulatory jurisdiction
            
        Returns:
            Compliance risk score (0-1)
        """
        compliance_risk = 0.0
        
        # Transactions above reporting thresholds
        if 'value' in wallet_data.columns:
            # US $10K CTR threshold
            high_value_txs = (wallet_data['value'] >= 10000).sum()
            if high_value_txs > 0:
                compliance_risk += min(high_value_txs * 0.1, 0.5)
            
            # Structuring detection (multiple transactions just below threshold)
            near_threshold_txs = ((wallet_data['value'] >= 9000) & 
                                 (wallet_data['value'] < 10000)).sum()
            if near_threshold_txs > 5:
                compliance_risk += 0.6
        
        # Cross-border transaction indicators
        # (Would require geographic data - simplified here)
        
        return min(compliance_risk, 1.0)
    
    def assess_wallet_risk(self, wallet_address: str,
                          wallet_data: pd.DataFrame,
                          text_data: Optional[List[str]] = None,
                          external_data: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """
        Comprehensive risk assessment for a wallet
        
        Args:
            wallet_address: Wallet address to assess
            wallet_data: Transaction data for the wallet
            text_data: Associated text data (social media, etc.)
            external_data: Additional external data sources
            
        Returns:
            RiskAssessment object with comprehensive results
        """
        logger.info(f"Assessing risk for wallet: {wallet_address}")
        
        # Calculate individual risk factors
        risk_factors = RiskFactors()
        evidence = {}
        
        # Clustering risk
        try:
            risk_factors.clustering_risk = self.calculate_clustering_risk(wallet_data)
            evidence['clustering'] = {'score': risk_factors.clustering_risk}
        except Exception as e:
            logger.warning(f"Error calculating clustering risk: {e}")
            risk_factors.clustering_risk = 0.5
        
        # Anomaly risk
        try:
            risk_factors.anomaly_risk = self.calculate_anomaly_risk(wallet_data)
            evidence['anomaly'] = {'score': risk_factors.anomaly_risk}
        except Exception as e:
            logger.warning(f"Error calculating anomaly risk: {e}")
            risk_factors.anomaly_risk = 0.5
        
        # NLP risk
        try:
            risk_factors.nlp_risk = self.calculate_nlp_risk(text_data or [])
            evidence['nlp'] = {'score': risk_factors.nlp_risk}
        except Exception as e:
            logger.warning(f"Error calculating NLP risk: {e}")
            risk_factors.nlp_risk = 0.0
        
        # Transaction pattern risk
        try:
            risk_factors.transaction_risk = self.calculate_transaction_risk(wallet_data)
            evidence['transaction'] = {'score': risk_factors.transaction_risk}
        except Exception as e:
            logger.warning(f"Error calculating transaction risk: {e}")
            risk_factors.transaction_risk = 0.5
        
        # Behavioral risk
        try:
            risk_factors.behavioral_risk = self.calculate_behavioral_risk(wallet_data)
            evidence['behavioral'] = {'score': risk_factors.behavioral_risk}
        except Exception as e:
            logger.warning(f"Error calculating behavioral risk: {e}")
            risk_factors.behavioral_risk = 0.5
        
        # Temporal risk
        try:
            risk_factors.temporal_risk = self.calculate_temporal_risk(wallet_data)
            evidence['temporal'] = {'score': risk_factors.temporal_risk}
        except Exception as e:
            logger.warning(f"Error calculating temporal risk: {e}")
            risk_factors.temporal_risk = 0.5
        
        # Network risk
        try:
            risk_factors.network_risk = self.calculate_network_risk(wallet_data)
            evidence['network'] = {'score': risk_factors.network_risk}
        except Exception as e:
            logger.warning(f"Error calculating network risk: {e}")
            risk_factors.network_risk = 0.5
        
        # Compliance risk
        try:
            risk_factors.compliance_risk = self.calculate_compliance_risk(wallet_data)
            evidence['compliance'] = {'score': risk_factors.compliance_risk}
        except Exception as e:
            logger.warning(f"Error calculating compliance risk: {e}")
            risk_factors.compliance_risk = 0.5
        
        # Calculate overall risk score
        overall_score, confidence = self._calculate_overall_risk(risk_factors)
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_score)
        
        # Generate contributing factors and recommendations
        contributing_factors = self._identify_contributing_factors(risk_factors)
        recommendations = self._generate_recommendations(risk_factors, risk_level)
        
        assessment = RiskAssessment(
            wallet_address=wallet_address,
            overall_risk_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            risk_factors=risk_factors,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            evidence=evidence,
            assessment_timestamp=datetime.now()
        )
        
        logger.info(f"Risk assessment completed. Overall score: {overall_score:.3f}, Level: {risk_level}")
        return assessment
    
    def _calculate_overall_risk(self, risk_factors: RiskFactors) -> Tuple[float, float]:
        """
        Calculate overall risk score and confidence
        
        Args:
            risk_factors: Individual risk factor scores
            
        Returns:
            Tuple of (overall_score, confidence)
        """
        # Weighted average approach
        weights = self.feature_weights
        
        overall_score = (
            risk_factors.clustering_risk * weights.get('clustering', 0.15) +
            risk_factors.anomaly_risk * weights.get('anomaly', 0.20) +
            risk_factors.nlp_risk * weights.get('nlp', 0.15) +
            risk_factors.transaction_risk * weights.get('transaction', 0.20) +
            risk_factors.behavioral_risk * weights.get('behavioral', 0.15) +
            risk_factors.temporal_risk * weights.get('temporal', 0.10) +
            risk_factors.network_risk * weights.get('network', 0.05)
        )
        
        # Calculate confidence based on consistency of risk factors
        risk_scores = [
            risk_factors.clustering_risk,
            risk_factors.anomaly_risk,
            risk_factors.nlp_risk,
            risk_factors.transaction_risk,
            risk_factors.behavioral_risk,
            risk_factors.temporal_risk,
            risk_factors.network_risk
        ]
        
        # Confidence is higher when risk scores are consistent
        risk_std = np.std(risk_scores)
        base_confidence = 1.0 - risk_std  # Lower std = higher confidence
        
        # Adjust confidence based on data availability
        available_factors = sum(1 for score in risk_scores if score > 0)
        data_confidence = available_factors / len(risk_scores)
        
        confidence = min(base_confidence * data_confidence, 1.0)
        
        return overall_score, confidence
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        if risk_score >= self.risk_thresholds['high']:
            return 'HIGH'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        elif risk_score >= self.risk_thresholds['low']:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _identify_contributing_factors(self, risk_factors: RiskFactors) -> List[str]:
        """Identify the main contributing factors to the risk score"""
        factors = []
        
        if risk_factors.anomaly_risk > 0.5:
            factors.append("Anomalous transaction patterns detected")
        
        if risk_factors.transaction_risk > 0.5:
            factors.append("Suspicious transaction characteristics")
        
        if risk_factors.behavioral_risk > 0.5:
            factors.append("Irregular behavioral patterns")
        
        if risk_factors.clustering_risk > 0.5:
            factors.append("Associated with suspicious wallet clusters")
        
        if risk_factors.nlp_risk > 0.3:
            factors.append("Negative sentiment or PII in associated text data")
        
        if risk_factors.temporal_risk > 0.4:
            factors.append("Suspicious timing patterns")
        
        if risk_factors.network_risk > 0.4:
            factors.append("Concerning network connections")
        
        if risk_factors.compliance_risk > 0.5:
            factors.append("Potential compliance violations")
        
        return factors
    
    def _generate_recommendations(self, risk_factors: RiskFactors, 
                                 risk_level: str) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == 'HIGH':
            recommendations.append("IMMEDIATE ACTION REQUIRED: Flag for manual investigation")
            recommendations.append("Consider freezing account pending review")
            recommendations.append("File Suspicious Activity Report (SAR) if applicable")
        
        elif risk_level == 'MEDIUM':
            recommendations.append("Enhanced monitoring recommended")
            recommendations.append("Request additional documentation")
            recommendations.append("Increase transaction review frequency")
        
        elif risk_level == 'LOW':
            recommendations.append("Continue routine monitoring")
            recommendations.append("Periodic risk reassessment suggested")
        
        # Specific recommendations based on risk factors
        if risk_factors.compliance_risk > 0.5:
            recommendations.append("Review against AML/KYC requirements")
        
        if risk_factors.transaction_risk > 0.5:
            recommendations.append("Analyze transaction patterns for structuring")
        
        if risk_factors.nlp_risk > 0.3:
            recommendations.append("Review associated social media/communication for threats")
        
        return recommendations
    
    def batch_assess_wallets(self, wallet_data: Dict[str, pd.DataFrame],
                            text_data: Optional[Dict[str, List[str]]] = None) -> Dict[str, RiskAssessment]:
        """
        Assess risk for multiple wallets in batch
        
        Args:
            wallet_data: Dictionary mapping wallet addresses to their transaction data
            text_data: Optional dictionary mapping wallet addresses to associated text
            
        Returns:
            Dictionary mapping wallet addresses to their risk assessments
        """
        assessments = {}
        total_wallets = len(wallet_data)
        
        logger.info(f"Starting batch risk assessment for {total_wallets} wallets")
        
        for i, (address, data) in enumerate(wallet_data.items()):
            try:
                associated_text = text_data.get(address, []) if text_data else []
                assessment = self.assess_wallet_risk(address, data, associated_text)
                assessments[address] = assessment
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{total_wallets} wallets")
                    
            except Exception as e:
                logger.error(f"Error assessing wallet {address}: {e}")
                continue
        
        logger.info(f"Batch assessment completed. {len(assessments)} wallets processed")
        return assessments
    
    def export_risk_assessments(self, assessments: Dict[str, RiskAssessment],
                               output_path: str) -> None:
        """
        Export risk assessments to CSV
        
        Args:
            assessments: Dictionary of risk assessments
            output_path: Path to save CSV file
        """
        data = []
        
        for address, assessment in assessments.items():
            row = {
                'wallet_address': address,
                'overall_risk_score': assessment.overall_risk_score,
                'risk_level': assessment.risk_level,
                'confidence': assessment.confidence,
                'clustering_risk': assessment.risk_factors.clustering_risk,
                'anomaly_risk': assessment.risk_factors.anomaly_risk,
                'nlp_risk': assessment.risk_factors.nlp_risk,
                'transaction_risk': assessment.risk_factors.transaction_risk,
                'behavioral_risk': assessment.risk_factors.behavioral_risk,
                'temporal_risk': assessment.risk_factors.temporal_risk,
                'network_risk': assessment.risk_factors.network_risk,
                'compliance_risk': assessment.risk_factors.compliance_risk,
                'contributing_factors': '; '.join(assessment.contributing_factors),
                'recommendations': '; '.join(assessment.recommendations[:3]),  # Top 3 recommendations
                'assessment_timestamp': assessment.assessment_timestamp.isoformat()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Risk assessments exported to {output_path}")
    
    def get_risk_statistics(self, assessments: Dict[str, RiskAssessment]) -> Dict[str, Any]:
        """
        Generate statistics from risk assessments
        
        Args:
            assessments: Dictionary of risk assessments
            
        Returns:
            Dictionary with risk statistics
        """
        if not assessments:
            return {}
        
        risk_scores = [a.overall_risk_score for a in assessments.values()]
        risk_levels = [a.risk_level for a in assessments.values()]
        
        from collections import Counter
        risk_distribution = Counter(risk_levels)
        
        statistics = {
            'total_wallets_assessed': len(assessments),
            'risk_score_statistics': {
                'mean': float(np.mean(risk_scores)),
                'median': float(np.median(risk_scores)),
                'std': float(np.std(risk_scores)),
                'min': float(np.min(risk_scores)),
                'max': float(np.max(risk_scores))
            },
            'risk_level_distribution': {
                'HIGH': risk_distribution.get('HIGH', 0),
                'MEDIUM': risk_distribution.get('MEDIUM', 0),
                'LOW': risk_distribution.get('LOW', 0),
                'MINIMAL': risk_distribution.get('MINIMAL', 0)
            },
            'high_risk_percentage': (risk_distribution.get('HIGH', 0) / len(assessments)) * 100
        }
        
        # Most common contributing factors
        all_factors = []
        for assessment in assessments.values():
            all_factors.extend(assessment.contributing_factors)
        
        factor_counts = Counter(all_factors)
        statistics['top_risk_factors'] = [
            {'factor': factor, 'count': count}
            for factor, count in factor_counts.most_common(10)
        ]
        
        return statistics
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained risk scoring model
        
        Args:
            model_path: Path to save the model
        """
        model_data = {
            'config': self.config,
            'risk_thresholds': self.risk_thresholds,
            'feature_weights': self.feature_weights,
            'is_trained': self.is_trained,
            'ensemble_models': self.ensemble_models,
            'scalers': self.scalers
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Risk scoring model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained risk scoring model
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.config = model_data['config']
        self.risk_thresholds = model_data['risk_thresholds']
        self.feature_weights = model_data['feature_weights']
        self.is_trained = model_data['is_trained']
        self.ensemble_models = model_data.get('ensemble_models', {})
        self.scalers = model_data.get('scalers', {})
        
        logger.info(f"Risk scoring model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the risk scoring model
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_type': 'RiskScorer',
            'is_initialized': self.clustering_model is not None,
            'is_trained': self.is_trained,
            'config': self.config,
            'risk_thresholds': self.risk_thresholds,
            'feature_weights': self.feature_weights,
            'ensemble_models': list(self.ensemble_models.keys()) if self.ensemble_models else [],
            'sub_models': {
                'clustering_model': self.clustering_model is not None,
                'anomaly_detector': self.anomaly_detector is not None,
                'nlp_processor': self.nlp_processor is not None,
                'data_preprocessor': self.data_preprocessor is not None
            }
        }
        
        return info
