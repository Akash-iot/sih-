"""
Custom ML Risk Scoring Pipeline for ETHEREYE
Implements advanced machine learning models for suspicious wallet classification
and comprehensive risk assessment using ensemble methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import joblib
import json
from pathlib import Path

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class BlockchainRiskScoringService:
    """
    Advanced ML-based risk scoring service for blockchain wallet analysis
    Uses ensemble methods and custom features for suspicious activity detection
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.label_encoder = LabelEncoder()
        
        # Model performance tracking
        self.model_metrics = {}
        self.training_history = []
        
        # Feature engineering parameters
        self.feature_columns = [
            # Basic features
            'balance_eth', 'balance_usd', 'transaction_count', 'unique_addresses',
            
            # Transaction patterns
            'avg_transaction_value', 'median_transaction_value', 'total_volume',
            'transaction_frequency', 'night_activity_ratio', 'weekend_activity_ratio',
            
            # Behavioral features
            'round_number_ratio', 'burst_activity_score', 'dormancy_periods',
            'gas_price_patterns', 'time_between_transactions_std',
            
            # Network features  
            'incoming_tx_ratio', 'outgoing_tx_ratio', 'self_transfer_ratio',
            'contract_interaction_ratio', 'token_diversity_score',
            
            # Risk indicators
            'mixer_interaction_count', 'exchange_interaction_count',
            'gambling_interaction_count', 'darknet_interaction_count',
            'suspicious_counterparty_ratio', 'blacklist_interaction_count',
            
            # Advanced features
            'address_reuse_score', 'temporal_clustering_score',
            'value_clustering_score', 'privacy_seeking_behavior_score'
        ]
        
        # Risk categories
        self.risk_categories = {
            0: 'legitimate',
            1: 'low_risk', 
            2: 'medium_risk',
            3: 'high_risk',
            4: 'critical_risk'
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble of ML models"""
        try:
            # Random Forest - Good for feature importance
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Gradient Boosting - Strong performance on tabular data
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
            # Logistic Regression - Interpretable baseline
            self.models['logistic_regression'] = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )
            
            # SVM - Good for complex decision boundaries
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced',
                C=1.0,
                gamma='scale'
            )
            
            # Neural Network - Can capture complex patterns
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.001
            )
            
            # Voting Classifier - Combines multiple models
            self.models['ensemble'] = VotingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('lr', self.models['logistic_regression'])
                ],
                voting='soft'
            )
            
            # Isolation Forest for anomaly detection
            self.models['isolation_forest'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize scalers
            self.scalers['standard'] = StandardScaler()
            self.scalers['robust'] = RobustScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def extract_risk_features(self, wallet_data: Dict) -> Dict[str, float]:
        """
        Extract comprehensive risk features from wallet data
        
        Args:
            wallet_data: Dictionary containing wallet information
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Basic features
            features['balance_eth'] = float(wallet_data.get('balance_eth', 0))
            features['balance_usd'] = float(wallet_data.get('balance_usd', 0))
            features['transaction_count'] = int(wallet_data.get('transaction_count', 0))
            
            transactions = wallet_data.get('transactions', [])
            
            if not transactions:
                # Return default features for wallets without transactions
                return {col: 0.0 for col in self.feature_columns}
            
            # Transaction value analysis
            tx_values = [float(tx.get('value_eth', 0)) for tx in transactions]
            tx_values = [v for v in tx_values if v > 0]  # Remove zero-value transactions
            
            if tx_values:
                features['avg_transaction_value'] = np.mean(tx_values)
                features['median_transaction_value'] = np.median(tx_values)
                features['total_volume'] = sum(tx_values)
                
                # Round number analysis
                round_numbers = sum(1 for val in tx_values if self._is_round_number(val))
                features['round_number_ratio'] = round_numbers / len(tx_values)
                
                # Value clustering analysis
                features['value_clustering_score'] = self._calculate_value_clustering(tx_values)
            else:
                features['avg_transaction_value'] = 0
                features['median_transaction_value'] = 0
                features['total_volume'] = 0
                features['round_number_ratio'] = 0
                features['value_clustering_score'] = 0
            
            # Temporal analysis
            timestamps = [tx.get('timestamp') for tx in transactions if tx.get('timestamp')]
            
            if timestamps:
                # Convert to datetime objects
                dt_timestamps = [pd.to_datetime(ts) for ts in timestamps]
                dt_timestamps.sort()
                
                # Transaction frequency
                if len(dt_timestamps) > 1:
                    time_span = (dt_timestamps[-1] - dt_timestamps[0]).total_seconds()
                    features['transaction_frequency'] = len(transactions) / max(time_span / 86400, 1)  # txs per day
                    
                    # Time between transactions analysis
                    time_diffs = [(dt_timestamps[i] - dt_timestamps[i-1]).total_seconds() 
                                for i in range(1, len(dt_timestamps))]
                    features['time_between_transactions_std'] = np.std(time_diffs)
                else:
                    features['transaction_frequency'] = 0
                    features['time_between_transactions_std'] = 0
                
                # Night activity analysis (UTC)
                night_txs = sum(1 for dt in dt_timestamps if dt.hour >= 22 or dt.hour <= 6)
                features['night_activity_ratio'] = night_txs / len(dt_timestamps)
                
                # Weekend activity analysis
                weekend_txs = sum(1 for dt in dt_timestamps if dt.weekday() >= 5)
                features['weekend_activity_ratio'] = weekend_txs / len(dt_timestamps)
                
                # Burst activity analysis
                features['burst_activity_score'] = self._calculate_burst_activity(dt_timestamps)
                
                # Dormancy analysis
                features['dormancy_periods'] = self._calculate_dormancy_periods(dt_timestamps)
                
                # Temporal clustering
                features['temporal_clustering_score'] = self._calculate_temporal_clustering(dt_timestamps)
            else:
                features['transaction_frequency'] = 0
                features['time_between_transactions_std'] = 0
                features['night_activity_ratio'] = 0
                features['weekend_activity_ratio'] = 0
                features['burst_activity_score'] = 0
                features['dormancy_periods'] = 0
                features['temporal_clustering_score'] = 0
            
            # Address interaction analysis
            from_addresses = set()
            to_addresses = set()
            self_transfers = 0
            
            for tx in transactions:
                if tx.get('from_address'):
                    from_addresses.add(tx['from_address'])
                if tx.get('to_address'):
                    to_addresses.add(tx['to_address'])
                
                # Check for self-transfers
                if (tx.get('from_address') == wallet_data.get('address') and 
                    tx.get('to_address') == wallet_data.get('address')):
                    self_transfers += 1
            
            unique_addresses = len(from_addresses.union(to_addresses))
            features['unique_addresses'] = unique_addresses
            
            # Transaction direction analysis
            incoming_txs = sum(1 for tx in transactions if tx.get('direction') == 'incoming')
            outgoing_txs = sum(1 for tx in transactions if tx.get('direction') == 'outgoing')
            total_directional = incoming_txs + outgoing_txs
            
            if total_directional > 0:
                features['incoming_tx_ratio'] = incoming_txs / total_directional
                features['outgoing_tx_ratio'] = outgoing_txs / total_directional
                features['self_transfer_ratio'] = self_transfers / total_directional
            else:
                features['incoming_tx_ratio'] = 0
                features['outgoing_tx_ratio'] = 0
                features['self_transfer_ratio'] = 0
            
            # Contract and token interaction analysis
            contract_txs = sum(1 for tx in transactions if tx.get('is_contract_interaction', False))
            features['contract_interaction_ratio'] = contract_txs / len(transactions)
            
            # Token diversity
            token_types = set()
            for tx in transactions:
                if tx.get('token_symbol'):
                    token_types.add(tx['token_symbol'])
            features['token_diversity_score'] = len(token_types)
            
            # Gas price analysis
            gas_prices = [int(tx.get('gas_price', 0)) for tx in transactions if tx.get('gas_price')]
            if gas_prices:
                features['gas_price_patterns'] = np.std(gas_prices) / np.mean(gas_prices)
            else:
                features['gas_price_patterns'] = 0
            
            # Risk indicator analysis
            features.update(self._analyze_risk_indicators(transactions, wallet_data.get('address', '')))
            
            # Advanced behavioral features
            features['address_reuse_score'] = self._calculate_address_reuse_score(transactions)
            features['privacy_seeking_behavior_score'] = self._calculate_privacy_behavior_score(transactions)
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in features:
                    features[col] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting risk features: {e}")
            return {col: 0.0 for col in self.feature_columns}
    
    def _is_round_number(self, value: float, tolerance: float = 1e-6) -> bool:
        """Check if a number is likely a round number"""
        if value == 0:
            return True
        
        round_patterns = [0.1, 0.5, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, 10000]
        for pattern in round_patterns:
            if abs(value - pattern) < tolerance:
                return True
            if value > pattern and abs(value % pattern) < tolerance:
                return True
        
        return False
    
    def _calculate_value_clustering(self, values: List[float]) -> float:
        """Calculate how clustered transaction values are"""
        if len(values) < 2:
            return 0.0
        
        # Use coefficient of variation
        return np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
    
    def _calculate_burst_activity(self, timestamps: List[datetime]) -> float:
        """Calculate burst activity score"""
        if len(timestamps) < 3:
            return 0.0
        
        # Look for periods with high transaction density
        time_windows = []
        window_size = timedelta(hours=1)
        
        for i, ts in enumerate(timestamps):
            count_in_window = sum(1 for t in timestamps if abs((t - ts).total_seconds()) <= window_size.total_seconds())
            time_windows.append(count_in_window)
        
        return max(time_windows) / len(timestamps) if time_windows else 0
    
    def _calculate_dormancy_periods(self, timestamps: List[datetime]) -> float:
        """Calculate number of dormancy periods"""
        if len(timestamps) < 2:
            return 0
        
        timestamps.sort()
        dormancy_threshold = timedelta(days=30)
        dormancy_periods = 0
        
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] > dormancy_threshold:
                dormancy_periods += 1
        
        return dormancy_periods
    
    def _calculate_temporal_clustering(self, timestamps: List[datetime]) -> float:
        """Calculate temporal clustering score"""
        if len(timestamps) < 3:
            return 0.0
        
        timestamps.sort()
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        
        # Calculate coefficient of variation for time differences
        return np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
    
    def _analyze_risk_indicators(self, transactions: List[Dict], address: str) -> Dict[str, float]:
        """Analyze various risk indicators"""
        risk_features = {}
        
        # Known service categories (in practice, these would be from external databases)
        mixer_keywords = ['mixer', 'tornado', 'wasabi', 'samourai']
        exchange_keywords = ['binance', 'coinbase', 'kraken', 'huobi', 'okex']
        gambling_keywords = ['casino', 'dice', 'lottery', 'bet']
        darknet_keywords = ['silk', 'alpha', 'dream', 'wall']
        
        # Count interactions with different service types
        mixer_interactions = 0
        exchange_interactions = 0
        gambling_interactions = 0
        darknet_interactions = 0
        blacklist_interactions = 0
        suspicious_counterparties = 0
        
        for tx in transactions:
            tx_memo = str(tx.get('memo', '')).lower()
            counterparty = tx.get('counterparty_address', '')
            
            # Check for mixer interactions
            if any(keyword in tx_memo for keyword in mixer_keywords):
                mixer_interactions += 1
            
            # Check for exchange interactions
            if any(keyword in tx_memo for keyword in exchange_keywords):
                exchange_interactions += 1
            
            # Check for gambling interactions
            if any(keyword in tx_memo for keyword in gambling_keywords):
                gambling_interactions += 1
            
            # Check for darknet interactions
            if any(keyword in tx_memo for keyword in darknet_keywords):
                darknet_interactions += 1
            
            # Simulate blacklist check (in practice, would check against real blacklists)
            if self._is_suspicious_address(counterparty):
                suspicious_counterparties += 1
            
            # Simulate known bad address check
            if self._is_blacklisted_address(counterparty):
                blacklist_interactions += 1
        
        # Calculate ratios
        total_txs = len(transactions)
        risk_features['mixer_interaction_count'] = mixer_interactions
        risk_features['exchange_interaction_count'] = exchange_interactions  
        risk_features['gambling_interaction_count'] = gambling_interactions
        risk_features['darknet_interaction_count'] = darknet_interactions
        risk_features['blacklist_interaction_count'] = blacklist_interactions
        risk_features['suspicious_counterparty_ratio'] = suspicious_counterparties / total_txs if total_txs > 0 else 0
        
        return risk_features
    
    def _is_suspicious_address(self, address: str) -> bool:
        """Check if address shows suspicious patterns (simplified)"""
        if not address:
            return False
        
        # Simple heuristics (in practice, would use more sophisticated detection)
        return (
            len(set(address.lower())) < 10 or  # Low character diversity
            address.lower().count('0') > len(address) * 0.7  # Too many zeros
        )
    
    def _is_blacklisted_address(self, address: str) -> bool:
        """Check against blacklist (simplified)"""
        # In practice, this would check against real blacklists/databases
        blacklisted_patterns = ['000000', '111111', 'aaaaaa', 'ffffff']
        return any(pattern in address.lower() for pattern in blacklisted_patterns)
    
    def _calculate_address_reuse_score(self, transactions: List[Dict]) -> float:
        """Calculate address reuse patterns"""
        addresses = []
        for tx in transactions:
            if tx.get('from_address'):
                addresses.append(tx['from_address'])
            if tx.get('to_address'):
                addresses.append(tx['to_address'])
        
        if not addresses:
            return 0.0
        
        unique_addresses = len(set(addresses))
        return 1.0 - (unique_addresses / len(addresses))
    
    def _calculate_privacy_behavior_score(self, transactions: List[Dict]) -> float:
        """Calculate privacy-seeking behavior score"""
        privacy_indicators = 0
        total_txs = len(transactions)
        
        if total_txs == 0:
            return 0.0
        
        for tx in transactions:
            memo = str(tx.get('memo', '')).lower()
            
            # Check for privacy-related keywords
            privacy_keywords = ['private', 'anonymous', 'untraceable', 'mixer', 'privacy']
            if any(keyword in memo for keyword in privacy_keywords):
                privacy_indicators += 1
            
            # Check for unusual gas prices (privacy-conscious users might pay extra)
            gas_price = int(tx.get('gas_price', 0))
            if gas_price > 50:  # Above average gas price
                privacy_indicators += 0.5
        
        return privacy_indicators / total_txs
    
    def train_risk_model(
        self, 
        training_data: List[Dict], 
        labels: List[str],
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the risk scoring model on labeled data
        
        Args:
            training_data: List of wallet data dictionaries
            labels: List of risk labels ('legitimate', 'low_risk', etc.)
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info(f"Training risk model on {len(training_data)} samples")
            
            # Extract features for all wallets
            features_list = []
            for wallet_data in training_data:
                features = self.extract_risk_features(wallet_data)
                features_list.append([features[col] for col in self.feature_columns])
            
            X = np.array(features_list)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(labels)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Train models
            model_results = {}
            
            for model_name, model in self.models.items():
                if model_name == 'isolation_forest':
                    # Isolation Forest is unsupervised
                    model.fit(X_train_selected)
                    anomaly_scores = model.decision_function(X_test_selected)
                    predictions = model.predict(X_test_selected)
                    
                    model_results[model_name] = {
                        'anomaly_detection': True,
                        'anomaly_scores': anomaly_scores.tolist(),
                        'predictions': predictions.tolist()
                    }
                else:
                    # Supervised models
                    model.fit(X_train_selected, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = None
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_selected)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv_folds)
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        selected_features = self.feature_selector.get_support(indices=True)
                        feature_importance = {
                            self.feature_columns[idx]: importance 
                            for idx, importance in zip(selected_features, model.feature_importances_)
                        }
                    
                    model_results[model_name] = {
                        'accuracy': float(accuracy),
                        'f1_score': float(f1),
                        'cv_mean': float(cv_scores.mean()),
                        'cv_std': float(cv_scores.std()),
                        'feature_importance': feature_importance,
                        'classification_report': classification_report(y_test, y_pred, output_dict=True)
                    }
            
            # Store metrics
            self.model_metrics = model_results
            
            # Record training history
            training_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'samples': len(training_data),
                'features': len(self.feature_columns),
                'selected_features': self.feature_selector.k,
                'test_size': test_size,
                'cv_folds': cv_folds,
                'label_distribution': {
                    self.label_encoder.classes_[i]: int(np.sum(y_encoded == i))
                    for i in range(len(self.label_encoder.classes_))
                }
            }
            self.training_history.append(training_record)
            
            logger.info("Model training completed successfully")
            
            return {
                'training_results': model_results,
                'training_record': training_record,
                'best_model': max(
                    [(name, results) for name, results in model_results.items() 
                     if 'accuracy' in results],
                    key=lambda x: x[1]['accuracy']
                )[0] if any('accuracy' in results for results in model_results.values()) else 'ensemble',
                'feature_selection': {
                    'total_features': len(self.feature_columns),
                    'selected_features': self.feature_selector.k,
                    'selected_feature_names': [
                        self.feature_columns[i] for i in self.feature_selector.get_support(indices=True)
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error training risk model: {e}")
            return {
                'error': f"Training failed: {str(e)}",
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def predict_risk_score(
        self, 
        wallet_data: Dict, 
        model_name: str = 'ensemble',
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict risk score for a wallet using trained models
        
        Args:
            wallet_data: Dictionary containing wallet information
            model_name: Name of model to use for prediction
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing risk prediction results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Check if model is trained
            if not hasattr(model, 'classes_') and model_name != 'isolation_forest':
                raise ValueError(f"Model {model_name} is not trained yet")
            
            # Extract features
            features = self.extract_risk_features(wallet_data)
            X = np.array([[features[col] for col in self.feature_columns]])
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scalers['standard'].transform(X)
            
            # Apply feature selection
            if self.feature_selector:
                X_selected = self.feature_selector.transform(X_scaled)
            else:
                X_selected = X_scaled
            
            result = {
                'wallet_address': wallet_data.get('address', 'unknown'),
                'model_used': model_name,
                'features_extracted': features,
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
            if model_name == 'isolation_forest':
                # Anomaly detection
                anomaly_score = model.decision_function(X_selected)[0]
                is_anomaly = model.predict(X_selected)[0] == -1
                
                result.update({
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'risk_category': 'high_risk' if is_anomaly else 'legitimate',
                    'confidence': abs(float(anomaly_score))
                })
            else:
                # Classification
                prediction = model.predict(X_selected)[0]
                risk_category = self.label_encoder.inverse_transform([prediction])[0]
                
                result.update({
                    'predicted_class': int(prediction),
                    'risk_category': risk_category,
                    'risk_level': self._get_risk_level_score(risk_category)
                })
                
                # Add probabilities if requested and available
                if return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_selected)[0]
                    prob_dict = {
                        self.label_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(probabilities)
                    }
                    result['class_probabilities'] = prob_dict
                    result['confidence'] = float(max(probabilities))
            
            # Add feature importance for interpretation
            if hasattr(model, 'feature_importances_') and self.feature_selector:
                selected_features = self.feature_selector.get_support(indices=True)
                top_features = {}
                for idx, importance in enumerate(model.feature_importances_):
                    if importance > 0.01:  # Only include significant features
                        feature_name = self.feature_columns[selected_features[idx]]
                        top_features[feature_name] = {
                            'importance': float(importance),
                            'value': features[feature_name]
                        }
                
                # Sort by importance
                result['top_risk_factors'] = dict(sorted(
                    top_features.items(), 
                    key=lambda x: x[1]['importance'], 
                    reverse=True
                )[:10])
            
            # Add risk explanation
            result['risk_explanation'] = self._generate_risk_explanation(features, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting risk score: {e}")
            return {
                'error': f"Risk prediction failed: {str(e)}",
                'wallet_address': wallet_data.get('address', 'unknown'),
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_risk_level_score(self, risk_category: str) -> float:
        """Convert risk category to numerical score"""
        risk_scores = {
            'legitimate': 0.1,
            'low_risk': 0.3,
            'medium_risk': 0.5,
            'high_risk': 0.8,
            'critical_risk': 1.0
        }
        return risk_scores.get(risk_category, 0.5)
    
    def _generate_risk_explanation(self, features: Dict, prediction_result: Dict) -> List[str]:
        """Generate human-readable risk explanation"""
        explanations = []
        
        # High-risk features
        if features.get('mixer_interaction_count', 0) > 0:
            explanations.append(f"Interacted with {features['mixer_interaction_count']} potential mixer services")
        
        if features.get('blacklist_interaction_count', 0) > 0:
            explanations.append(f"Interacted with {features['blacklist_interaction_count']} blacklisted addresses")
        
        if features.get('suspicious_counterparty_ratio', 0) > 0.3:
            explanations.append(f"High ratio of suspicious counterparties ({features['suspicious_counterparty_ratio']:.2%})")
        
        # Pattern-based risks
        if features.get('round_number_ratio', 0) > 0.7:
            explanations.append("High frequency of round-number transactions (potential structuring)")
        
        if features.get('night_activity_ratio', 0) > 0.8:
            explanations.append("Unusual activity patterns (high nighttime activity)")
        
        if features.get('burst_activity_score', 0) > 0.5:
            explanations.append("Burst activity pattern detected")
        
        if features.get('privacy_seeking_behavior_score', 0) > 0.3:
            explanations.append("Privacy-seeking behavior indicators present")
        
        # Positive indicators
        if features.get('exchange_interaction_count', 0) > 5:
            explanations.append("Regular exchange interactions (lower risk)")
        
        if features.get('transaction_frequency', 0) > 0 and features.get('transaction_frequency', 0) < 10:
            explanations.append("Normal transaction frequency")
        
        if not explanations:
            explanations.append("No significant risk indicators detected")
        
        return explanations
    
    def batch_predict_risk(self, wallets_data: List[Dict], model_name: str = 'ensemble') -> Dict[str, Any]:
        """
        Predict risk scores for multiple wallets
        
        Args:
            wallets_data: List of wallet data dictionaries
            model_name: Name of model to use
            
        Returns:
            Dictionary containing batch prediction results
        """
        try:
            batch_results = []
            summary_stats = {
                'total_wallets': len(wallets_data),
                'processing_errors': 0,
                'risk_distribution': {}
            }
            
            for i, wallet_data in enumerate(wallets_data):
                try:
                    result = self.predict_risk_score(wallet_data, model_name, return_probabilities=False)
                    
                    if 'error' not in result:
                        batch_results.append(result)
                        
                        # Update risk distribution
                        risk_category = result.get('risk_category', 'unknown')
                        summary_stats['risk_distribution'][risk_category] = (
                            summary_stats['risk_distribution'].get(risk_category, 0) + 1
                        )
                    else:
                        summary_stats['processing_errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing wallet {i}: {e}")
                    summary_stats['processing_errors'] += 1
                    continue
            
            return {
                'batch_predictions': batch_results,
                'summary_statistics': summary_stats,
                'model_used': model_name,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'model_performance': self.model_metrics.get(model_name, {})
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {
                'error': f"Batch prediction failed: {str(e)}",
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            'available_models': list(self.models.keys()),
            'trained_models': [name for name, model in self.models.items() 
                             if hasattr(model, 'classes_') or name == 'isolation_forest'],
            'feature_columns': self.feature_columns,
            'risk_categories': self.risk_categories,
            'model_metrics': self.model_metrics,
            'training_history': self.training_history,
            'feature_selection_enabled': self.feature_selector is not None
        }
    
    def save_models(self, filepath: str):
        """Save trained models and preprocessors"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_selector': self.feature_selector,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'risk_categories': self.risk_categories,
                'model_metrics': self.model_metrics,
                'training_history': self.training_history,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models and preprocessors"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_selector = model_data['feature_selector']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.risk_categories = model_data['risk_categories']
            self.model_metrics = model_data['model_metrics']
            self.training_history = model_data['training_history']
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")