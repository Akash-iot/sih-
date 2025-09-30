"""
Isolation Forest Anomaly Detection Module for ETHEREYE
=====================================================

This module implements Isolation Forest for detecting anomalous blockchain activities:
- Suspicious transaction patterns
- Unusual wallet behaviors  
- Potential money laundering activities
- Fraudulent transactions
- Coordinated attacks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResults:
    """Container for anomaly detection results"""
    predictions: np.ndarray  # 1 for normal, -1 for anomaly
    anomaly_scores: np.ndarray  # Lower scores indicate more anomalous
    anomaly_indices: np.ndarray  # Indices of detected anomalies
    feature_importance: Dict[str, float]  # Feature contributions to anomalies
    threshold: float  # Decision threshold used
    n_anomalies: int  # Number of detected anomalies
    anomaly_rate: float  # Percentage of anomalies
    statistics: Dict[str, Any]  # Various statistics

class IsolationForestDetector:
    """
    Isolation Forest implementation for blockchain anomaly detection
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 contamination: Union[float, str] = 0.1,
                 max_samples: Union[int, float, str] = 'auto',
                 max_features: Union[int, float] = 1.0,
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            n_estimators: Number of base estimators in the ensemble
            contamination: Proportion of outliers in the data set
            max_samples: Number of samples to draw from X to train each base estimator
            max_features: Number of features to draw from X to train each base estimator
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.results = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            scale_features: bool = True) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model
        
        Args:
            X: Input data
            feature_columns: Specific columns to use for detection
            scale_features: Whether to scale features before training
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Isolation Forest model...")
        
        # Prepare data
        if feature_columns:
            self.feature_columns = feature_columns
            X_features = X[feature_columns]
        else:
            X_features = X.select_dtypes(include=[np.number])
            self.feature_columns = X_features.columns.tolist()
        
        # Handle missing values
        X_features = X_features.fillna(X_features.median())
        
        # Scale features if requested
        if scale_features:
            self.scaler = RobustScaler()  # Use RobustScaler as it's less sensitive to outliers
            X_processed = self.scaler.fit_transform(X_features)
        else:
            X_processed = X_features.values
        
        # Initialize and fit the model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(X_processed)
        self.is_fitted = True
        
        logger.info(f"Model trained successfully with {len(self.feature_columns)} features")
        return self
    
    def predict(self, X: pd.DataFrame, 
                return_scores: bool = True) -> Union[np.ndarray, AnomalyResults]:
        """
        Predict anomalies in the data
        
        Args:
            X: Input data to predict on
            return_scores: Whether to return detailed results
            
        Returns:
            Anomaly predictions or detailed results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data
        X_features = X[self.feature_columns].fillna(X[self.feature_columns].median())
        
        if self.scaler:
            X_processed = self.scaler.transform(X_features)
        else:
            X_processed = X_features.values
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        anomaly_scores = self.model.score_samples(X_processed)
        
        if not return_scores:
            return predictions
        
        # Calculate detailed results
        anomaly_indices = np.where(predictions == -1)[0]
        n_anomalies = len(anomaly_indices)
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        # Calculate feature importance (approximate)
        feature_importance = self._calculate_feature_importance(X_features, anomaly_scores)
        
        # Calculate statistics
        statistics = {
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'std_anomaly_score': float(np.std(anomaly_scores)),
            'min_anomaly_score': float(np.min(anomaly_scores)),
            'max_anomaly_score': float(np.max(anomaly_scores)),
            'threshold_score': float(np.percentile(anomaly_scores, (1 - self.contamination) * 100)) if isinstance(self.contamination, float) else None
        }
        
        self.results = AnomalyResults(
            predictions=predictions,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            feature_importance=feature_importance,
            threshold=statistics['threshold_score'],
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            statistics=statistics
        )
        
        logger.info(f"Detected {n_anomalies} anomalies ({anomaly_rate:.2f}% of data)")
        return self.results
    
    def fit_predict(self, X: pd.DataFrame,
                   feature_columns: Optional[List[str]] = None,
                   scale_features: bool = True) -> AnomalyResults:
        """
        Fit the model and predict anomalies in one step
        
        Args:
            X: Input data
            feature_columns: Specific columns to use
            scale_features: Whether to scale features
            
        Returns:
            Anomaly detection results
        """
        self.fit(X, feature_columns, scale_features)
        return self.predict(X, return_scores=True)
    
    def _calculate_feature_importance(self, X: pd.DataFrame, 
                                     anomaly_scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate approximate feature importance for anomaly detection
        """
        importance_scores = {}
        
        # Calculate correlation between each feature and anomaly scores
        for i, col in enumerate(self.feature_columns):
            correlation = np.abs(np.corrcoef(X[col], anomaly_scores)[0, 1])
            importance_scores[col] = correlation if not np.isnan(correlation) else 0.0
        
        # Normalize to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return importance_scores
    
    def evaluate_performance(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance when ground truth is available
        
        Args:
            X: Input features
            y_true: True anomaly labels (1 for normal, -1 for anomaly)
            
        Returns:
            Performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        results = self.predict(X, return_scores=True)
        y_pred = results.predictions
        
        # Convert to binary format for some metrics (0=normal, 1=anomaly)
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(np.mean(y_pred == y_true)),
            'precision': float(np.sum((y_pred == -1) & (y_true == -1)) / np.sum(y_pred == -1)) if np.sum(y_pred == -1) > 0 else 0.0,
            'recall': float(np.sum((y_pred == -1) & (y_true == -1)) / np.sum(y_true == -1)) if np.sum(y_true == -1) > 0 else 0.0,
            'f1_score': 0.0,
            'confusion_matrix': confusion_matrix(y_true_binary, y_pred_binary).tolist(),
            'classification_report': classification_report(y_true_binary, y_pred_binary, output_dict=True)
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Calculate AUC if possible
        try:
            # Use anomaly scores for AUC calculation (invert since lower scores = more anomalous)
            auc_scores = -results.anomaly_scores
            metrics['auc_roc'] = float(roc_auc_score(y_true_binary, auc_scores))
        except Exception:
            metrics['auc_roc'] = None
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, 
                      cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation to assess model stability
        
        Args:
            X: Input data
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Prepare data
        X_features = X[self.feature_columns].fillna(X[self.feature_columns].median())
        
        if self.scaler:
            X_processed = self.scaler.fit_transform(X_features)
        else:
            X_processed = X_features.values
        
        # Create a fresh model for CV
        cv_model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Custom scoring function for anomaly detection
        def anomaly_score(estimator, X):
            predictions = estimator.predict(X)
            # Return the proportion of inliers (higher is better for cross-validation)
            return np.mean(predictions == 1)
        
        scores = cross_val_score(cv_model, X_processed, cv=cv_folds, 
                               scoring=anomaly_score, n_jobs=-1)
        
        cv_results = {
            'scores': scores.tolist(),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }
        
        logger.info(f"Cross-validation completed. Mean score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
        return cv_results
    
    def optimize_contamination(self, X: pd.DataFrame,
                              contamination_range: List[float] = None) -> Dict[str, Any]:
        """
        Optimize the contamination parameter
        
        Args:
            X: Input data
            contamination_range: Range of contamination values to test
            
        Returns:
            Optimization results
        """
        if contamination_range is None:
            contamination_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        logger.info("Optimizing contamination parameter...")
        
        # Prepare data
        X_features = X[self.feature_columns].fillna(X[self.feature_columns].median())
        
        if self.scaler:
            X_processed = self.scaler.fit_transform(X_features)
        else:
            X_processed = X_features.values
        
        results = []
        best_score = -np.inf
        best_contamination = contamination_range[0]
        
        for contamination in contamination_range:
            try:
                # Fit model with current contamination
                model = IsolationForest(
                    n_estimators=self.n_estimators,
                    contamination=contamination,
                    max_samples=self.max_samples,
                    max_features=self.max_features,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                model.fit(X_processed)
                predictions = model.predict(X_processed)
                scores = model.score_samples(X_processed)
                
                n_anomalies = np.sum(predictions == -1)
                anomaly_rate = n_anomalies / len(predictions)
                
                # Score based on separation between normal and anomalous samples
                normal_scores = scores[predictions == 1]
                anomaly_scores = scores[predictions == -1]
                
                if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                    separation_score = np.mean(normal_scores) - np.mean(anomaly_scores)
                else:
                    separation_score = 0
                
                result = {
                    'contamination': contamination,
                    'n_anomalies': int(n_anomalies),
                    'anomaly_rate': float(anomaly_rate),
                    'mean_score': float(np.mean(scores)),
                    'separation_score': float(separation_score),
                    'score_std': float(np.std(scores))
                }
                
                results.append(result)
                
                # Update best parameters
                if separation_score > best_score:
                    best_score = separation_score
                    best_contamination = contamination
                    
            except Exception as e:
                logger.warning(f"Error with contamination={contamination}: {e}")
                continue
        
        optimization_results = {
            'best_contamination': best_contamination,
            'best_score': best_score,
            'all_results': results
        }
        
        logger.info(f"Best contamination: {best_contamination} (score: {best_score:.3f})")
        return optimization_results
    
    def get_anomaly_explanations(self, X: pd.DataFrame, 
                                anomaly_indices: Optional[List[int]] = None,
                                top_n: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        Get explanations for detected anomalies
        
        Args:
            X: Original data
            anomaly_indices: Specific indices to explain (default: all anomalies)
            top_n: Number of top contributing features to show
            
        Returns:
            Explanations for each anomaly
        """
        if self.results is None:
            raise ValueError("Must run prediction first")
        
        if anomaly_indices is None:
            anomaly_indices = self.results.anomaly_indices
        
        explanations = {}
        
        for idx in anomaly_indices[:top_n]:  # Limit to top_n for performance
            explanation = {
                'index': int(idx),
                'anomaly_score': float(self.results.anomaly_scores[idx]),
                'feature_values': {},
                'feature_deviations': {},
                'top_contributing_features': []
            }
            
            # Get feature values for this anomaly
            for col in self.feature_columns:
                value = X.iloc[idx][col]
                explanation['feature_values'][col] = float(value) if pd.notnull(value) else None
                
                # Calculate how much this deviates from the mean
                mean_val = X[col].mean()
                std_val = X[col].std()
                if std_val > 0:
                    deviation = abs(value - mean_val) / std_val
                    explanation['feature_deviations'][col] = float(deviation)
                else:
                    explanation['feature_deviations'][col] = 0.0
            
            # Sort features by deviation
            sorted_features = sorted(explanation['feature_deviations'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            explanation['top_contributing_features'] = [
                {
                    'feature': feature,
                    'deviation': deviation,
                    'value': explanation['feature_values'][feature],
                    'mean': float(X[feature].mean()),
                    'std': float(X[feature].std())
                }
                for feature, deviation in sorted_features[:5]  # Top 5 features
            ]
            
            explanations[idx] = explanation
        
        return explanations
    
    def visualize_results(self, X: pd.DataFrame, 
                         features_to_plot: List[str] = None,
                         figsize: Tuple[int, int] = (15, 12)) -> None:
        """
        Visualize anomaly detection results
        
        Args:
            X: Original data
            features_to_plot: Features to include in visualization
            figsize: Figure size
        """
        if self.results is None:
            raise ValueError("Must run prediction first")
        
        # Select features for plotting
        if features_to_plot is None:
            # Use features with highest importance
            sorted_features = sorted(self.results.feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)
            features_to_plot = [f[0] for f in sorted_features[:4]]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Isolation Forest Anomaly Detection Results', fontsize=16)
        
        # 1. Anomaly score distribution
        ax1 = axes[0, 0]
        normal_scores = self.results.anomaly_scores[self.results.predictions == 1]
        anomaly_scores = self.results.anomaly_scores[self.results.predictions == -1]
        
        ax1.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
        ax1.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
        ax1.axvline(x=self.results.threshold, color='black', linestyle='--', label='Threshold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Anomaly Score Distribution')
        ax1.legend()
        
        # 2. Feature importance
        ax2 = axes[0, 1]
        features = list(self.results.feature_importance.keys())[:10]  # Top 10
        importances = [self.results.feature_importance[f] for f in features]
        
        ax2.barh(features, importances)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Feature Importance')
        
        # 3. Anomaly rate over time (if timestamp available)
        ax3 = axes[0, 2]
        if 'timestamp' in X.columns:
            X_with_pred = X.copy()
            X_with_pred['anomaly'] = self.results.predictions == -1
            X_with_pred['timestamp'] = pd.to_datetime(X_with_pred['timestamp'])
            
            # Group by day and calculate anomaly rate
            daily_stats = X_with_pred.set_index('timestamp').resample('D')['anomaly'].agg(['sum', 'count'])
            daily_stats['anomaly_rate'] = daily_stats['sum'] / daily_stats['count'] * 100
            
            ax3.plot(daily_stats.index, daily_stats['anomaly_rate'])
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Anomaly Rate (%)')
            ax3.set_title('Anomaly Rate Over Time')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No timestamp data\\navailable', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Temporal Analysis')
        
        # 4. Scatter plot of two most important features
        ax4 = axes[1, 0]
        if len(features_to_plot) >= 2:
            feat1, feat2 = features_to_plot[0], features_to_plot[1]
            normal_mask = self.results.predictions == 1
            anomaly_mask = self.results.predictions == -1
            
            ax4.scatter(X[normal_mask][feat1], X[normal_mask][feat2], 
                       c='blue', alpha=0.6, label='Normal', s=20)
            ax4.scatter(X[anomaly_mask][feat1], X[anomaly_mask][feat2], 
                       c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
            
            ax4.set_xlabel(feat1)
            ax4.set_ylabel(feat2)
            ax4.set_title(f'{feat1} vs {feat2}')
            ax4.legend()
        
        # 5. Anomaly severity distribution
        ax5 = axes[1, 1]
        # Bin anomalies by severity (based on anomaly score)
        if len(anomaly_scores) > 0:
            severity_bins = np.percentile(anomaly_scores, [0, 33, 66, 100])
            severity_labels = ['High', 'Medium', 'Low']
            severity_counts = []
            
            for i in range(len(severity_bins)-1):
                count = np.sum((anomaly_scores >= severity_bins[i]) & 
                              (anomaly_scores < severity_bins[i+1]))
                severity_counts.append(count)
            
            ax5.pie(severity_counts, labels=severity_labels, autopct='%1.1f%%')
            ax5.set_title('Anomaly Severity Distribution')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        summary_text = f"""Detection Summary:
        
        Total Samples: {len(self.results.predictions):,}
        Anomalies Detected: {self.results.n_anomalies:,}
        Anomaly Rate: {self.results.anomaly_rate:.2f}%
        
        Score Statistics:
        Mean: {self.results.statistics['mean_anomaly_score']:.3f}
        Std: {self.results.statistics['std_anomaly_score']:.3f}
        Min: {self.results.statistics['min_anomaly_score']:.3f}
        Max: {self.results.statistics['max_anomaly_score']:.3f}
        
        Model Parameters:
        N Estimators: {self.n_estimators}
        Contamination: {self.contamination}
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                verticalalignment='center', fontsize=9)
        ax6.axis('off')
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.show()
    
    def export_anomalies(self, X: pd.DataFrame, 
                        output_path: str = None,
                        include_explanations: bool = True) -> pd.DataFrame:
        """
        Export detected anomalies
        
        Args:
            X: Original data
            output_path: Path to save anomalies CSV
            include_explanations: Whether to include anomaly explanations
            
        Returns:
            DataFrame with anomaly data
        """
        if self.results is None:
            raise ValueError("Must run prediction first")
        
        # Get anomalous samples
        anomaly_mask = self.results.predictions == -1
        anomalies_df = X[anomaly_mask].copy()
        
        # Add anomaly information
        anomalies_df['anomaly_score'] = self.results.anomaly_scores[anomaly_mask]
        anomalies_df['anomaly_rank'] = anomalies_df['anomaly_score'].rank()
        
        # Add severity classification
        score_percentiles = np.percentile(self.results.anomaly_scores[anomaly_mask], [33, 66])
        anomalies_df['severity'] = 'Low'
        anomalies_df.loc[anomalies_df['anomaly_score'] <= score_percentiles[0], 'severity'] = 'High'
        anomalies_df.loc[(anomalies_df['anomaly_score'] > score_percentiles[0]) & 
                        (anomalies_df['anomaly_score'] <= score_percentiles[1]), 'severity'] = 'Medium'
        
        # Add explanations if requested
        if include_explanations:
            explanations = self.get_anomaly_explanations(X, self.results.anomaly_indices)
            
            # Add top contributing feature info
            anomalies_df['top_anomalous_feature'] = ''
            anomalies_df['top_anomalous_feature_deviation'] = 0.0
            
            for idx, explanation in explanations.items():
                if len(explanation['top_contributing_features']) > 0:
                    top_feature = explanation['top_contributing_features'][0]
                    row_idx = anomalies_df.index[anomalies_df.index.get_loc(idx) if idx in anomalies_df.index else 0]
                    anomalies_df.loc[row_idx, 'top_anomalous_feature'] = top_feature['feature']
                    anomalies_df.loc[row_idx, 'top_anomalous_feature_deviation'] = top_feature['deviation']
        
        # Sort by anomaly score (most anomalous first)
        anomalies_df = anomalies_df.sort_values('anomaly_score')
        
        if output_path:
            anomalies_df.to_csv(output_path, index=False)
            logger.info(f"Anomalies exported to {output_path}")
        
        return anomalies_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_type': 'IsolationForest',
            'is_fitted': self.is_fitted,
            'n_estimators': self.n_estimators,
            'contamination': self.contamination,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None
        }
        
        if self.results:
            info.update({
                'n_anomalies_detected': self.results.n_anomalies,
                'anomaly_rate': self.results.anomaly_rate,
                'detection_threshold': self.results.threshold
            })
        
        return info
