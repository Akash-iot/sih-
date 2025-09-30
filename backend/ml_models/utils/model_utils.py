"""
Model Utilities for ETHEREYE ML Pipeline
=======================================

This module provides utility functions and classes for:
- Model persistence and loading
- Performance evaluation
- Data validation and preprocessing helpers
- Configuration management
- Logging and monitoring utilities
"""

import os
import json
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, roc_auc_score
)
import warnings

logger = logging.getLogger(__name__)

class ModelUtils:
    """
    Utility class for common ML model operations
    """
    
    @staticmethod
    def setup_logging(log_level: str = 'INFO', 
                     log_file: Optional[str] = None) -> None:
        \"\"\"
        Setup logging configuration for ML pipeline
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
        \"\"\"
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
        # Suppress warnings from transformers and other libraries
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    @staticmethod
    def create_model_directory(base_path: str, model_name: str) -> str:
        \"\"\"
        Create directory structure for model artifacts
        
        Args:
            base_path: Base directory path
            model_name: Name of the model
            
        Returns:
            Path to created model directory
        \"\"\"
        model_dir = Path(base_path) / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (model_dir / 'models').mkdir(exist_ok=True)
        (model_dir / 'metrics').mkdir(exist_ok=True)
        (model_dir / 'plots').mkdir(exist_ok=True)
        (model_dir / 'data').mkdir(exist_ok=True)
        
        logger.info(f\"Created model directory: {model_dir}\")
        return str(model_dir)
    
    @staticmethod
    def save_model_config(config: Dict[str, Any], 
                         model_dir: str, 
                         filename: str = 'config.json') -> None:
        \"\"\"
        Save model configuration to JSON file
        
        Args:
            config: Configuration dictionary
            model_dir: Model directory path
            filename: Config filename
        \"\"\"
        config_path = Path(model_dir) / filename
        
        # Convert non-serializable objects to strings
        serializable_config = ModelUtils._make_serializable(config)
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2, default=str)
        
        logger.info(f\"Model configuration saved to {config_path}\")
    
    @staticmethod
    def load_model_config(model_dir: str, 
                         filename: str = 'config.json') -> Dict[str, Any]:
        \"\"\"
        Load model configuration from JSON file
        
        Args:
            model_dir: Model directory path
            filename: Config filename
            
        Returns:
            Configuration dictionary
        \"\"\"
        config_path = Path(model_dir) / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f\"Configuration file not found: {config_path}\")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f\"Model configuration loaded from {config_path}\")
        return config
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        \"\"\"Convert non-serializable objects to serializable format\"\"\"
        if isinstance(obj, dict):
            return {k: ModelUtils._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelUtils._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    @staticmethod
    def save_model_artifacts(model: Any, 
                           model_dir: str,
                           model_name: str = 'model',
                           use_joblib: bool = True) -> str:
        \"\"\"
        Save model artifacts
        
        Args:
            model: Model object to save
            model_dir: Model directory path
            model_name: Name for model file
            use_joblib: Whether to use joblib (recommended for sklearn models)
            
        Returns:
            Path to saved model file
        \"\"\"
        models_dir = Path(model_dir) / 'models'
        
        if use_joblib:
            model_path = models_dir / f\"{model_name}.joblib\"
            joblib.dump(model, model_path)
        else:
            model_path = models_dir / f\"{model_name}.pkl\"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f\"Model saved to {model_path}\")
        return str(model_path)
    
    @staticmethod
    def load_model_artifacts(model_path: str) -> Any:
        \"\"\"
        Load model artifacts
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded model object
        \"\"\"
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f\"Model file not found: {model_path}\")
        
        if model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        elif model_path.suffix == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f\"Unsupported model file format: {model_path.suffix}\")
        
        logger.info(f\"Model loaded from {model_path}\")
        return model
    
    @staticmethod
    def evaluate_classification_model(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray] = None,
                                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        \"\"\"
        Comprehensive evaluation of classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: Class names (optional)
            
        Returns:
            Dictionary with evaluation metrics
        \"\"\"
        metrics = {}
        
        # Basic classification metrics
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC if probabilities provided
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                logger.warning(f\"Could not calculate ROC AUC: {e}\")
                metrics['roc_auc'] = None
        
        # Additional metrics
        metrics['accuracy'] = (y_true == y_pred).mean()
        metrics['total_samples'] = len(y_true)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = 'Confusion Matrix',
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> None:
        \"\"\"
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        \"\"\"
        plt.figure(figsize=figsize)
        
        if isinstance(cm, list):
            cm = np.array(cm)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f\"Confusion matrix plot saved to {save_path}\")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray,
                      y_prob: np.ndarray, 
                      title: str = 'ROC Curve',
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> None:
        \"\"\"
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        \"\"\"
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc=\"lower right\")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f\"ROC curve plot saved to {save_path}\")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   title: str = 'Precision-Recall Curve',
                                   figsize: Tuple[int, int] = (8, 6),
                                   save_path: Optional[str] = None) -> None:
        \"\"\"
        Plot precision-recall curve
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        \"\"\"
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f\"Precision-recall curve plot saved to {save_path}\")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def validate_data_schema(data: pd.DataFrame, 
                           required_columns: List[str],
                           optional_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        \"\"\"
        Validate data schema and quality
        
        Args:
            data: Input DataFrame
            required_columns: List of required columns
            optional_columns: List of optional columns
            
        Returns:
            Dictionary with validation results
        \"\"\"
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check required columns
        missing_required = set(required_columns) - set(data.columns)
        if missing_required:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f\"Missing required columns: {list(missing_required)}\")
        
        # Check data types and missing values
        for col in data.columns:
            col_info = {
                'dtype': str(data[col].dtype),
                'missing_count': data[col].isnull().sum(),
                'missing_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_values': data[col].nunique()
            }
            validation_result['summary'][col] = col_info
            
            # Warning for high missing values
            if col_info['missing_percentage'] > 50:
                validation_result['warnings'].append(f\"Column {col} has {col_info['missing_percentage']:.1f}% missing values\")
        
        # Check data size
        validation_result['summary']['total_rows'] = len(data)
        validation_result['summary']['total_columns'] = len(data.columns)
        
        if len(data) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append(\"Dataset is empty\")
        
        return validation_result
    
    @staticmethod
    def detect_data_drift(reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         numerical_threshold: float = 0.1,
                         categorical_threshold: float = 0.1) -> Dict[str, Any]:
        \"\"\"
        Detect data drift between reference and current datasets
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset to compare
            numerical_threshold: Threshold for numerical drift detection
            categorical_threshold: Threshold for categorical drift detection
            
        Returns:
            Dictionary with drift detection results
        \"\"\"
        from scipy import stats
        
        drift_result = {
            'has_drift': False,
            'drifted_columns': [],
            'column_analysis': {}
        }
        
        common_columns = set(reference_data.columns) & set(current_data.columns)
        
        for col in common_columns:
            ref_col = reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            analysis = {'column': col, 'drift_detected': False}
            
            if pd.api.types.is_numeric_dtype(ref_col):
                # Kolmogorov-Smirnov test for numerical columns
                statistic, p_value = stats.ks_2samp(ref_col, curr_col)
                analysis.update({
                    'test': 'ks_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < numerical_threshold
                })
            else:
                # Chi-square test for categorical columns
                ref_counts = ref_col.value_counts()
                curr_counts = curr_col.value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                    statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    analysis.update({
                        'test': 'chi_square',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift_detected': p_value < categorical_threshold
                    })
            
            drift_result['column_analysis'][col] = analysis
            
            if analysis['drift_detected']:
                drift_result['has_drift'] = True
                drift_result['drifted_columns'].append(col)
        
        return drift_result
    
    @staticmethod
    def generate_data_profile(data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Generate comprehensive data profile
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with data profile
        \"\"\"
        profile = {
            'overview': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'missing_cells': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
                'duplicated_rows': data.duplicated().sum()
            },
            'columns': {}
        }
        
        for col in data.columns:
            col_profile = {
                'dtype': str(data[col].dtype),
                'missing_count': data[col].isnull().sum(),
                'missing_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_count': data[col].nunique(),
                'unique_percentage': (data[col].nunique() / len(data)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(data[col]):
                col_profile.update({
                    'mean': float(data[col].mean()) if not data[col].empty else None,
                    'std': float(data[col].std()) if not data[col].empty else None,
                    'min': float(data[col].min()) if not data[col].empty else None,
                    'max': float(data[col].max()) if not data[col].empty else None,
                    'median': float(data[col].median()) if not data[col].empty else None,
                    'q25': float(data[col].quantile(0.25)) if not data[col].empty else None,
                    'q75': float(data[col].quantile(0.75)) if not data[col].empty else None
                })
            else:
                top_values = data[col].value_counts().head(5)
                col_profile['top_values'] = top_values.to_dict()
            
            profile['columns'][col] = col_profile
        
        return profile
    
    @staticmethod
    def create_model_report(model_info: Dict[str, Any],
                          metrics: Dict[str, Any],
                          data_profile: Dict[str, Any],
                          model_dir: str) -> str:
        \"\"\"
        Create comprehensive model report
        
        Args:
            model_info: Model information
            metrics: Model evaluation metrics
            data_profile: Data profile information
            model_dir: Model directory path
            
        Returns:
            Path to generated report file
        \"\"\"
        report = {
            'generated_at': datetime.now().isoformat(),
            'model_info': model_info,
            'performance_metrics': metrics,
            'data_profile': data_profile
        }
        
        report_path = Path(model_dir) / 'model_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also create a markdown summary
        md_path = Path(model_dir) / 'model_report.md'
        ModelUtils._create_markdown_report(report, md_path)
        
        logger.info(f\"Model report saved to {report_path}\")
        return str(report_path)
    
    @staticmethod
    def _create_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
        \"\"\"Create markdown version of model report\"\"\"
        md_content = f\"\"\"# Model Report
        
Generated: {report['generated_at']}

## Model Information
- Model Type: {report['model_info'].get('model_type', 'Unknown')}
- Is Trained: {report['model_info'].get('is_trained', False)}

## Performance Metrics
\"\"\"
        
        # Add classification metrics if available
        if 'classification_report' in report['performance_metrics']:
            md_content += \"\\n### Classification Report\\n\"
            cr = report['performance_metrics']['classification_report']
            if 'accuracy' in cr:
                md_content += f\"- Accuracy: {cr['accuracy']:.3f}\\n\"
            if 'macro avg' in cr:
                md_content += f\"- Macro F1-Score: {cr['macro avg']['f1-score']:.3f}\\n\"
        
        # Add data profile summary
        if 'data_profile' in report:
            dp = report['data_profile']
            md_content += f\"\"\"
## Data Profile Summary
- Dataset Shape: {dp['overview']['shape']}
- Missing Data: {dp['overview']['missing_percentage']:.2f}%
- Duplicated Rows: {dp['overview']['duplicated_rows']}
- Total Columns: {len(dp['columns'])}
\"\"\"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    @staticmethod
    def monitor_model_performance(predictions: np.ndarray,
                                 actuals: Optional[np.ndarray] = None,
                                 timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        \"\"\"
        Monitor model performance over time
        
        Args:
            predictions: Model predictions
            actuals: Actual values (optional)
            timestamp: Timestamp for monitoring
            
        Returns:
            Dictionary with monitoring metrics
        \"\"\"
        monitoring_data = {
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'prediction_count': len(predictions),
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
        if actuals is not None:
            monitoring_data['performance'] = {
                'mae': float(np.mean(np.abs(predictions - actuals))),
                'mse': float(np.mean((predictions - actuals) ** 2)),
                'correlation': float(np.corrcoef(predictions, actuals)[0, 1]) if len(predictions) > 1 else 0.0
            }
        
        return monitoring_data