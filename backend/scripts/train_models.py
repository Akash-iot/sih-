"""
Model Training and Evaluation Scripts for ETHEREYE
=================================================

This script provides utilities for training and evaluating all ML models:
- Data preparation and validation
- Model training with hyperparameter tuning
- Model evaluation and reporting
- Model persistence and versioning
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.clustering.dbscan_model import DBSCANClustering
from ml_models.anomaly_detection.isolation_forest_model import IsolationForestDetector
from ml_models.nlp_pii_extraction.nlp_processor import NLPPIIExtractor
from ml_models.risk_scoring.risk_model import RiskScorer
from ml_models.preprocessing.data_processor import DataPreprocessor
from ml_models.utils.model_utils import ModelUtils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model trainer for ETHEREYE ML pipeline
    """
    
    def __init__(self, data_path: str, output_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            data_path: Path to training data directory
            output_dir: Directory to save trained models
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.clustering_model = DBSCANClustering()
        self.anomaly_detector = IsolationForestDetector()
        self.nlp_processor = NLPPIIExtractor()
        self.risk_scorer = RiskScorer()
        self.data_preprocessor = DataPreprocessor()
        
        self.training_results = {}
    
    def load_training_data(self) -> dict:
        """
        Load training data from files
        
        Returns:
            Dictionary containing different datasets
        """
        logger.info("Loading training data...")
        
        data = {}
        
        # Load transaction data
        tx_file = self.data_path / "transactions.csv"
        if tx_file.exists():
            data['transactions'] = pd.read_csv(tx_file)
            logger.info(f"Loaded {len(data['transactions'])} transactions")
        
        # Load address data
        addr_file = self.data_path / "addresses.csv"
        if addr_file.exists():
            data['addresses'] = pd.read_csv(addr_file)
            logger.info(f"Loaded {len(data['addresses'])} addresses")
        
        # Load text data
        text_file = self.data_path / "text_data.json"
        if text_file.exists():
            with open(text_file, 'r') as f:
                data['texts'] = json.load(f)
            logger.info(f"Loaded {len(data['texts'])} text samples")
        
        # Load risk labels (if available)
        risk_file = self.data_path / "risk_labels.csv"
        if risk_file.exists():
            data['risk_labels'] = pd.read_csv(risk_file)
            logger.info(f"Loaded {len(data['risk_labels'])} risk labels")
        
        return data
    
    def train_clustering_model(self, data: pd.DataFrame) -> dict:
        """
        Train DBSCAN clustering model
        
        Args:
            data: Training data
            
        Returns:
            Training results
        """
        logger.info("Training DBSCAN clustering model...")
        
        start_time = datetime.now()
        
        # Preprocess data
        processed_data = self.data_preprocessor.preprocess_address_data(data.copy())
        
        # Optimize parameters
        optimization_result = self.clustering_model.optimize_parameters(processed_data)
        best_params = optimization_result['best_params']
        
        # Update model with best parameters
        self.clustering_model.eps = best_params['eps']
        self.clustering_model.min_samples = best_params['min_samples']
        
        # Train model
        results = self.clustering_model.fit(processed_data)
        
        # Save model
        model_dir = ModelUtils.create_model_directory(str(self.output_dir), "clustering")
        model_path = ModelUtils.save_model_artifacts(self.clustering_model, model_dir, "dbscan_model")
        
        # Save configuration
        config = {
            'model_type': 'DBSCAN',
            'parameters': best_params,
            'training_data_shape': processed_data.shape,
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        ModelUtils.save_model_config(config, model_dir)
        
        training_result = {
            'model_path': model_path,
            'model_dir': model_dir,
            'n_clusters': results.n_clusters,
            'n_noise': results.n_noise,
            'silhouette_score': results.silhouette_score,
            'best_parameters': best_params,
            'training_time': config['training_time']
        }
        
        logger.info(f"Clustering model trained successfully. Found {results.n_clusters} clusters.")
        return training_result
    
    def train_anomaly_detector(self, data: pd.DataFrame) -> dict:
        """
        Train Isolation Forest anomaly detector
        
        Args:
            data: Training data
            
        Returns:
            Training results
        """
        logger.info("Training Isolation Forest anomaly detector...")
        
        start_time = datetime.now()
        
        # Preprocess data
        processed_data = self.data_preprocessor.preprocess_transaction_data(data.copy())
        
        # Optimize contamination parameter
        optimization_result = self.anomaly_detector.optimize_contamination(processed_data)
        best_contamination = optimization_result['best_contamination']
        
        # Update model with best parameters
        self.anomaly_detector.contamination = best_contamination
        
        # Train model
        results = self.anomaly_detector.fit_predict(processed_data)
        
        # Perform cross-validation
        cv_results = self.anomaly_detector.cross_validate(processed_data)
        
        # Save model
        model_dir = ModelUtils.create_model_directory(str(self.output_dir), "anomaly_detection")
        model_path = ModelUtils.save_model_artifacts(self.anomaly_detector, model_dir, "isolation_forest_model")
        
        # Save configuration
        config = {
            'model_type': 'IsolationForest',
            'parameters': {
                'contamination': best_contamination,
                'n_estimators': self.anomaly_detector.n_estimators
            },
            'training_data_shape': processed_data.shape,
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        ModelUtils.save_model_config(config, model_dir)
        
        training_result = {
            'model_path': model_path,
            'model_dir': model_dir,
            'n_anomalies': results.n_anomalies,
            'anomaly_rate': results.anomaly_rate,
            'best_contamination': best_contamination,
            'cv_scores': cv_results,
            'training_time': config['training_time']
        }
        
        logger.info(f"Anomaly detector trained successfully. Detected {results.n_anomalies} anomalies.")
        return training_result
    
    def train_nlp_processor(self, texts: list) -> dict:
        """
        Train/Initialize NLP processor
        
        Args:
            texts: List of text samples
            
        Returns:
            Training results
        """
        logger.info("Initializing NLP processor...")
        
        start_time = datetime.now()
        
        # Test NLP processor with sample texts
        sample_texts = texts[:100] if len(texts) > 100 else texts
        results = self.nlp_processor.batch_analyze_texts(sample_texts)
        
        # Generate insights
        insights = self.nlp_processor.extract_social_media_insights(sample_texts)
        
        # Save model info
        model_dir = ModelUtils.create_model_directory(str(self.output_dir), "nlp_processing")
        
        config = {
            'model_type': 'NLP_PII_Extractor',
            'models_loaded': self.nlp_processor.get_model_info(),
            'sample_size': len(sample_texts),
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        ModelUtils.save_model_config(config, model_dir)
        
        training_result = {
            'model_dir': model_dir,
            'sample_results': len([r for r in results if r is not None]),
            'insights': insights,
            'model_info': config['models_loaded'],
            'training_time': config['training_time']
        }
        
        logger.info("NLP processor initialized successfully.")
        return training_result
    
    def train_risk_scorer(self, wallet_data: dict, risk_labels: pd.DataFrame = None) -> dict:
        """
        Train risk scoring model
        
        Args:
            wallet_data: Dictionary of wallet transaction data
            risk_labels: Optional risk labels for supervised training
            
        Returns:
            Training results
        """
        logger.info("Training risk scoring model...")
        
        start_time = datetime.now()
        
        # Initialize risk scorer
        self.risk_scorer.initialize_models()
        
        # Prepare sample wallet data
        sample_wallets = {}
        for i, (addr, txs) in enumerate(list(wallet_data.items())[:50]):  # Sample 50 wallets
            sample_wallets[addr] = pd.DataFrame(txs)
        
        # Perform batch assessment
        assessments = self.risk_scorer.batch_assess_wallets(sample_wallets)
        
        # Generate statistics
        stats = self.risk_scorer.get_risk_statistics(assessments)
        
        # Save model
        model_dir = ModelUtils.create_model_directory(str(self.output_dir), "risk_scoring")
        self.risk_scorer.save_model(str(Path(model_dir) / "risk_scorer.joblib"))
        
        # Save configuration
        config = {
            'model_type': 'RiskScorer',
            'risk_thresholds': self.risk_scorer.risk_thresholds,
            'feature_weights': self.risk_scorer.feature_weights,
            'sample_size': len(sample_wallets),
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        ModelUtils.save_model_config(config, model_dir)
        
        training_result = {
            'model_dir': model_dir,
            'assessments_count': len(assessments),
            'risk_statistics': stats,
            'model_config': config,
            'training_time': config['training_time']
        }
        
        logger.info(f"Risk scorer trained successfully. Assessed {len(assessments)} wallets.")
        return training_result
    
    def evaluate_models(self, test_data: dict) -> dict:
        """
        Evaluate all trained models
        
        Args:
            test_data: Test datasets
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating trained models...")
        
        evaluation_results = {}
        
        # Evaluate clustering if test data available
        if 'addresses' in test_data and hasattr(self.clustering_model, 'results'):
            processed_test = self.data_preprocessor.preprocess_address_data(test_data['addresses'].copy())
            test_predictions = self.clustering_model.predict(processed_test)
            
            evaluation_results['clustering'] = {
                'test_samples': len(processed_test),
                'clusters_found': len(set(test_predictions)) - (1 if -1 in test_predictions else 0),
                'noise_points': np.sum(test_predictions == -1)
            }
        
        # Evaluate anomaly detection
        if 'transactions' in test_data and self.anomaly_detector.is_fitted:
            processed_test = self.data_preprocessor.preprocess_transaction_data(test_data['transactions'].copy())
            test_results = self.anomaly_detector.predict(processed_test, return_scores=True)
            
            evaluation_results['anomaly_detection'] = {
                'test_samples': len(processed_test),
                'anomalies_detected': test_results.n_anomalies,
                'anomaly_rate': test_results.anomaly_rate
            }
        
        # Evaluate NLP processor
        if 'texts' in test_data:
            test_texts = test_data['texts'][:50]  # Sample for performance
            nlp_results = self.nlp_processor.batch_analyze_texts(test_texts)
            
            evaluation_results['nlp_processing'] = {
                'test_samples': len(test_texts),
                'processed_successfully': len([r for r in nlp_results if r is not None]),
                'pii_found': sum(len(r.pii_data.pii_entities) for r in nlp_results if r)
            }
        
        return evaluation_results
    
    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report
        
        Returns:
            Path to generated report
        """
        logger.info("Generating training report...")
        
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'training_results': self.training_results,
            'model_summary': {
                'clustering': {
                    'status': 'trained' if 'clustering' in self.training_results else 'not_trained',
                    'n_clusters': self.training_results.get('clustering', {}).get('n_clusters', 0)
                },
                'anomaly_detection': {
                    'status': 'trained' if 'anomaly_detection' in self.training_results else 'not_trained',
                    'anomaly_rate': self.training_results.get('anomaly_detection', {}).get('anomaly_rate', 0)
                },
                'nlp_processing': {
                    'status': 'initialized' if 'nlp_processing' in self.training_results else 'not_initialized'
                },
                'risk_scoring': {
                    'status': 'trained' if 'risk_scoring' in self.training_results else 'not_trained',
                    'assessments': self.training_results.get('risk_scoring', {}).get('assessments_count', 0)
                }
            }
        }
        
        report_path = self.output_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        md_path = self.output_dir / 'training_summary.md'
        self._generate_markdown_summary(report, md_path)
        
        logger.info(f"Training report saved to {report_path}")
        return str(report_path)
    
    def _generate_markdown_summary(self, report: dict, output_path: Path):
        """Generate markdown summary of training results"""
        md_content = f"""# ETHEREYE ML Models Training Report

Training completed: {report['training_timestamp']}

## Model Summary

"""
        
        for model, info in report['model_summary'].items():
            status = info['status']
            md_content += f"### {model.title().replace('_', ' ')}\n"
            md_content += f"- Status: {status}\n"
            
            if model == 'clustering' and 'n_clusters' in info:
                md_content += f"- Clusters found: {info['n_clusters']}\\n"
            elif model == 'anomaly_detection' and 'anomaly_rate' in info:
                md_content += f"- Anomaly rate: {info['anomaly_rate']:.2f}%\\n"
            elif model == 'risk_scoring' and 'assessments' in info:
                md_content += f"- Assessments performed: {info['assessments']}\\n"
            
            md_content += "\\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def run_full_training(self) -> dict:
        """
        Run full training pipeline
        
        Returns:
            Complete training results
        """
        logger.info("Starting full training pipeline...")
        
        # Load data
        data = self.load_training_data()
        
        if not data:
            raise ValueError("No training data found. Please provide data files.")
        
        # Train models
        if 'addresses' in data:
            self.training_results['clustering'] = self.train_clustering_model(data['addresses'])
        
        if 'transactions' in data:
            self.training_results['anomaly_detection'] = self.train_anomaly_detector(data['transactions'])
        
        if 'texts' in data:
            self.training_results['nlp_processing'] = self.train_nlp_processor(data['texts'])
        
        if 'transactions' in data:
            # Prepare wallet data for risk scoring
            wallet_data = {}
            if 'addresses' in data:
                for _, row in data['addresses'].head(20).iterrows():  # Sample wallets
                    addr = row.get('address', f'wallet_{len(wallet_data)}')
                    # Create sample transactions for this wallet
                    wallet_txs = data['transactions'].head(50).to_dict('records')
                    wallet_data[addr] = wallet_txs
            
            if wallet_data:
                self.training_results['risk_scoring'] = self.train_risk_scorer(
                    wallet_data, 
                    data.get('risk_labels')
                )
        
        # Generate report
        report_path = self.generate_training_report()
        
        logger.info("Full training pipeline completed successfully!")
        return {
            'training_results': self.training_results,
            'report_path': report_path
        }

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train ETHEREYE ML models')
    parser.add_argument('--data-path', required=True, help='Path to training data directory')
    parser.add_argument('--output-dir', default='trained_models', help='Output directory for trained models')
    parser.add_argument('--models', nargs='+', choices=['clustering', 'anomaly', 'nlp', 'risk', 'all'], 
                       default=['all'], help='Models to train')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    ModelUtils.setup_logging(args.log_level)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.data_path, args.output_dir)
        
        if 'all' in args.models:
            # Run full training pipeline
            results = trainer.run_full_training()
        else:
            # Train specific models
            data = trainer.load_training_data()
            
            if 'clustering' in args.models and 'addresses' in data:
                trainer.training_results['clustering'] = trainer.train_clustering_model(data['addresses'])
            
            if 'anomaly' in args.models and 'transactions' in data:
                trainer.training_results['anomaly_detection'] = trainer.train_anomaly_detector(data['transactions'])
            
            if 'nlp' in args.models and 'texts' in data:
                trainer.training_results['nlp_processing'] = trainer.train_nlp_processor(data['texts'])
            
            if 'risk' in args.models and 'transactions' in data:
                wallet_data = {'sample_wallet': data['transactions'].head(100).to_dict('records')}
                trainer.training_results['risk_scoring'] = trainer.train_risk_scorer(wallet_data)
            
            # Generate report
            trainer.generate_training_report()
        
        print("Training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()