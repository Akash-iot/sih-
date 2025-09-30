"""
Advanced Clustering Service for ETHEREYE
Implements DBSCAN clustering and IsolationForest anomaly detection
for blockchain address analysis and pattern recognition
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)

class BlockchainClusteringService:
    """
    Advanced clustering service for blockchain address analysis
    Uses DBSCAN for density-based clustering and IsolationForest for anomaly detection
    """
    
    def __init__(self):
        self.dbscan_model = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_columns = [
            'balance_eth', 'transaction_count', 'avg_transaction_value',
            'transaction_frequency', 'unique_counterparties', 'gas_usage_avg',
            'contract_interactions', 'time_active_days', 'incoming_ratio',
            'outgoing_ratio', 'round_number_ratio', 'night_activity_ratio'
        ]
        
    def extract_address_features(self, addresses_data: List[Dict]) -> pd.DataFrame:
        """
        Extract comprehensive features from blockchain addresses for clustering
        
        Args:
            addresses_data: List of address data dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for addr_data in addresses_data:
            try:
                # Basic features
                balance = float(addr_data.get('balance_eth', 0))
                tx_count = int(addr_data.get('transaction_count', 0))
                
                # Calculate derived features
                transactions = addr_data.get('transactions', [])
                
                if transactions:
                    tx_values = [float(tx.get('value_eth', 0)) for tx in transactions]
                    tx_timestamps = [tx.get('timestamp') for tx in transactions if tx.get('timestamp')]
                    
                    avg_tx_value = np.mean(tx_values) if tx_values else 0
                    
                    # Calculate transaction frequency (txs per day)
                    if len(tx_timestamps) > 1:
                        first_tx = min(tx_timestamps)
                        last_tx = max(tx_timestamps)
                        days_active = (pd.to_datetime(last_tx) - pd.to_datetime(first_tx)).days
                        tx_frequency = tx_count / max(days_active, 1)
                    else:
                        tx_frequency = 0
                    
                    # Unique counterparties
                    counterparties = set()
                    for tx in transactions:
                        if tx.get('from_address'):
                            counterparties.add(tx['from_address'])
                        if tx.get('to_address'):
                            counterparties.add(tx['to_address'])
                    unique_counterparties = len(counterparties)
                    
                    # Gas usage patterns
                    gas_values = [int(tx.get('gas_used', 0)) for tx in transactions]
                    avg_gas = np.mean(gas_values) if gas_values else 0
                    
                    # Contract interactions
                    contract_txs = sum(1 for tx in transactions if tx.get('is_contract_interaction', False))
                    contract_ratio = contract_txs / len(transactions) if transactions else 0
                    
                    # Time activity analysis
                    time_active = days_active if len(tx_timestamps) > 1 else 1
                    
                    # Direction ratios
                    incoming_txs = sum(1 for tx in transactions if tx.get('direction') == 'incoming')
                    outgoing_txs = sum(1 for tx in transactions if tx.get('direction') == 'outgoing')
                    total_directional = incoming_txs + outgoing_txs
                    
                    incoming_ratio = incoming_txs / total_directional if total_directional > 0 else 0
                    outgoing_ratio = outgoing_txs / total_directional if total_directional > 0 else 0
                    
                    # Suspicious patterns
                    round_numbers = sum(1 for val in tx_values if self._is_round_number(val))
                    round_ratio = round_numbers / len(tx_values) if tx_values else 0
                    
                    # Night activity (assuming UTC)
                    night_txs = 0
                    for ts in tx_timestamps:
                        if ts:
                            hour = pd.to_datetime(ts).hour
                            if hour >= 22 or hour <= 6:  # Night hours
                                night_txs += 1
                    night_ratio = night_txs / len(tx_timestamps) if tx_timestamps else 0
                    
                else:
                    # Default values for addresses without transactions
                    avg_tx_value = 0
                    tx_frequency = 0
                    unique_counterparties = 0
                    avg_gas = 0
                    contract_ratio = 0
                    time_active = 1
                    incoming_ratio = 0
                    outgoing_ratio = 0
                    round_ratio = 0
                    night_ratio = 0
                
                feature_vector = {
                    'address': addr_data.get('address', ''),
                    'balance_eth': balance,
                    'transaction_count': tx_count,
                    'avg_transaction_value': avg_tx_value,
                    'transaction_frequency': tx_frequency,
                    'unique_counterparties': unique_counterparties,
                    'gas_usage_avg': avg_gas,
                    'contract_interactions': contract_ratio,
                    'time_active_days': time_active,
                    'incoming_ratio': incoming_ratio,
                    'outgoing_ratio': outgoing_ratio,
                    'round_number_ratio': round_ratio,
                    'night_activity_ratio': night_ratio
                }
                
                features.append(feature_vector)
                
            except Exception as e:
                logger.error(f"Error extracting features for address {addr_data.get('address', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(features)
    
    def _is_round_number(self, value: float, tolerance: float = 1e-6) -> bool:
        """Check if a number is likely a round number"""
        if value == 0:
            return True
        
        # Check for common round patterns
        round_patterns = [1, 5, 10, 25, 50, 100, 500, 1000]
        for pattern in round_patterns:
            if abs(value - pattern) < tolerance:
                return True
            if value > pattern and abs(value % pattern) < tolerance:
                return True
        
        return False
    
    def perform_dbscan_clustering(
        self, 
        addresses_data: List[Dict], 
        eps: float = 0.5, 
        min_samples: int = 5,
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering on blockchain addresses
        
        Args:
            addresses_data: List of address data dictionaries
            eps: Maximum distance between samples in the same neighborhood
            min_samples: Minimum samples in a neighborhood for core point
            optimize_params: Whether to optimize eps and min_samples
            
        Returns:
            Dictionary containing cluster results and metrics
        """
        try:
            # Extract features
            features_df = self.extract_address_features(addresses_data)
            
            if len(features_df) < 2:
                return {
                    "error": "Insufficient data for clustering",
                    "addresses_analyzed": len(features_df)
                }
            
            # Prepare feature matrix
            X = features_df[self.feature_columns].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply PCA if dataset is large
            if X_scaled.shape[1] > 6:
                X_scaled = self.pca.fit_transform(X_scaled)
            
            # Optimize parameters if requested
            if optimize_params and len(features_df) > 10:
                eps, min_samples = self._optimize_dbscan_params(X_scaled)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            self.dbscan_model = dbscan
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            silhouette_avg = 0
            if n_clusters > 1 and n_noise < len(cluster_labels) - 1:
                try:
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                except:
                    silhouette_avg = 0
            
            # Prepare results
            results = []
            cluster_stats = {}
            
            for idx, (_, row) in enumerate(features_df.iterrows()):
                cluster_id = int(cluster_labels[idx])
                
                result = {
                    "address": row['address'],
                    "cluster_id": cluster_id,
                    "is_noise": cluster_id == -1,
                    "features": {col: float(row[col]) for col in self.feature_columns}
                }
                
                results.append(result)
                
                # Collect cluster statistics
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {
                        "size": 0,
                        "addresses": [],
                        "avg_balance": 0,
                        "avg_tx_count": 0,
                        "risk_indicators": []
                    }
                
                cluster_stats[cluster_id]["size"] += 1
                cluster_stats[cluster_id]["addresses"].append(row['address'])
                cluster_stats[cluster_id]["avg_balance"] += row['balance_eth']
                cluster_stats[cluster_id]["avg_tx_count"] += row['transaction_count']
                
                # Identify risk indicators
                if row['round_number_ratio'] > 0.5:
                    cluster_stats[cluster_id]["risk_indicators"].append("high_round_numbers")
                if row['night_activity_ratio'] > 0.7:
                    cluster_stats[cluster_id]["risk_indicators"].append("unusual_timing")
                if row['transaction_frequency'] > 10:
                    cluster_stats[cluster_id]["risk_indicators"].append("high_frequency")
            
            # Finalize cluster statistics
            for cluster_id, stats in cluster_stats.items():
                if stats["size"] > 0:
                    stats["avg_balance"] /= stats["size"]
                    stats["avg_tx_count"] /= stats["size"]
                    stats["risk_indicators"] = list(set(stats["risk_indicators"]))
            
            return {
                "clustering_results": results,
                "cluster_statistics": cluster_stats,
                "metrics": {
                    "n_clusters": n_clusters,
                    "n_noise_points": n_noise,
                    "silhouette_score": float(silhouette_avg),
                    "eps_used": eps,
                    "min_samples_used": min_samples
                },
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "total_addresses_analyzed": len(features_df)
            }
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return {
                "error": f"Clustering failed: {str(e)}",
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def detect_anomalies(
        self, 
        addresses_data: List[Dict], 
        contamination: float = 0.1,
        n_estimators: int = 100
    ) -> Dict[str, Any]:
        """
        Detect anomalous addresses using Isolation Forest
        
        Args:
            addresses_data: List of address data dictionaries
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            # Extract features
            features_df = self.extract_address_features(addresses_data)
            
            if len(features_df) < 2:
                return {
                    "error": "Insufficient data for anomaly detection",
                    "addresses_analyzed": len(features_df)
                }
            
            # Prepare feature matrix
            X = features_df[self.feature_columns].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = StandardScaler().fit_transform(X)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.decision_function(X_scaled)
            
            self.isolation_forest = iso_forest
            
            # Prepare results
            results = []
            anomaly_stats = {
                "normal": {"count": 0, "addresses": []},
                "anomaly": {"count": 0, "addresses": [], "risk_factors": []}
            }
            
            for idx, (_, row) in enumerate(features_df.iterrows()):
                is_anomaly = anomaly_labels[idx] == -1
                anomaly_score = float(anomaly_scores[idx])
                
                # Calculate risk factors for anomalies
                risk_factors = []
                if is_anomaly:
                    if row['round_number_ratio'] > 0.5:
                        risk_factors.append("excessive_round_numbers")
                    if row['night_activity_ratio'] > 0.8:
                        risk_factors.append("suspicious_timing")
                    if row['transaction_frequency'] > 50:
                        risk_factors.append("extremely_high_frequency")
                    if row['avg_transaction_value'] > 100:
                        risk_factors.append("high_value_transactions")
                    if row['unique_counterparties'] > 100:
                        risk_factors.append("excessive_connections")
                    if row['contract_interactions'] > 0.8:
                        risk_factors.append("heavy_contract_usage")
                
                result = {
                    "address": row['address'],
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "confidence": abs(anomaly_score),
                    "risk_factors": risk_factors,
                    "features": {col: float(row[col]) for col in self.feature_columns}
                }
                
                results.append(result)
                
                # Update statistics
                category = "anomaly" if is_anomaly else "normal"
                anomaly_stats[category]["count"] += 1
                anomaly_stats[category]["addresses"].append(row['address'])
                
                if is_anomaly:
                    anomaly_stats[category]["risk_factors"].extend(risk_factors)
            
            # Remove duplicates from risk factors
            anomaly_stats["anomaly"]["risk_factors"] = list(set(anomaly_stats["anomaly"]["risk_factors"]))
            
            return {
                "anomaly_results": results,
                "statistics": anomaly_stats,
                "metrics": {
                    "total_addresses": len(features_df),
                    "normal_addresses": anomaly_stats["normal"]["count"],
                    "anomalous_addresses": anomaly_stats["anomaly"]["count"],
                    "contamination_used": contamination,
                    "n_estimators_used": n_estimators
                },
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "error": f"Anomaly detection failed: {str(e)}",
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def _optimize_dbscan_params(self, X: np.ndarray) -> Tuple[float, int]:
        """
        Optimize DBSCAN parameters using silhouette analysis
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Tuple of optimized (eps, min_samples)
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Estimate eps using k-distance graph
            n_neighbors = min(max(3, int(len(X) * 0.1)), 10)
            neigh = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs = neigh.fit(X)
            distances, indices = nbrs.kneighbors(X)
            
            # Sort distances to find the elbow
            k_distances = np.sort(distances[:, n_neighbors-1], axis=0)
            
            # Use gradient to find elbow point
            gradients = np.gradient(k_distances)
            elbow_idx = np.argmax(gradients)
            eps = k_distances[elbow_idx]
            
            # Optimize min_samples
            min_samples = max(2, min(n_neighbors, int(len(X) * 0.05)))
            
            return float(eps), int(min_samples)
            
        except Exception as e:
            logger.warning(f"Parameter optimization failed: {e}, using defaults")
            return 0.5, 5
    
    def combine_clustering_and_anomaly_detection(
        self, 
        addresses_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Combine DBSCAN clustering with anomaly detection for comprehensive analysis
        
        Args:
            addresses_data: List of address data dictionaries
            
        Returns:
            Combined analysis results
        """
        try:
            # Perform clustering
            clustering_results = self.perform_dbscan_clustering(addresses_data)
            
            # Perform anomaly detection
            anomaly_results = self.detect_anomalies(addresses_data)
            
            if "error" in clustering_results or "error" in anomaly_results:
                return {
                    "error": "One or both analyses failed",
                    "clustering_error": clustering_results.get("error"),
                    "anomaly_error": anomaly_results.get("error")
                }
            
            # Combine results
            combined_results = []
            
            for cluster_result in clustering_results["clustering_results"]:
                address = cluster_result["address"]
                
                # Find corresponding anomaly result
                anomaly_result = next(
                    (ar for ar in anomaly_results["anomaly_results"] if ar["address"] == address),
                    None
                )
                
                if anomaly_result:
                    combined_result = {
                        "address": address,
                        "cluster_id": cluster_result["cluster_id"],
                        "is_noise": cluster_result["is_noise"],
                        "is_anomaly": anomaly_result["is_anomaly"],
                        "anomaly_score": anomaly_result["anomaly_score"],
                        "risk_factors": anomaly_result["risk_factors"],
                        "combined_risk_level": self._calculate_combined_risk(
                            cluster_result["is_noise"],
                            anomaly_result["is_anomaly"],
                            anomaly_result["anomaly_score"],
                            len(anomaly_result["risk_factors"])
                        ),
                        "features": cluster_result["features"]
                    }
                    combined_results.append(combined_result)
            
            return {
                "combined_analysis": combined_results,
                "clustering_metrics": clustering_results["metrics"],
                "anomaly_metrics": anomaly_results["metrics"],
                "cluster_statistics": clustering_results["cluster_statistics"],
                "summary": self._generate_analysis_summary(combined_results),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in combined analysis: {e}")
            return {
                "error": f"Combined analysis failed: {str(e)}",
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def _calculate_combined_risk(
        self, 
        is_noise: bool, 
        is_anomaly: bool, 
        anomaly_score: float, 
        risk_factor_count: int
    ) -> str:
        """Calculate combined risk level from clustering and anomaly detection"""
        risk_score = 0
        
        if is_noise:
            risk_score += 1
        if is_anomaly:
            risk_score += 2
        if abs(anomaly_score) > 0.5:
            risk_score += 1
        if risk_factor_count > 2:
            risk_score += 1
        
        if risk_score >= 4:
            return "critical"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "medium"
        elif risk_score >= 1:
            return "low"
        else:
            return "normal"
    
    def _generate_analysis_summary(self, combined_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for combined analysis"""
        total_addresses = len(combined_results)
        
        risk_distribution = {"normal": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        cluster_distribution = {}
        anomaly_count = 0
        noise_count = 0
        
        for result in combined_results:
            risk_level = result["combined_risk_level"]
            risk_distribution[risk_level] += 1
            
            cluster_id = result["cluster_id"]
            if cluster_id not in cluster_distribution:
                cluster_distribution[cluster_id] = 0
            cluster_distribution[cluster_id] += 1
            
            if result["is_anomaly"]:
                anomaly_count += 1
            if result["is_noise"]:
                noise_count += 1
        
        return {
            "total_addresses_analyzed": total_addresses,
            "risk_level_distribution": risk_distribution,
            "cluster_distribution": cluster_distribution,
            "anomalous_addresses": anomaly_count,
            "noise_addresses": noise_count,
            "high_risk_addresses": risk_distribution["high"] + risk_distribution["critical"],
            "analysis_quality": "good" if total_addresses > 50 else "limited"
        }
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            models = {
                'dbscan_model': self.dbscan_model,
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.utcnow().isoformat()
            }
            joblib.dump(models, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            models = joblib.load(filepath)
            self.dbscan_model = models['dbscan_model']
            self.isolation_forest = models['isolation_forest']
            self.scaler = models['scaler']
            self.pca = models['pca']
            self.feature_columns = models['feature_columns']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")