"""
DBSCAN Clustering Module for ETHEREYE
====================================

This module implements DBSCAN clustering for identifying groups of similar
blockchain addresses and transaction patterns. Useful for detecting:
- Address clusters belonging to the same entity
- Similar transaction behavior patterns
- Potential coordinated activities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClusteringResults:
    """Container for clustering results"""
    labels: np.ndarray
    core_samples: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_centers: np.ndarray
    cluster_stats: Dict[int, Dict[str, Any]]

class DBSCANClustering:
    """
    DBSCAN clustering implementation for blockchain address analysis
    """
    
    def __init__(self, 
                 eps: float = 0.5, 
                 min_samples: int = 5, 
                 metric: str = 'euclidean',
                 algorithm: str = 'auto'):
        """
        Initialize DBSCAN clustering
        
        Args:
            eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples: Number of samples in a neighborhood for a point to be considered as a core point
            metric: Distance metric to use
            algorithm: Algorithm used by NearestNeighbors module
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.results = None
        
    def fit(self, X: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> ClusteringResults:
        """
        Fit DBSCAN clustering to the data
        
        Args:
            X: Input data
            feature_columns: Columns to use for clustering
            
        Returns:
            ClusteringResults object
        """
        logger.info("Starting DBSCAN clustering...")
        
        # Prepare data
        if feature_columns:
            self.feature_columns = feature_columns
            X_features = X[feature_columns]
        else:
            X_features = X.select_dtypes(include=[np.number])
            self.feature_columns = X_features.columns.tolist()
        
        # Handle missing values
        X_features = X_features.fillna(X_features.median())
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit DBSCAN
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm
        )
        
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate silhouette score (only if we have clusters)
        if n_clusters > 1 and n_noise < len(labels) - 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        else:
            silhouette_avg = -1
            calinski_harabasz = -1
        
        # Calculate cluster centers (mean of each cluster)
        cluster_centers = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_mask = labels == label
                center = X_scaled[cluster_mask].mean(axis=0)
                cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_stats(X, labels)
        
        self.results = ClusteringResults(
            labels=labels,
            core_samples=self.model.core_sample_indices_,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette_score=silhouette_avg,
            calinski_harabasz_score=calinski_harabasz,
            cluster_centers=cluster_centers,
            cluster_stats=cluster_stats
        )
        
        logger.info(f"Clustering completed. Found {n_clusters} clusters with {n_noise} noise points")
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data
        Note: DBSCAN doesn't naturally support prediction on new data,
        so we use nearest cluster center approach
        
        Args:
            X: New data to predict
            
        Returns:
            Predicted cluster labels
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data
        X_features = X[self.feature_columns].fillna(X[self.feature_columns].median())
        X_scaled = self.scaler.transform(X_features)
        
        # Find nearest cluster center for each point
        if len(self.results.cluster_centers) == 0:
            return np.full(len(X), -1)  # All noise if no clusters
        
        # Calculate distances to all cluster centers
        from scipy.spatial.distance import cdist
        distances = cdist(X_scaled, self.results.cluster_centers, metric=self.metric)
        
        # Assign to nearest cluster if within eps distance
        labels = []
        for i, point_distances in enumerate(distances):
            min_distance_idx = np.argmin(point_distances)
            min_distance = point_distances[min_distance_idx]
            
            if min_distance <= self.eps:
                labels.append(min_distance_idx)
            else:
                labels.append(-1)  # Noise
        
        return np.array(labels)
    
    def optimize_parameters(self, X: pd.DataFrame, 
                          eps_range: List[float] = None, 
                          min_samples_range: List[int] = None) -> Dict[str, Any]:
        """
        Optimize DBSCAN parameters using grid search
        
        Args:
            X: Input data
            eps_range: Range of eps values to test
            min_samples_range: Range of min_samples values to test
            
        Returns:
            Dictionary with best parameters and scores
        """
        if eps_range is None:
            eps_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        if min_samples_range is None:
            min_samples_range = [3, 5, 10, 15, 20]
        
        logger.info("Optimizing DBSCAN parameters...")
        
        best_score = -1
        best_params = {}
        results = []
        
        # Prepare data
        X_features = X.select_dtypes(include=[np.number]).fillna(X.select_dtypes(include=[np.number]).median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    # Fit DBSCAN with current parameters
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # Skip if no clusters or all noise
                    if n_clusters < 2 or n_noise == len(labels):
                        continue
                    
                    # Calculate silhouette score
                    score = silhouette_score(X_scaled, labels)
                    
                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': score,
                        'noise_ratio': n_noise / len(labels)
                    }
                    
                    results.append(result)
                    
                    # Update best parameters based on silhouette score and noise ratio
                    adjusted_score = score * (1 - result['noise_ratio'] * 0.5)
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_params = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'score': score,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise
                        }
                
                except Exception as e:
                    logger.warning(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        logger.info(f"Best parameters: {best_params}")
        return {
            'best_params': best_params,
            'all_results': results
        }
    
    def _calculate_cluster_stats(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each cluster"""
        cluster_stats = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = X[cluster_mask]
            
            if label == -1:  # Noise points
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{label}'
            
            stats = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(labels) * 100)
            }
            
            # Calculate statistics for numerical columns
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if not cluster_data[col].empty:
                    stats[f'{col}_mean'] = float(cluster_data[col].mean())
                    stats[f'{col}_std'] = float(cluster_data[col].std())
                    stats[f'{col}_median'] = float(cluster_data[col].median())
                    stats[f'{col}_min'] = float(cluster_data[col].min())
                    stats[f'{col}_max'] = float(cluster_data[col].max())
            
            cluster_stats[int(label)] = stats
        
        return cluster_stats
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary of all clusters
        
        Returns:
            DataFrame with cluster summary statistics
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        summary_data = []
        for label, stats in self.results.cluster_stats.items():
            row = {
                'cluster_id': label,
                'cluster_name': 'noise' if label == -1 else f'cluster_{label}',
                'size': stats['size'],
                'percentage': stats['percentage']
            }
            
            # Add key statistics
            for key, value in stats.items():
                if key not in ['size', 'percentage'] and '_mean' in key:
                    feature_name = key.replace('_mean', '')
                    row[f'{feature_name}_avg'] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def visualize_clusters(self, X: pd.DataFrame, 
                          features_to_plot: List[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize clustering results
        
        Args:
            X: Original data
            features_to_plot: Features to include in visualization
            figsize: Figure size
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        # Select features for plotting
        if features_to_plot is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            features_to_plot = numeric_cols[:4].tolist()  # Take first 4 numeric columns
        
        # If we have more than 2 features, use PCA for visualization
        if len(features_to_plot) > 2:
            X_plot = X[features_to_plot].fillna(X[features_to_plot].median())
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X_plot))
            plot_data = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            x_col, y_col = 'PC1', 'PC2'
            title_suffix = f" (PCA of {', '.join(features_to_plot)})"
        else:
            plot_data = X[features_to_plot[:2]]
            x_col, y_col = features_to_plot[0], features_to_plot[1]
            title_suffix = ""
        
        plot_data['cluster'] = self.results.labels
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'DBSCAN Clustering Results{title_suffix}', fontsize=16)
        
        # Scatter plot of clusters
        ax1 = axes[0, 0]
        unique_labels = set(self.results.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                cluster_data = plot_data[plot_data['cluster'] == label]
                ax1.scatter(cluster_data[x_col], cluster_data[y_col], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                cluster_data = plot_data[plot_data['cluster'] == label]
                ax1.scatter(cluster_data[x_col], cluster_data[y_col], 
                           c=[color], s=50, alpha=0.6, label=f'Cluster {label}')
        
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title('Cluster Visualization')
        ax1.legend()
        
        # Cluster size distribution
        ax2 = axes[0, 1]
        cluster_sizes = [stats['size'] for label, stats in self.results.cluster_stats.items() if label != -1]
        cluster_labels = [f'Cluster {label}' for label in self.results.cluster_stats.keys() if label != -1]
        
        if cluster_sizes:
            ax2.bar(cluster_labels, cluster_sizes)
            ax2.set_title('Cluster Size Distribution')
            ax2.set_ylabel('Number of Points')
            plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Silhouette analysis (if applicable)
        ax3 = axes[1, 0]
        if self.results.n_clusters > 1:
            from sklearn.metrics import silhouette_samples
            X_features = X[self.feature_columns].fillna(X[self.feature_columns].median())
            X_scaled = self.scaler.transform(X_features)
            silhouette_values = silhouette_samples(X_scaled, self.results.labels)
            
            y_lower = 10
            for i, label in enumerate(sorted(set(self.results.labels))):
                if label == -1:
                    continue
                
                cluster_silhouette_values = silhouette_values[self.results.labels == label]
                cluster_silhouette_values.sort()
                
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                ax3.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, cluster_silhouette_values, alpha=0.7)
                
                ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                y_lower = y_upper + 10
            
            ax3.axvline(x=self.results.silhouette_score, color="red", linestyle="--")
            ax3.set_title('Silhouette Analysis')
        
        # Summary statistics
        ax4 = axes[1, 1]
        summary_text = f"""Clustering Summary:
        
        Number of Clusters: {self.results.n_clusters}
        Noise Points: {self.results.n_noise}
        Silhouette Score: {self.results.silhouette_score:.3f}
        Calinski-Harabasz Score: {self.results.calinski_harabasz_score:.3f}
        
        Parameters:
        eps: {self.eps}
        min_samples: {self.min_samples}
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
                verticalalignment='center', fontsize=10)
        ax4.axis('off')
        ax4.set_title('Clustering Summary')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, X: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Export clustering results
        
        Args:
            X: Original data
            output_path: Path to save results CSV
            
        Returns:
            DataFrame with clustering results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        # Create results dataframe
        results_df = X.copy()
        results_df['cluster_id'] = self.results.labels
        results_df['cluster_name'] = ['noise' if label == -1 else f'cluster_{label}' 
                                     for label in self.results.labels]
        results_df['is_core_sample'] = False
        results_df.loc[self.results.core_samples, 'is_core_sample'] = True
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results exported to {output_path}")
        
        return results_df