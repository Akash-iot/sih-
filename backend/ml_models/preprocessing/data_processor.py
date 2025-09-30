"""
Data Preprocessing Module for ETHEREYE ML Pipeline
=================================================

This module handles preprocessing of blockchain data for machine learning models.
Includes feature engineering, normalization, and data transformation utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing for blockchain analytics
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_columns = []
        
    def preprocess_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data for ML models
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing transaction data")
        
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            
        # Extract temporal features
        processed_df = self._extract_temporal_features(processed_df)
        
        # Calculate transaction metrics
        processed_df = self._calculate_transaction_metrics(processed_df)
        
        # Engineer address-based features
        processed_df = self._engineer_address_features(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Normalize numerical features
        processed_df = self._normalize_features(processed_df)
        
        return processed_df
    
    def preprocess_address_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess address data for clustering and risk scoring
        
        Args:
            df: DataFrame with address data
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing address data")
        
        processed_df = df.copy()
        
        # Calculate address statistics
        processed_df = self._calculate_address_statistics(processed_df)
        
        # Extract behavioral patterns
        processed_df = self._extract_behavioral_patterns(processed_df)
        
        # Create interaction features
        processed_df = self._create_interaction_features(processed_df)
        
        # Normalize features
        processed_df = self._normalize_features(processed_df)
        
        return processed_df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from transaction data"""
        if 'timestamp' not in df.columns:
            return df
            
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Time since first transaction
        if len(df) > 0:
            first_tx = df['timestamp'].min()
            df['days_since_first'] = (df['timestamp'] - first_tx).dt.days
        
        return df
    
    def _calculate_transaction_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction-specific metrics"""
        # Value-based features
        if 'value' in df.columns:
            df['log_value'] = np.log1p(df['value'])
            df['value_squared'] = df['value'] ** 2
        
        # Gas-based features
        if 'gas_used' in df.columns and 'gas_price' in df.columns:
            df['total_gas_cost'] = df['gas_used'] * df['gas_price']
            df['log_gas_cost'] = np.log1p(df['total_gas_cost'])
        
        # Transaction complexity indicators
        if 'input_data' in df.columns:
            df['input_length'] = df['input_data'].str.len()
            df['has_input_data'] = (df['input_length'] > 2).astype(int)
        
        return df
    
    def _engineer_address_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features based on addresses"""
        # From address features
        if 'from_address' in df.columns:
            from_stats = df.groupby('from_address').agg({
                'value': ['count', 'sum', 'mean', 'std'],
                'timestamp': ['min', 'max']
            }).fillna(0)
            
            from_stats.columns = ['from_tx_count', 'from_total_value', 'from_avg_value', 
                                'from_std_value', 'from_first_tx', 'from_last_tx']
            
            df = df.merge(from_stats, left_on='from_address', right_index=True, how='left')
        
        # To address features
        if 'to_address' in df.columns:
            to_stats = df.groupby('to_address').agg({
                'value': ['count', 'sum', 'mean', 'std'],
                'timestamp': ['min', 'max']
            }).fillna(0)
            
            to_stats.columns = ['to_tx_count', 'to_total_value', 'to_avg_value',
                               'to_std_value', 'to_first_tx', 'to_last_tx']
            
            df = df.merge(to_stats, left_on='to_address', right_index=True, how='left')
        
        return df
    
    def _calculate_address_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive address statistics"""
        # Transaction frequency patterns
        df['avg_tx_per_day'] = df['tx_count'] / ((df['last_tx_date'] - df['first_tx_date']).dt.days + 1)
        
        # Value distribution metrics
        df['value_concentration'] = df['max_tx_value'] / df['total_value'] if 'total_value' in df.columns else 0
        df['value_coefficient_variation'] = df['std_tx_value'] / df['avg_tx_value'] if 'avg_tx_value' in df.columns else 0
        
        # Activity patterns
        df['active_days'] = df['unique_days_active']
        df['burst_ratio'] = df['max_daily_txs'] / df['avg_daily_txs'] if 'avg_daily_txs' in df.columns else 1
        
        return df
    
    def _extract_behavioral_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral patterns from address data"""
        # Round-number preference (psychological indicator)
        if 'avg_tx_value' in df.columns:
            df['prefers_round_numbers'] = (df['avg_tx_value'] % 1000 < 100).astype(int)
        
        # Interaction diversity
        if 'unique_counterparties' in df.columns and 'tx_count' in df.columns:
            df['counterparty_diversity'] = df['unique_counterparties'] / df['tx_count']
        
        # Time pattern regularity
        if 'std_time_between_txs' in df.columns and 'avg_time_between_txs' in df.columns:
            df['time_regularity'] = 1 / (1 + df['std_time_between_txs'] / df['avg_time_between_txs'])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different metrics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create some key interaction features
        for i, col1 in enumerate(numeric_cols[:5]):  # Limit to prevent explosion
            for col2 in numeric_cols[i+1:6]:
                if col1 != col2:
                    interaction_name = f"{col1}_x_{col2}"
                    df[interaction_name] = df[col1] * df[col2]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['main'] = scaler
        
        return df
    
    def prepare_text_features(self, texts: List[str], max_features: int = 1000) -> np.ndarray:
        """
        Prepare text features using TF-IDF vectorization
        
        Args:
            texts: List of text strings
            max_features: Maximum number of features to extract
            
        Returns:
            TF-IDF feature matrix
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        self.vectorizers['text'] = vectorizer
        
        return tfidf_matrix.toarray()
    
    def reduce_dimensionality(self, data: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Reduce dimensionality using PCA
        
        Args:
            data: Input data matrix
            n_components: Number of components to keep
            
        Returns:
            Reduced dimension data
        """
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        self.encoders['pca'] = pca
        
        return reduced_data
    
    def get_feature_importance_scores(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, float]:
        """
        Calculate feature importance scores
        
        Args:
            df: DataFrame with features
            target_col: Target column for supervised importance
            
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col:
            feature_cols = [col for col in numeric_cols if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
            
            # Determine if classification or regression
            if y.dtype == 'object' or y.nunique() < 10:
                # Classification
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importance_scores = dict(zip(feature_cols, rf.feature_importances_))
            else:
                # Regression
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importance_scores = dict(zip(feature_cols, rf.feature_importances_))
        else:
            # Unsupervised importance (variance-based)
            importance_scores = {}
            for col in numeric_cols:
                importance_scores[col] = df[col].var()
        
        return importance_scores
    
    def create_sliding_window_features(self, df: pd.DataFrame, 
                                     time_col: str, 
                                     value_cols: List[str], 
                                     window_sizes: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Create sliding window aggregation features
        
        Args:
            df: DataFrame with time series data
            time_col: Column name for timestamps
            value_cols: Columns to aggregate
            window_sizes: Window sizes in days
            
        Returns:
            DataFrame with sliding window features
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        for window_size in window_sizes:
            for col in value_cols:
                # Rolling statistics
                df[f"{col}_rolling_mean_{window_size}d"] = df[col].rolling(
                    window=f"{window_size}D", on=time_col
                ).mean()
                
                df[f"{col}_rolling_std_{window_size}d"] = df[col].rolling(
                    window=f"{window_size}D", on=time_col
                ).std()
                
                df[f"{col}_rolling_max_{window_size}d"] = df[col].rolling(
                    window=f"{window_size}D", on=time_col
                ).max()
                
                df[f"{col}_rolling_min_{window_size}d"] = df[col].rolling(
                    window=f"{window_size}D", on=time_col
                ).min()
        
        return df