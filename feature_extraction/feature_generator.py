"""
Feature Generation Module for Financial Order Book Data

This module provides a flexible and extensible system for generating features
from preprocessed order book data. Features can be generated individually or
in batches, and new features can be easily added.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from abc import ABC, abstractmethod
import warnings


try:
    from feature_extraction.features import (
        BidAskImbalanceFeature,
        SpreadFeature,
        BookSlopeFeature,
        VWAPFeature,
        LiquidityRatioFeature,
        VolatilityFeature,
        MomentumFeature,
        TrendFeature,
        VolumeFeature,
        CumulativeVolumeFeature,
        InstReturnFeature,
        ReturnsAllSignedForXmsFeature,
        CumulativeReturnVsVolatilityFeature,
        CumulativeReturnTransferEntropy,
        SharpeRatioClassificationFeature,
        SharpeRatioTransferEntropy,

    )
except ImportError:
    from features import (
        BidAskImbalanceFeature,
        SpreadFeature,
        BookSlopeFeature,
        VWAPFeature,
        LiquidityRatioFeature,
        VolatilityFeature,
        MomentumFeature,
        TrendFeature,
        VolumeFeature,
        CumulativeVolumeFeature,
        InstReturnFeature,
        ReturnsAllSignedForXmsFeature,
        CumulativeReturnVsVolatilityFeature,
        CumulativeReturnTransferEntropy,
        SharpeRatioClassificationFeature,
        SharpeRatioTransferEntropy,
    )

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class BaseFeature(ABC):
    """Base class for all feature generators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate the feature from cleaned data."""
        pass


class FeatureGenerator:
    """Main feature generator class that manages all features."""
    
    def __init__(self):
        self.features: Dict[str, BaseFeature] = {}
        self._register_default_features()
    
    def _register_default_features(self):
        """Register all default features."""
        # Default parameters
        n_levels = 5
        window = 20
        
        # Register all default features
        self.register_feature(BidAskImbalanceFeature(n_levels))
        self.register_feature(SpreadFeature())
        
        # Book slope features
        self.register_feature(BookSlopeFeature('bid', n_levels))
        self.register_feature(BookSlopeFeature('ask', n_levels))
        
        # VWAP features
        self.register_feature(VWAPFeature('bid', n_levels))
        self.register_feature(VWAPFeature('ask', n_levels))
        
        # Liquidity ratio
        self.register_feature(LiquidityRatioFeature(n_levels))
        
        # Rate features
        self.register_feature(VolatilityFeature(window))
        self.register_feature(MomentumFeature(window))
        self.register_feature(TrendFeature(window))
        
        # Volume features
        self.register_feature(VolumeFeature('bid', 1, window))
        self.register_feature(VolumeFeature('ask', 1, window))
        
        # Cumulative volume features
        self.register_feature(CumulativeVolumeFeature('bid', n_levels))
        self.register_feature(CumulativeVolumeFeature('ask', n_levels))

        self.register_feature(InstReturnFeature())

        self.register_feature(ReturnsAllSignedForXmsFeature(5))
        self.register_feature(ReturnsAllSignedForXmsFeature(10))
        self.register_feature(ReturnsAllSignedForXmsFeature(20))

        # Cumulative Return 
        self.register_feature(CumulativeReturnVsVolatilityFeature(5))
        self.register_feature(CumulativeReturnVsVolatilityFeature(10))
        self.register_feature(CumulativeReturnVsVolatilityFeature(20))
        self.register_feature(CumulativeReturnTransferEntropy(5))

        #Sharpe ratio
        self.register_feature(SharpeRatioClassificationFeature(5))
        self.register_feature(SharpeRatioTransferEntropy(5))
    
    def register_feature(self, feature: BaseFeature):
        """Register a new feature."""
        self.features[feature.name] = feature
    
    def unregister_feature(self, feature_name: str):
        """Unregister a feature."""
        if feature_name in self.features:
            del self.features[feature_name]
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self.features.keys())
    
    def get_feature_info(self, feature_name: str) -> str:
        """Get description of a feature."""
        if feature_name in self.features:
            return self.features[feature_name].description
        return f"Feature '{feature_name}' not found"
    
    def generate_feature(self, feature_name: str, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate a single feature."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' is not registered")
        
        return self.features[feature_name].generate(df_cleaned, **kwargs)
    
    def generate_features(self, df_cleaned: pd.DataFrame, feature_names: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Generate multiple features."""
        if feature_names is None:
            feature_names = self.list_features()
        
        features_df = pd.DataFrame(index=df_cleaned.index)
        
        for feature_name in feature_names:
            if feature_name in self.features:
                try:
                    features_df[feature_name] = self.generate_feature(feature_name, df_cleaned, **kwargs)
                except Exception as e:
                    warnings.warn(f"Failed to generate feature '{feature_name}': {str(e)}")
            else:
                warnings.warn(f"Feature '{feature_name}' is not registered")
        
        for col in df_cleaned.columns:
            features_df[col] = df_cleaned[col]
        
        return features_df
    
    def generate_all_features(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate all registered features."""
        return self.generate_features(df_cleaned, **kwargs)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw order book data to create df_cleaned.
    
    This function computes weighted average prices and volumes for each level and side
    by grouping data by timestamp (ignoring row_id).
    """
    df_cleaned = pd.DataFrame(index=df.index.get_level_values(0).unique())
    
    # For each level and side, compute the volume-weighted average price per timestamp
    for side in ['bid', 'ask']:
        for i in range(1, 11):  # Assuming 10 levels
            price_col = f'level-{i}-{side}-price'
            volume_col = f'level-{i}-{side}-volume'
            
            # Check if columns exist in the data
            if price_col in df.columns and volume_col in df.columns:
                # Weighted average price per timestamp
                weighted_avg_price = (
                    df[price_col] * df[volume_col]
                ).groupby(level=0).sum() / df[volume_col].groupby(level=0).sum()
                
                # Average volume per timestamp
                avg_volume = df[volume_col].groupby(level=0).mean()
                
                # Assign to df_cleaned
                df_cleaned[price_col] = weighted_avg_price
                df_cleaned[volume_col] = avg_volume
    
    return df_cleaned


def load_and_generate_features(coin: str, data_version: int, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load data and generate features in one step.
    
    Args:
        coin: Coin symbol (e.g., 'ETH', 'XBT')
        data_version: Data version number (e.g., 0, 1, 2)
        feature_names: List of specific features to generate (None for all)
    
    Returns:
        DataFrame with generated features
    """    # Load preprocessed data
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'preprocessed', f'DATA_{data_version}', f'{coin}_EUR.parquet')
    df = pd.read_parquet(data_path)
    
    # Preprocess data
    df_cleaned = preprocess_data(df)
    
    # Generate features
    generator = FeatureGenerator()
    features = generator.generate_features(df_cleaned, feature_names)
    
    return features


def save_features(features: pd.DataFrame, coin: str, data_version: int, output_dir: str = None):
    """Save features to parquet file."""
    import os
    
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, 'data', 'features')
    
    output_path = os.path.join(output_dir, f'DATA_{data_version}')
    os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, f'{coin}_EUR.parquet')
    features.to_parquet(file_path, index=True)
    print(f"Features saved to {file_path}")


def recompute_feature(coin: str, data_version: int, feature_name: str, output_dir: str = None):
    """
    Recompute a single feature for a coin and data version, updating the features file in place.
    Other features are left untouched and still present.

    If the features file exists, it is assumed to already contain the correct index and columns (from clean_df).
    If the required columns for the feature are missing, the function will load and preprocess the raw data to obtain them.

    Args:
        coin: Coin symbol (e.g., 'ETH', 'XBT')
        data_version: Data version number (e.g., 0, 1, 2)
        feature_name: Name of the feature to recompute
        output_dir: Directory where features are stored (default: project_root/data/features)
    """
    import os
    import pandas as pd

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'features')
    output_path = os.path.join(output_dir, f'DATA_{data_version}')
    os.makedirs(output_path, exist_ok=True)
    features_file = os.path.join(output_path, f'{coin}_EUR.parquet')

    generator = FeatureGenerator()
    if feature_name not in generator.features:
        raise ValueError(f"Feature '{feature_name}' is not registered.")

    # If features file exists, use its index and columns for recomputation
    if os.path.exists(features_file):
        features = pd.read_parquet(features_file)
        required_cols = generator.features[feature_name].generate.__code__.co_varnames
        # Try to recompute using existing features file
        try:
            new_feature = generator.features[feature_name].generate(features)
            features[feature_name] = new_feature
            features.to_parquet(features_file, index=True)
            print(f"Feature '{feature_name}' recomputed (in-place) and saved to {features_file}")
            return
        except KeyError as e:
            print(f"Missing columns in features file: {e}. Falling back to raw data.")
            # Fallback to raw data below

    # If features file does not exist or required columns are missing, load and preprocess data
    data_path = os.path.join(project_root, 'data', 'preprocessed', f'DATA_{data_version}', f'{coin}_EUR.parquet')
    df = pd.read_parquet(data_path)
    df_cleaned = preprocess_data(df)
    new_feature = generator.generate_feature(feature_name, df_cleaned)
    if os.path.exists(features_file):
        features = pd.read_parquet(features_file)
        features[feature_name] = new_feature
    else:
        features = pd.DataFrame(index=df_cleaned.index)
        features[feature_name] = new_feature
    features.to_parquet(features_file, index=True)
    print(f"Feature '{feature_name}' recomputed and saved to {features_file}")

# Example CLI usage (add to your CLI script):
# from feature_extraction.feature_generator import recompute_feature
# recompute_feature('ETH', 1, 'spread')
    
if __name__ == "__main__":
    # Example usage
    generator = FeatureGenerator()
    
    # List all features
    print("Available features:")
    for feature_name in generator.list_features():
        print(f"  - {feature_name}: {generator.get_feature_info(feature_name)}")
    
    # Generate features for ETH data
    features = load_and_generate_features('ETH', 2)
    print(f"\nGenerated {len(features.columns)} features for {len(features)} timestamps")
    print(f"Feature columns: {list(features.columns)}")
