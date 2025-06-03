"""
Feature Generation Module for Financial Order Book Data

This module provides a flexible and extensible system for generating features
from preprocessed order book data. Features can be generated individually or
in batches, and new features can be easily added.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from abc import ABC, abstractmethod
import warnings


class BaseFeature(ABC):
    """Base class for all feature generators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate the feature from cleaned data."""
        pass


class BidAskImbalanceFeature(BaseFeature):
    """Generate bid-ask imbalance features."""
    
    def __init__(self, n_levels: int = 5):
        super().__init__(
            name=f"bid-ask-imbalance-{n_levels}-levels",
            description=f"Bid-ask imbalance using top {n_levels} levels. "
                       "Close to 1 → strong buying pressure, Close to -1 → strong selling pressure, Near 0 → balanced depth."
        )
        self.n_levels = n_levels
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        # Calculate cumulative volumes for bid and ask sides
        v_bid = df_cleaned["level-1-bid-volume"].copy()
        v_ask = df_cleaned["level-1-ask-volume"].copy()
        
        for i in range(2, self.n_levels + 1):
            v_bid += df_cleaned[f"level-{i}-bid-volume"]
            v_ask += df_cleaned[f"level-{i}-ask-volume"]
        
        # Calculate imbalance: (V_bid - V_ask)/(V_bid + V_ask)
        return (v_bid - v_ask) / (v_bid + v_ask)


class SpreadFeature(BaseFeature):
    """Generate bid-ask spread feature."""
    
    def __init__(self):
        super().__init__(
            name="spread",
            description="Bid-ask spread: level-1-ask-price - level-1-bid-price"
        )
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        return df_cleaned["level-1-ask-price"] - df_cleaned["level-1-bid-price"]


class BookSlopeFeature(BaseFeature):
    """Generate book slope features for bid and ask sides."""
    
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        
        super().__init__(
            name=f"slope-{side}-{n_levels}-levels",
            description=f"Book slope for {side} side using {n_levels} levels. "
                       "Measures steepness of liquidity: (P_N - P_1) / V_sum"
        )
        self.side = side
        self.n_levels = n_levels
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        price_cols = [f"level-{i}-{self.side}-price" for i in range(1, self.n_levels + 1)]
        volume_cols = [f"level-{i}-{self.side}-volume" for i in range(1, self.n_levels + 1)]
        
        p_n = df_cleaned[price_cols[-1]]
        p_1 = df_cleaned[price_cols[0]]
        v_sum = sum([df_cleaned[col] for col in volume_cols])
        
        return (p_n - p_1) / v_sum


class VWAPFeature(BaseFeature):
    """Generate Volume-Weighted Average Price features."""
    
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        
        super().__init__(
            name=f"vwap-{side}-{n_levels}-levels",
            description=f"Volume-Weighted Average Price for {side} side using {n_levels} levels"
        )
        self.side = side
        self.n_levels = n_levels
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        price_cols = [f"level-{i}-{self.side}-price" for i in range(1, self.n_levels + 1)]
        volume_cols = [f"level-{i}-{self.side}-volume" for i in range(1, self.n_levels + 1)]
        
        prices = df_cleaned[price_cols]
        volumes = df_cleaned[volume_cols]
        
        return (prices * volumes).sum(axis=1) / volumes.sum(axis=1)


class LiquidityRatioFeature(BaseFeature):
    """Generate liquidity ratio feature."""
    
    def __init__(self, n_levels: int = 5):
        super().__init__(
            name="liquidity-ratio",
            description=f"Liquidity ratio: V_bid/V_ask using {n_levels} levels"
        )
        self.n_levels = n_levels
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        # Calculate cumulative volumes
        v_bid = df_cleaned["level-1-bid-volume"].copy()
        v_ask = df_cleaned["level-1-ask-volume"].copy()
        
        for i in range(2, self.n_levels + 1):
            v_bid += df_cleaned[f"level-{i}-bid-volume"]
            v_ask += df_cleaned[f"level-{i}-ask-volume"]
        
        return v_bid / v_ask


class VolatilityFeature(BaseFeature):
    """Generate instantaneous volatility feature."""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name="rate-inst-volatility",
            description=f"Instantaneous volatility using rolling window of {window} periods"
        )
        self.window = window
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.rolling(window=self.window).var()


class MomentumFeature(BaseFeature):
    """Generate momentum feature."""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name="rate-momentum",
            description=f"Momentum: change in mid-price over {window} periods"
        )
        self.window = window
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.diff(periods=self.window)


class TrendFeature(BaseFeature):
    """Generate trend indicator feature."""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name="rate-mid-price-trend",
            description=f"Trend indicator: rolling mean of mid-price over {window} periods"
        )
        self.window = window
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.rolling(window=self.window).mean()


class VolumeFeature(BaseFeature):
    """Generate volume features."""
    
    def __init__(self, side: str, level: int = 1, window: int = 20):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        
        super().__init__(
            name=f"rate-{side}-volume-level-{level}",
            description=f"Average {side} volume at level {level} over {window} periods"
        )
        self.side = side
        self.level = level
        self.window = window
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        volume_col = f"level-{self.level}-{self.side}-volume"
        return df_cleaned[volume_col].rolling(window=self.window).mean()


class CumulativeVolumeFeature(BaseFeature):
    """Generate cumulative volume features."""
    
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        
        super().__init__(
            name=f"V-{side}-{n_levels}-levels",
            description=f"Cumulative {side} volume across {n_levels} levels"
        )
        self.side = side
        self.n_levels = n_levels
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        volume = df_cleaned[f"level-1-{self.side}-volume"].copy()
        
        for i in range(2, self.n_levels + 1):
            volume += df_cleaned[f"level-{i}-{self.side}-volume"]
        
        return volume


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

    If the features file exists, it is assumed to already contain the correct index and columns (from clean_df),
    so recomputing clean_df and loading preprocessed data is not needed.

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
        # Only recompute the requested feature using the existing features DataFrame
        # (Assume features file has all columns from clean_df)
        new_feature = generator.features[feature_name].generate(features)
        features[feature_name] = new_feature
        features.to_parquet(features_file, index=True)
        print(f"Feature '{feature_name}' recomputed (in-place) and saved to {features_file}")
        return

    # If features file does not exist, fall back to old logic (load and preprocess data)
    data_path = os.path.join(project_root, 'data', 'preprocessed', f'DATA_{data_version}', f'{coin}_EUR.parquet')
    df = pd.read_parquet(data_path)
    df_cleaned = preprocess_data(df)
    new_feature = generator.generate_feature(feature_name, df_cleaned)
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
