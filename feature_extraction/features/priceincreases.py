from ..base import BaseFeature
import pandas as pd
import numpy as np

class PriceIncreasesFeature(BaseFeature):
    """Generate PriceIncreases feature."""
    
    def __init__(self, time_window: float = 5000, threshold: float = 0.0):
        super().__init__(
            name=f"price-increase-next-{time_window}ms-with-{threshold}-threshold",
            description="Custom feature"
        )
        self.time_window = time_window
        self.threshold = threshold
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate a binary target: 1 if mid-price increases over the next 5 seconds, else 0.
        Assumes df_cleaned has a 'timestamp' column in milliseconds and level-1 bid/ask prices.
        
        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments
        
        Returns:
            pd.Series: The generated feature values
        """
        # Calculate mid-price
        time_window = self.time_window
        timestamps = df_cleaned.index.values
        target_times = timestamps + time_window
        idx_next = np.searchsorted(timestamps, target_times, side="left")
        idx_next[idx_next >= len(timestamps)] = len(timestamps) - 1
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        # Get future mid-prices
        future_mid_price = mid_price.values[idx_next]
        # Target: 1 if future mid-price > current, else 0
        target = (future_mid_price > mid_price.values+self.threshold).astype(int)
        return pd.Series(target, index=df_cleaned.index, name=self.name)
