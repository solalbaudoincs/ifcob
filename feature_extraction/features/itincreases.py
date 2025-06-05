from ..base import BaseFeature
import pandas as pd
import numpy as np


class ItIncreasesFeature(BaseFeature):
    """Generate ItIncreases feature."""
    
    def __init__(self, base_feature: BaseFeature, time_window : float = 200, threshold: float = 0.0):
        super().__init__(
            name=f"{base_feature.name}-itincreases-after-{time_window}ms-with-threshold-{threshold}",
            description="Custom feature"
        )
        self.time_window = time_window
        self.threshold = threshold
        self.base_feature = base_feature
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the ItIncreases feature.
        
        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments
        
        Returns:
            pd.Series: The generated feature values
        """
        serie = self.base_feature.generate(df_cleaned, **kwargs)
        time_window = self.time_window
        timestamps = df_cleaned.index.values
        target_times = timestamps + time_window
        idx_next = np.searchsorted(timestamps, target_times, side="left")
        idx_next[idx_next >= len(timestamps)] = len(timestamps) - 1
        # Get future mid-prices
        future = serie.values[idx_next]
        # Target: 1 if future mid-price > current, else 0
        target = (future > serie+self.threshold).astype(int)
        return pd.Series(target, index=df_cleaned.index, name=self.name)
