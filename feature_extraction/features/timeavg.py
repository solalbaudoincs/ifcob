from ..base import BaseFeature
import pandas as pd
import numpy as np


class TimeAvgFeature(BaseFeature):
    """Generate TimeAvg feature."""
    
    def __init__(self, base_feature, time_window : float = 5000):
        super().__init__(
            name=f"avg-{time_window}ms-of-{base_feature.name}",
            description="Custom feature"
        )
        self.time_window = time_window
        self.base_feature = base_feature
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the TimeAvg feature.
        
        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments
        
        Returns:
            pd.Series: The generated feature values
        """
        base_generated = self.base_feature.generate(df_cleaned, **kwargs)

        # Implement the logic to calculate the average over the specified time window

        timpestamps = df_cleaned.index.values

        idx_prev = np.searchsorted(timpestamps, timpestamps - self.time_window)
        idx_prev[idx_prev >= len(timpestamps)] = len(timpestamps) - 1
        idx_prev[idx_prev < 0] = 0

        cumsum = np.insert(base_generated.values.cumsum(), 0, 0)
        start_idx = idx_prev
        end_idx = np.arange(len(timpestamps)) + 1

        div = end_idx - start_idx
        div[div < 1] = 1
        average = (cumsum[end_idx] - cumsum[start_idx]) / div
        
        return pd.Series(average, index=df_cleaned.index, name=self.name)
