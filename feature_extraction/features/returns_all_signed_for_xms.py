from feature_extraction.base import BaseFeature
import pandas as pd
import numpy as np

class ReturnsAllSignedForXmsFeature(BaseFeature):
    """Generate return-all-signed-for-x-ms feature."""
    def __init__(self, time : float = 10):
        super().__init__(
            name=f"return-all-signed-for-{time}-ms",
            description="return-evolution-for-x-ms feature, it will probably be used as a target for classification."
        )
        self.time = time
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        time_window = self.time
        timestamps = df_cleaned.index.values
        target_times = timestamps + time_window
        idx_next = np.searchsorted(timestamps, target_times, side="left")
        idx_next[idx_next >= len(timestamps)] = len(timestamps) - 1
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        returns = mid_price.diff()
        positive_mask = returns > 0
        cumsum = np.cumsum(~positive_mask)
        start_idx = np.arange(len(returns))
        end_idx = idx_next
        cumsum_padded = np.concatenate([[0], cumsum])
        num_non_positive = cumsum_padded[end_idx + 1] - cumsum_padded[start_idx]
        all_positive =  (0 == num_non_positive)
        all_negative = (end_idx - start_idx + 1) == num_non_positive
        result = np.zeros(len(returns), dtype=int)
        result[all_positive] = 1
        result[all_negative] = -1
        return pd.Series(result, index=df_cleaned.index, name=self.name)
