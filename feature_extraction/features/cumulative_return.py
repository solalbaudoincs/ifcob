from feature_extraction.base import BaseFeature
import pandas as pd
import numpy as np

class CumulativeReturnVsVolatilityFeature(BaseFeature):
    """Generate a feature based on average return vs. local volatility over a time window."""
    def __init__(self, time: float = 10):
        super().__init__(
            name=f"return-vs-volatility-{time}-ms",
            description="Compare cumulative average return to volatility over a window."
        )
        self.time = time

    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        time_window = self.time
        timestamps = df_cleaned.index.values
        target_times = timestamps + time_window
        idx_next = np.searchsorted(timestamps, target_times, side="left")
        idx_next[idx_next >= len(timestamps)] = len(timestamps) - 1

        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        returns = mid_price.diff().fillna(0).to_numpy()

        # Cumulative sum and cumulative square sum
        cumsum = np.cumsum(returns)
        cumsum2 = np.cumsum(returns ** 2)

        # Padding for easier slicing
        cumsum_padded = np.concatenate([[0], cumsum])
        cumsum2_padded = np.concatenate([[0], cumsum2])

        start_idx = np.arange(len(returns))
        end_idx = idx_next

        sum_returns = cumsum_padded[end_idx + 1] - cumsum_padded[start_idx]
        sum_squares = cumsum2_padded[end_idx + 1] - cumsum2_padded[start_idx]
        window_lengths = (end_idx - start_idx + 1).astype(float)

        avg_returns = sum_returns / window_lengths
        var_returns = (sum_squares / window_lengths) - (avg_returns ** 2)
        std_returns = np.sqrt(np.maximum(var_returns, 1e-12))  # avoid sqrt of negative due to float errors

        result = np.zeros(len(returns), dtype=int)
        result[avg_returns > std_returns] = 1
        result[avg_returns < -std_returns] = -1

        return pd.Series(result, index=df_cleaned.index, name=self.name)
