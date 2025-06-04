from feature_extraction.base import BaseFeature
import pandas as pd
import numpy as np

from feature_extraction.base import BaseFeature
import pandas as pd
import numpy as np

class SharpeRatioClassificationFeature(BaseFeature):
    """Classify signal into 3 classes using a dynamically calibrated Sharpe-like ratio."""

    def __init__(self, time: float = 10):
        super().__init__(
            name=f"sharpe-ratio-quantile-calibrated-{time}-ms",
            description="Sharpe-like ratio with dynamic threshold to enforce ~10%-80%-10% class split."
        )
        self.time = time

    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        time_window = self.time
        timestamps = df_cleaned.index.values
        target_times = timestamps + time_window
        idx_next = np.searchsorted(timestamps, target_times, side="left")
        idx_next[idx_next >= len(timestamps)] = len(timestamps) - 1

        # Mid-price and returns
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        returns = mid_price.diff().fillna(0).to_numpy()

        # Cumulative sums
        cumsum = np.cumsum(returns)
        cumsum2 = np.cumsum(returns ** 2)
        cumsum_padded = np.concatenate([[0], cumsum])
        cumsum2_padded = np.concatenate([[0], cumsum2])

        start_idx = np.arange(len(returns))
        end_idx = idx_next
        window_lengths = (end_idx - start_idx + 1).astype(float)

        # Moyenne et écart-type
        sum_returns = cumsum_padded[end_idx + 1] - cumsum_padded[start_idx]
        sum_squares = cumsum2_padded[end_idx + 1] - cumsum2_padded[start_idx]
        mean_returns = sum_returns / window_lengths
        var_returns = (sum_squares / window_lengths) - (mean_returns ** 2)
        std_returns = np.sqrt(np.maximum(var_returns, 1e-12))

        # Sharpe-like ratio
        sharpe_like = mean_returns / std_returns

        # Seuils dynamiques pour classer ~10% haut, 10% bas
        upper_thresh = np.nanpercentile(sharpe_like, 90)
        lower_thresh = np.nanpercentile(sharpe_like, 10)

        # Classification
        result = np.zeros(len(sharpe_like), dtype=int)
        result[sharpe_like > upper_thresh] = 1
        result[sharpe_like < lower_thresh] = -1

        return pd.Series(result, index=df_cleaned.index, name=self.name)
