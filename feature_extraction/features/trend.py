from feature_extraction.base import BaseFeature
import pandas as pd

class TrendFeature(BaseFeature):
    """Generate trend indicator feature."""
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"rate-mid-price-trend-{window}-sample",
            description=f"Trend indicator: rolling mean of mid-price over {window} periods"
        )
        self.window = window
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.rolling(window=self.window).mean()
