from feature_extraction.base import BaseFeature
import pandas as pd

class MomentumFeature(BaseFeature):
    """Generate momentum feature."""
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"rate-momentum-{window}-sample",
            description=f"Momentum: change in mid-price over {window} periods"
        )
        self.window = window
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.diff(periods=self.window)
