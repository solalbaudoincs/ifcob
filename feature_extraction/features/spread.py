from feature_extraction.base import BaseFeature
import pandas as pd

class SpreadFeature(BaseFeature):
    """Generate bid-ask spread feature."""
    def __init__(self):
        super().__init__(
            name="spread",
            description="Bid-ask spread: level-1-ask-price - level-1-bid-price"
        )
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        return df_cleaned["level-1-ask-price"] - df_cleaned["level-1-bid-price"]
