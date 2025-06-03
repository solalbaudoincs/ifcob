from feature_extraction.base import BaseFeature
import pandas as pd
import numpy as np

class InstReturnFeature(BaseFeature):
    """Generate inst-return feature."""
    def __init__(self, time : float = 50):
        super().__init__(
            name=f"inst-return",
            description="inst-return feature, it will probably be used as a target for prediction."
        )
        self.time = time
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        midprice = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        df_cleaned["timestamp"] = df_cleaned.index
        result = midprice.diff()/df_cleaned["timestamp"].diff()
        return pd.Series(result, name=self.name)
