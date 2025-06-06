from ..base import BaseFeature
import pandas as pd


class MidPriceFeature(BaseFeature):
    """Generate MidPrice feature."""
    
    def __init__(self):
        super().__init__(
            name=f"mid-price",
            description="Custom feature"
        )
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the MidPrice feature.
        
        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments
        
        Returns:
            pd.Series: The generated feature values
        """
        # TODO: Implement feature generation logic here
        # Example:
        # mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        # return mid_price.rolling(window=self.window).std()
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return pd.Series(mid_price, index=df_cleaned.index, name=self.name)
