from feature_extraction.base import BaseFeature
import pandas as pd

class LiquidityRatioFeature(BaseFeature):
    """Generate liquidity ratio feature."""
    def __init__(self, n_levels: int = 5):
        super().__init__(
            name=f"liquidity-ratio-{n_levels}-levels",
            description=f"Liquidity ratio: V_bid/V_ask using {n_levels} levels"
        )
        self.n_levels = n_levels
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        v_bid = df_cleaned["level-1-bid-volume"].copy()
        v_ask = df_cleaned["level-1-ask-volume"].copy()
        for i in range(2, self.n_levels + 1):
            v_bid += df_cleaned[f"level-{i}-bid-volume"]
            v_ask += df_cleaned[f"level-{i}-ask-volume"]
        return v_bid / v_ask
