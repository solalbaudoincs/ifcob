from feature_extraction.base import BaseFeature
import pandas as pd

class BidAskImbalanceFeature(BaseFeature):
    """Generate bid-ask imbalance features."""
    def __init__(self, n_levels: int = 5):
        super().__init__(
            name=f"bid-ask-imbalance-{n_levels}-levels",
            description=f"Bid-ask imbalance using top {n_levels} levels. "
                       "Close to 1 → strong buying pressure, Close to -1 → strong selling pressure, Near 0 → balanced depth."
        )
        self.n_levels = n_levels
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        v_bid = df_cleaned["level-1-bid-volume"].copy()
        v_ask = df_cleaned["level-1-ask-volume"].copy()
        for i in range(2, self.n_levels + 1):
            v_bid += df_cleaned[f"level-{i}-bid-volume"]
            v_ask += df_cleaned[f"level-{i}-ask-volume"]
        return (v_bid - v_ask) / (v_bid + v_ask)
