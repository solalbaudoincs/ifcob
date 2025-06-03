from feature_extraction.base import BaseFeature
import pandas as pd

class VWAPFeature(BaseFeature):
    """Generate Volume-Weighted Average Price features."""
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        super().__init__(
            name=f"vwap-{side}-{n_levels}-levels",
            description=f"Volume-Weighted Average Price for {side} side using {n_levels} levels"
        )
        self.side = side
        self.n_levels = n_levels
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        price_cols = [f"level-{i}-{self.side}-price" for i in range(1, self.n_levels + 1)]
        volume_cols = [f"level-{i}-{self.side}-volume" for i in range(1, self.n_levels + 1)]
        prices = df_cleaned[price_cols]
        volumes = df_cleaned[volume_cols]
        return (prices * volumes).sum(axis=1) / volumes.sum(axis=1)
