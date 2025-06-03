from feature_extraction.base import BaseFeature
import pandas as pd

class CumulativeVolumeFeature(BaseFeature):
    """Generate cumulative volume features."""
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        super().__init__(
            name=f"V-{side}-{n_levels}-levels",
            description=f"Cumulative {side} volume across {n_levels} levels"
        )
        self.side = side
        self.n_levels = n_levels
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        volume = df_cleaned[f"level-1-{self.side}-volume"].copy()
        for i in range(2, self.n_levels + 1):
            volume += df_cleaned[f"level-{i}-{self.side}-volume"]
        return volume
