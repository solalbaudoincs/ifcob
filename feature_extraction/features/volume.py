from feature_extraction.base import BaseFeature
import pandas as pd

class VolumeFeature(BaseFeature):
    """Generate volume features."""
    def __init__(self, side: str, level: int = 1, window: int = 20):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        super().__init__(
            name=f"rate-{side}-volume-level-{level}",
            description=f"Average {side} volume at level {level} over {window} periods"
        )
        self.side = side
        self.level = level
        self.window = window
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        volume_col = f"level-{self.level}-{self.side}-volume"
        return df_cleaned[volume_col].rolling(window=self.window).mean()
