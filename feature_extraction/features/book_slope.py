from feature_extraction.base import BaseFeature
import pandas as pd

class BookSlopeFeature(BaseFeature):
    """Generate book slope features for bid and ask sides."""
    def __init__(self, side: str, n_levels: int = 5):
        if side not in ['bid', 'ask']:
            raise ValueError("side must be 'bid' or 'ask'")
        super().__init__(
            name=f"slope-{side}-{n_levels}-levels",
            description=f"Book slope for {side} side using {n_levels} levels. "
                       "Measures steepness of liquidity: (P_N - P_1) / V_sum"
        )
        self.side = side
        self.n_levels = n_levels
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        price_cols = [f"level-{i}-{self.side}-price" for i in range(1, self.n_levels + 1)]
        volume_cols = [f"level-{i}-{self.side}-volume" for i in range(1, self.n_levels + 1)]
        p_n = df_cleaned[price_cols[-1]]
        p_1 = df_cleaned[price_cols[0]]
        v_sum = sum([df_cleaned[col] for col in volume_cols])
        return (p_n - p_1) / v_sum
