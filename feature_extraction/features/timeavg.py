from ..base import BaseFeature
import pandas as pd
import numpy as np


class TimeAvgFeature(BaseFeature):
    """Generate TimeAvg feature."""
    
    def __init__(self, base_feature : BaseFeature, time_window : float = 5000):
        super().__init__(
            name=f"avg-{time_window}ms-of-{base_feature.name}",
            description="Custom feature"
        )
        self.time_window = time_window
        self.base_feature = base_feature
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the TimeAvg feature.

        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments

        Returns:
            pd.Series: The generated feature values
        """
        base_generated = self.base_feature.generate(df_cleaned, **kwargs)

        # Use pandas rolling with a time-based window (assumes index is datetime or numeric in ms)
        # By default, pandas.rolling is backward-looking (includes current and previous values)
        if np.issubdtype(df_cleaned.index.dtype, np.datetime64):
            window_str = f"{int(self.time_window)}ms"
            average = base_generated.rolling(window=window_str, min_periods=1).mean()
        else:
            # If index is numeric (e.g., ms), convert to TimedeltaIndex for rolling
            df_temp = base_generated.copy()
            df_temp.index = pd.to_datetime(df_cleaned.index, unit='ms')
            window_str = f"{int(self.time_window)}ms"
            average = df_temp.rolling(window=window_str, min_periods=1).mean()
            average.index = df_cleaned.index  # restore original index

        return pd.Series(average, index=df_cleaned.index, name=self.name)
