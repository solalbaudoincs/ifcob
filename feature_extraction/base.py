from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    """Base class for all feature generators."""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    @abstractmethod
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate the feature from cleaned data."""
        pass
