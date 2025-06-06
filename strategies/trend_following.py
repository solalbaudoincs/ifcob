from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import joblib
import pandas as pd
import time


class TFCumulativeReturnStrategy(Strategy):
    """
    Trend Following Strategy for ETH Trading.

    This naive strategy doesn't use a prediction model but instead follows a simple trend-following logic:
        - if the cumulative return of ETH is above a certain threshold, it buys 0.1 ETH.
        - if the cumulative return is below a certain threshold, it sells 0.1 ETH.
    The threshold is determined by the volatility of the return of ETH.
    This strategy serves as a baseline for benchmark against more complex models.

    Args:
        window_size (int): The window size (in ms) for the model features.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate a trading action for ETH based on the latest XBT features and the RF model prediction.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
                - {"ETH": -0.1} for sell
                - {"ETH": 0.1} for buy
                - {"ETH": 0} for hold
        """
        
        cumulative_return = data["ETH"][f"return-vs-volatility-{self.window_size}-ms"].iloc[-1].values[0]
        
        return {"ETH": 0.1 * cumulative_return}


class TFSharpeRatioStrategy(Strategy):
    """
    Trend Following Strategy for ETH Trading.

    This naive strategy doesn't use a prediction model but instead follows a simple trend-following logic:
        - if the Sharpe ratio of the return of ETH is above a certain threshold, it buys 0.1 ETH.
        - if the Sharpe ratio is below a certain threshold, it sells 0.1 ETH.
    The threshold is determined by the volatility of the return of ETH.
    This strategy serves as a baseline for benchmark against more complex models.

    Args:
        window_size (int): The window size (in ms) for the model features.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate a trading action for ETH based on the latest XBT features and the RF model prediction.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
                - {"ETH": -0.1} for sell
                - {"ETH": 0.1} for buy
                - {"ETH": 0} for hold
        """
        
        sharpe_ratio = data["ETH"].iloc[-1][[f"sharpe-ratio-quantile-calibrated-{self.window_size}-ms"]].values[0]
        
        return {"ETH": 0.1 * sharpe_ratio}

class TFImbalanceStrategy(Strategy):
    """
    Trend Following Strategy for ETH Trading.

    This naive strategy doesn't use a prediction model but instead follows a simple trend-following logic:
        - if the Sharpe ratio of the return of ETH is above a certain threshold, it buys 0.1 ETH.
        - if the Sharpe ratio is below a certain threshold, it sells 0.1 ETH.
    The threshold is determined by the volatility of the return of ETH.
    This strategy serves as a baseline for benchmark against more complex models.

    Args:
        window_size (int): The window size (in ms) for the model features.
    """

    def __init__(self, levels=5):
        super().__init__()
        self.levels = levels

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate a trading action for ETH based on the latest XBT features and the RF model prediction.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
                - {"ETH": -0.1} for sell
                - {"ETH": 0.1} for buy
                - {"ETH": 0} for hold
        """
        
        imbalance = data["ETH"].iloc[-1][[f"bid-ask-imbalance-{self.levels}-levels"]].values[0]
        
        if imbalance > 0.7:
            return {"ETH": 0.1}
        if imbalance < 0.3:
            return {"ETH": -0.1}
        return {"ETH": 0.0}