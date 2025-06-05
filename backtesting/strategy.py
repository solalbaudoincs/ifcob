from .types import MarketData, Coin, Action, FeesGraph
from .portfolio import Portfolio
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Base class for trading strategies.

    All custom strategies should inherit from this class and implement the get_action method.

    Methods:
        - get_action(data, current_portfolio, fees_graph): Returns an action dict for the current timestep.

    Example usage:
        class MyStrategy(Strategy):
            def get_action(self, data, current_portfolio, fees_graph):
                # Implement trading logic here
                return {"ETH": 0.1}
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, data: 'MarketData', current_portfolio: 'Portfolio', fees_graph: 'FeesGraph') -> 'Action':
        """
        Generate a trading action for the current timestep.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
        """
        raise NotImplementedError("get_action must be implemented by subclasses.")
