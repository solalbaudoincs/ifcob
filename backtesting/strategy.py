from .types import MarketData, Coin, Action, FeesGraph
from .portfolio import Portfolio
from abc import ABC, abstractmethod


class Strategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Given market data and current portfolio, return the trading action.
        Must be implemented by concrete strategy classes.
        """
        raise NotImplementedError
