from .types import MarketData, Symbol
from .portolio import Portfolio
from typing import TypeAlias
from abc import ABC

Action : TypeAlias = dict[Symbol, float]
"""For each symbol, how much to buy/sell"""


class Strategy(ABC):
    
    def __init__(self):
        pass
    
    @classmethod
    def get_action(self, data : MarketData, current_portfolio : Portfolio, fees: float) -> Action:
        raise NotImplementedError