from .types import Symbol, MarketData, OrderBookData


def estimate_price(data : OrderBookData)-> float:
    """Estimates current price from latest data"""
    raise NotImplementedError




class Portfolio:
    
    def __init__(self, symbols : list[Symbol], initial_amount: float) -> None:
        self.positions = {'cash' : initial_amount}
        self.symbols = symbols
        for symbol in symbols:
            self.positions[symbol] = 0.
    
    def get_value(lastData : MarketData) -> float:
        return sum(
            estimate_price(data) for data in MarketData
        )