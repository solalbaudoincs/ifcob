from .types import OrderBookData, Strategy

class Backtester:
    
    def __init__(self, dataloaders : list[OrderBookData], strategies : list[Strategy]) -> None:
        self.dataloaders = dataloaders
        self.strategies = strategies
        
        
    def backtest():
        #todo
        pass
    