from .types import OrderBookDataLoader, Strategy

class Backtester:
    
    def __init__(self, dataloaders : list[OrderBookDataLoader], strategies : list[Strategy]) -> None:
        self.dataloaders = dataloaders
        self.strategies = strategies
        
        
    def backtest():
        #todo
        pass
    