from abc import ABC
import pandas as pd
from typing import TypeAlias

Symbol   : TypeAlias = str
Filepath : TypeAlias = str
BookData : TypeAlias = pd.DataFrame 
""" A df w/ 41 columns"""
class OrderBookDataLoader(ABC):
    
    def __init__(self):
        pass
    
    def get_books_from_range(start_time : float, end_time : float) -> dict[Symbol, BookData]:
        raise NotImplementedError
    
class Strategy(ABC):
    # todo
    pass