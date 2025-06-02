from .types import OrderBookData
import pandas as pd

    
class OrderBookDataFromDf(OrderBookData):
    
    def __init__(self, csv_filepaths : list[str]) -> None:
        self.csv_filepaths = csv_filepaths
        self.dfs = {
            csv_filepaths : pd.read_csv()
        }
    
    def get_book_from_range(self, fp : str, start_time : float, end_time : float):
        return # pour toi Chakib
    
    def get_books_from_range(self, start_time : float, end_time : float):
        return {
            fp : self.get_book_from_range(fp, start_time, end_time)
            for fp in self.dfs.keys()
        }