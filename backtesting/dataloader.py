from .types import OrderBookDataLoader, Symbol, Filepath
import pandas as pd


class OrderBookDataFromDf(OrderBookDataLoader):
    
    def __init__(self, sources : list[tuple[Symbol,Filepath]]) -> None:
        self.csv_filepaths = sources
        self.dfs = {
            symbol : pd.read_csv(filepath)
            for symbol, filepath in sources
        }
    
    def get_book_from_range(self, symbol : str, start_time : float, end_time : float) -> pd.DataFrame:
        df = self.dfs[symbol]
        min_time = df['timestep'].min()
        max_time = df['timestep'].max()

        if start_time < min_time or end_time > max_time:
            raise ValueError(f"Requested time range [{start_time}, {end_time}] is outside the available data range [{min_time}, {max_time}] for symbol {symbol}.")
        
        if start_time > end_time:
            raise ValueError(f"Start time {start_time} cannot be after end time {end_time}.")

        # Create a view of the DataFrame for the specified time range
        # Since df is sorted by 'timestep', use searchsorted for fast slicing
        timesteps = df['timestep'].values
        start_idx = timesteps.searchsorted(start_time, side='left')
        end_idx = timesteps.searchsorted(end_time, side='right')
        view = df.iloc[start_idx:end_idx]
        return view
    
    def get_books_from_range(self, start_time : float, end_time : float) -> list[BookData]:
        return {
            symbol : self.get_book_from_range(symbol, start_time, end_time)
            for symbol in self.dfs.keys()
        }