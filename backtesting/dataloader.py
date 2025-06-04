from .types import OrderBookDataLoader, Coin, Filepath, MarketData, TimeStep
import pandas as pd
import numpy as np


class OrderBookDataFromDf(OrderBookDataLoader):
    
    def __init__(self, sources : list[tuple[Coin,Filepath]]) -> None:
        self.csv_filepaths = sources
        self.dfs = {}
        self.coins = list(coin for coin, _ in sources)
        for coin, filepath in sources:
            df = pd.read_parquet(filepath)
            # Reset index to make timestamp a column and flatten multi-index if present
            df["timestamp"] = df.index.values
            df = df.dropna()
            self.dfs[coin] = df
    
    def get_book_from_range(self, coin: Coin, start_time: TimeStep, end_time: TimeStep) -> pd.DataFrame:
        df = self.dfs[coin]
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()

        if start_time < min_time or end_time > max_time:
            raise ValueError(f"Requested time range [{start_time}, {end_time}] is outside the available data range [{min_time}, {max_time}] for coin {coin}.")
        
        if start_time > end_time:
            raise ValueError(f"Start time {start_time} cannot be after end time {end_time}.")

        # Create a view of the DataFrame for the specified time range
        # Since df is sorted by 'timestamp', use searchsorted for fast slicing
        timesteps = df['timestamp'].values
        start_idx = timesteps.searchsorted(start_time, side='left')
        end_idx = timesteps.searchsorted(end_time, side='right')
        view = df.iloc[start_idx:end_idx]
        return view
    
    def get_books_from_range(self, start_time: TimeStep, end_time: TimeStep) -> MarketData:
        return {
            coin : self.get_book_from_range(coin, start_time, end_time)
            for coin in self.dfs.keys()
        }
    
    def get_coin_at_timestep(self, coin: Coin, time_step: int) -> pd.DataFrame:
        df = self.dfs[coin]
        return df.iloc[[time_step]]
    
    def get_time_step_values(self) -> dict[Coin, np.ndarray]:
        return {
            coin : self.dfs[coin]['timestamp'].values
            for coin in self.dfs.keys()
        }

