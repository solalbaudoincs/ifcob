from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import TypeAlias

Coin     : TypeAlias = str
"""Symbol for a traded asset (e.g., 'ETH', 'XBT', 'EURC')."""
Filepath : TypeAlias = str
OrderBookData : TypeAlias = pd.DataFrame 
"""A DataFrame with price, volume, and extracted features for a single coin."""

MarketData : TypeAlias = dict[Coin, OrderBookData]
"""Dictionary mapping coin symbols to their respective market data DataFrames."""
TimeStep : TypeAlias = float  # Changed from str to float for numeric timestamps
"""Timestamp for a market data row (float, usually ms since epoch)."""

Action: TypeAlias = dict[Coin, float]
"""Dictionary mapping coin symbols to trade amounts (positive for buy, negative for sell)."""
FeePrice: TypeAlias = float

FeesGraph: TypeAlias = dict[
    Coin,
    list[tuple[Coin, FeePrice]]
]
"""Dictionary mapping coin symbols to a list of (target_coin, fee_rate) tuples."""

class OrderBookDataLoader(ABC):
    """
    Abstract base class for loading order book data for backtesting.

    Methods:
        - get_books_from_range(start_time, end_time): Get market data for all symbols within the specified time range.
        - get_time_step_values(): Get all available timesteps for each coin in the dataset.
        - get_coin_at_timestep(coin, time_step): Get the order book data for a specific coin at a given time step.
        - chronological_iterator(): Generator yielding unique timestamps in chronological order from multiple coin datasets.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_books_from_range(self, start_time: TimeStep, end_time: TimeStep) -> MarketData:
        """
        Get market data for all symbols within the specified time range.

        Args:
            start_time (TimeStep): Start of the time range.
            end_time (TimeStep): End of the time range.
        Returns:
            MarketData: Dictionary mapping coin symbols to DataFrames of market data within the range.
        """
        pass
    
    @abstractmethod
    def get_time_step_values(self) -> dict[Coin, np.ndarray]:
        """
        Get all available timesteps for each coin in the dataset.

        Returns:
            dict[Coin, np.ndarray]: Dictionary mapping each coin to a numpy array of all available timestamps.
        """
        """Get all of the timesteps"""
        pass
    
    @abstractmethod
    def get_coin_at_timestep(self, coin: Coin, time_step: TimeStep) -> pd.DataFrame:
        """
        Get the order book data for a specific coin at a given time step.

        Args:
            coin (Coin): The identifier for the cryptocurrency (e.g., 'BTC', 'ETH').
            time_step (TimeStep): The specific timestamp to retrieve data for.
        Returns:
            pd.DataFrame: DataFrame containing the order book data for the specified coin at the given time step. If no data is available, returns an empty DataFrame.
        """
        pass

    def chronological_iterator(self):
        """
        Returns a generator that yields unique timestamps in chronological order from multiple coin datasets.
        Each yield is a tuple (min_time, coin_indices), where min_time is the next chronological timestamp and coin_indices is a dictionary mapping coin names to their current indices.
        """
        # Get all timestep values for all coins
        all_timesteps = self.get_time_step_values()
        
        # Keep track of current indices for each coin's timestep array
        coin_indices = {}
        for coin, timesteps in all_timesteps.items():
            if len(timesteps) > 0:
                coin_indices[coin] = 0
        
        def indices_generator():
            
            seen = set()
            
            while coin_indices:
            # Find the minimum timestep across all current positions
                min_time = None
                min_coins = set()
                
                for coin, idx in coin_indices.items():
                    timesteps = all_timesteps[coin]
                    if idx < len(timesteps):
                        current_time = timesteps[idx]
                    if min_time is None or current_time < min_time:
                        min_time = current_time
                        min_coins = {coin}
                    elif current_time == min_time:
                      min_coins.add(coin)
                
                if min_time is None:
                    break
                
                # Only yield if we haven't seen this timestep before
                if min_time not in seen:
                    seen.add(min_time)
                    yield min_time, coin_indices
                
                # Advance the index for the coin with minimum timestep
                for coin in min_coins:
                    coin_indices[coin] += 1
                    if coin_indices[coin] >= len(all_timesteps[coin]):
                        del coin_indices[coin]
        
        return indices_generator()

class Strategy(ABC):
    """
    Abstract base class for trading strategies. All custom strategies should implement get_action.
    """
    @abstractmethod
    def get_action(self, data: MarketData, current_portfolio, fees_graph: FeesGraph):
        """
        Generate trading actions based on market data and current portfolio.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.
        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
        """
        pass