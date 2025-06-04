from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import TypeAlias

Coin     : TypeAlias = str
"""Name of cryptocurency"""
Filepath : TypeAlias = str
OrderBookData : TypeAlias = pd.DataFrame 
"""A df w/ price /volume and extracted features"""

MarketData : TypeAlias = dict[Coin, OrderBookData]
TimeStep : TypeAlias = float  # Changed from str to float for numeric timestamps

# Move these type definitions here to avoid circular imports
Action: TypeAlias = dict[Coin, float]
"""For each symbol, how much to buy/sell"""
FeePrice: TypeAlias = float

FeesGraph: TypeAlias = dict[
    Coin,
    list[tuple[Coin, FeePrice]]
]
"""oriented graph of transaction from coin to coin"""

class OrderBookDataLoader(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_books_from_range(self, start_time: TimeStep, end_time: TimeStep) -> MarketData:
        """Get market data for all symbols within the specified time range"""
        pass
    
    @abstractmethod
    def get_time_step_values(self) -> dict[Coin, np.ndarray]:
        """
        Get all available timesteps for each coin in the dataset.
        
        Returns a dictionary mapping each coin to a numpy array containing
        all the timestamp values available in the order book data for that coin.
        The timestamps are typically sorted in ascending order.
        
        Returns:
            dict[Coin, np.ndarray]: A dictionary where keys are coin identifiers
                                   and values are numpy arrays of timestamp values
                                   representing all available time steps for each coin.
        
        Example:
            >>> loader = OrderBookDataFromDf(sources)
            >>> timesteps = loader.get_time_step_values()
            >>> timesteps['BTC']  # Returns array of all BTC timestamps
            array([1609459200000, 1609459260000, 1609459320000, ...])
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
            pd.DataFrame: A DataFrame containing the order book data for the specified coin
                          at the given time step. If no data is available, returns an empty DataFrame.
        
        Example:
            >>> loader = OrderBookDataFromDf(sources)
            >>> btc_data = loader.get_coin_at_timestep('BTC', 1609459200000)
            >>> print(btc_data.head())
        """
        pass

    def chronological_iterator(self):
        """
            Returns a generator that yields unique timestamps in chronological order from multiple coin datasets.
            This function iterates through timesteps from multiple coins and yields each unique
            timestamp exactly once, along with the current indices for all coins. It finds the
            next minimum timestamp across all coins and advances the corresponding indices.
            Yields:
                tuple: A tuple containing:
                    - min_time: The next chronological timestamp across all coins
                    - coin_indices: Dictionary mapping coin names to their current indices
            Notes:
                - Ensures each timestamp is yielded only once using a seen set
                - Automatically removes coins that have exhausted their timesteps
                - Maintains chronological order across all coin datasets
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
    
    @abstractmethod
    def get_action(self, data: MarketData, current_portfolio, fees: TimeStep):
        """Generate trading actions based on market data and current portfolio"""
        pass