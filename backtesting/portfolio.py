from .types import Coin, MarketData, OrderBookData, FeesGraph
from typing import Dict


def estimate_price(data: OrderBookData, side) -> float:
    """Estimates current price from latest orderbook data using bid-ask level 1"""
    if data.empty:
        raise ValueError("Cannot estimate price from empty data")
    
    # Use level-1 bid and ask prices (best bid/ask)
    if side == 'ask' and 'level-1-ask-price' in data.columns:
        return data['level-1-ask-price'].iloc[-1]
    
    if side == 'bid' and 'level-1-bid-price' in data.columns:
        return data['level-1-bid-price'].iloc[-1]
    
    raise ValueError("Cannot estimate price: no valid price columns found")


def get_bid_ask_spread(data: OrderBookData) -> float:
    """Calculate the bid-ask spread from level 1 data"""
    if data.empty:
        raise ValueError("Cannot calculate spread from empty data")
    
    if 'level-1-bid-price' in data.columns and 'level-1-ask-price' in data.columns:
        latest_bid = data['level-1-bid-price'].iloc[-1]
        latest_ask = data['level-1-ask-price'].iloc[-1]
        return latest_ask - latest_bid
    
    raise ValueError("Cannot calculate spread: no level-1 bid/ask prices found")



def get_fee_for_trade(coin_from: Coin, coin_to: Coin, fees_graph: FeesGraph) -> float:
    """Get the fee rate for trading from one coin to another using the fees graph"""
    if coin_from not in fees_graph:
        raise ValueError(f"No fee information available for trading from {coin_from}")
    
    for target_coin, fee_rate in fees_graph[coin_from]:
        if target_coin == coin_to:
            return fee_rate
    
    raise ValueError(f"No direct trading path from {coin_from} to {coin_to}")


class Portfolio:
    """
    Portfolio class for managing asset positions and executing trades.

    Methods:
        - get_position(coin): Returns the current position for a given coin.
        - can_execute_trade(coin_from, coin_to, amount, fees_graph): Checks if a trade can be executed given current balances and fees.
        - execute_trade(coin_from, coin_to, price, amount, fees_graph, reverse=False): Executes a trade, updating positions and applying fees.
        - get_value(market_data): Returns the total portfolio value in EURC using the latest market data.

    Example usage:
        portfolio = Portfolio(['ETH', 'XBT'], 10000)
        if portfolio.can_execute_trade('EURC', 'ETH', 100, fees_graph):
            portfolio.execute_trade('EURC', 'ETH', 2000, 1, fees_graph)
    """
    
    def __init__(self, coins: list[Coin], initial_amount: float) -> None:
        self.positions: Dict[Coin, float] = {'EURC': initial_amount}  # (euro stable coin)
        """For each coin, self.position[coin] is the amount of coin owned"""
        
        self.coins = coins
        for coin in coins:
            self.positions[coin] = 0.0
    
    def get_position(self, coin: Coin) -> float:
        """Get current position for a specific coin"""
        return self.positions.get(coin, 0.0)
    
    def can_execute_trade(self, coin_from, coin_to, amount, fees_graph):
        """
        Check if the portfolio has enough of coin_from to execute a trade for the specified amount (including fees).

        Args:
            coin_from (str): The coin to be spent.
            coin_to (str): The coin to be acquired.
            amount (float): The amount of coin_from required for the trade (including fees).
            fees_graph (FeesGraph): The fee structure for trades.

        Returns:
            bool: True if the trade can be executed, False otherwise.
        """
        if coin_from not in self.positions:
            return False
        
        fee_rate = get_fee_for_trade(coin_from, coin_to, fees_graph)
        required_amount = amount * (1 + fee_rate)
        return self.positions[coin_from] >= required_amount
 
    def update_position(self, coin: Coin, amount: float) -> None:
        """
        Update the position for a specific coin by adding the specified amount.

        Args:
            coin (str): The coin to update.
            amount (float): The amount to add (can be negative to reduce position).
        """
        if coin not in self.positions:
            raise ValueError(f"Coin {coin} not found in portfolio")
        
        self.positions[coin] += amount
        
        # Ensure no negative positions
        if self.positions[coin] < 0:
            raise ValueError(f"Cannot have negative position for {coin}")
    
    def execute_trade(self, coin_from, coin_to, ratio, volume, fees_graph, reverse=False):
        """
        Execute a trade between two coins at a given price and amount, applying transaction fees as specified in the fees graph.

        Args:
            coin_from (str): The coin being spent (e.g., 'EURC' for a buy, 'ETH' for a sell).
            coin_to (str): The coin being acquired (e.g., 'ETH' for a buy, 'EURC' for a sell).
            price (float): The execution price per unit of coin_to in terms of coin_from (or the opposite if reverse=True).
            amount (float): The amount of coin_to obtained (or the amount of coin_from sold if reverse=True).
            fees_graph (FeesGraph): The fee structure for trades.
            reverse (bool): If True, executes the trade in reverse (used for sell orders).

        Returns:
            bool: True if the trade was executed and positions updated, False otherwise.
        """

        if reverse and (not self.can_execute_trade(coin_from, coin_to, volume , fees_graph)):
            return False
        elif (not reverse) and (not self.can_execute_trade(coin_from, coin_to, volume*ratio, fees_graph)):
            return False
        
        try:
            fee_rate = get_fee_for_trade(coin_from, coin_to, fees_graph)
            
            # Deduct from source coin (including fees)
            if reverse:
                # If reverse is True, we are trading coin_to for coin_from
                total_ratio = ratio * (1 + fee_rate)
                self.update_position(coin_from, -volume)
                self.update_position(coin_to, volume*total_ratio)
            else:
                # Normal trade: coin_from for coin_to
                total_ratio = ratio * (1 + fee_rate)

                self.update_position(coin_from, -total_ratio * volume)
                self.update_position(coin_to, volume)
            
            return True
        
        except ValueError:
            return False
    
    def get_value(self, market_data):
        """
        Calculate the total portfolio value in EURC using the latest market data.

        Args:
            market_data (MarketData): Dictionary of DataFrames for each symbol, containing recent market prices.

        Returns:
            float: The total portfolio value in EURC.
        """
        total_value = self.positions.get('EURC', 0.0)
        
        for coin, amount in self.positions.items():
            if coin != 'EURC' and amount != 0:
                if coin in market_data:
                    price = estimate_price(market_data[coin], "bid")
                    total_value += amount * price
                else:
                    # If no market data available, assume zero value (or could raise error)
                    pass

        return total_value