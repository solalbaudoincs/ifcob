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
    
    def __init__(self, coins: list[Coin], initial_amount: float) -> None:
        self.positions: Dict[Coin, float] = {'EURC': initial_amount}  # (euro stable coin)
        """For each coin, self.position[coin] is the amount of coin owned"""
        
        self.coins = coins
        for coin in coins:
            self.positions[coin] = 0.0
    
    def get_value(self, market_data: MarketData) -> float:
        """Calculate total portfolio value in EURC"""
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
    
    def update_position(self, coin: Coin, amount: float) -> None:
        """Update position for a specific coin"""
        if coin in self.positions:
            self.positions[coin] += amount
        else:
            self.positions[coin] = amount
    
    def get_position(self, coin: Coin) -> float:
        """Get current position for a specific coin"""
        return self.positions.get(coin, 0.0)
    
    def can_execute_trade(self, coin_from: Coin, coin_to: Coin, amount: float, fees_graph: FeesGraph) -> bool:
        """Check if a trade can be executed given current positions and fees graph"""
        if coin_from not in self.positions:
            return False
        
        fee_rate = get_fee_for_trade(coin_from, coin_to, fees_graph)
        required_amount = amount * (1 + fee_rate)
        return self.positions[coin_from] >= required_amount
 
    
    def execute_trade(self, coin_from: Coin, coin_to: Coin, ratio : float, volume : float, fees_graph: FeesGraph, reverse = False) -> bool:
        """Execute a trade between two coins using fees graph, returns True if successful
            ratio is the price ratio coin_from/coin_to by default, volume is the amount of coin_to to obtained
            Thoses can be reversed by setting reverse=True, in which case ration is coin_to/coin_from and volume is the amount of coin_from sold
        """

        #print(f"Executing trade: {coin_from} -> {coin_to}, ratio: {ratio}, volume: {volume}, reverse: {reverse} ")

        if reverse and (not self.can_execute_trade(coin_from, coin_to, volume , fees_graph)):
            return False
        elif (not reverse) and (not self.can_execute_trade(coin_from, coin_to, volume*ratio, fees_graph)):
            return False
        
        #print(f"positions before trade: {self.positions}")
        
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
            
            #print(f"positions after trade: {self.positions}")

            return True
        
        except ValueError:
            return False