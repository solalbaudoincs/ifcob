from .types import Coin, MarketData, OrderBookData, FeesGraph
from typing import Dict


def estimate_price(data: OrderBookData) -> float:
    """Estimates current price from latest orderbook data using bid-ask level 1"""
    if data.empty:
        raise ValueError("Cannot estimate price from empty data")
    
    # Use level-1 bid and ask prices (best bid/ask)
    if 'level-1-bid-price' in data.columns and 'level-1-ask-price' in data.columns:
        latest_bid = data['level-1-bid-price'].iloc[-1]
        latest_ask = data['level-1-ask-price'].iloc[-1]
        # Mid-price is the average of best bid and ask
        return (latest_bid + latest_ask) / 2.0
    
    # Fallback: if somehow the standard columns exist
    if 'bid' in data.columns and 'ask' in data.columns:
        latest_bid = data['bid'].iloc[-1]
        latest_ask = data['ask'].iloc[-1]
        return (latest_bid + latest_ask) / 2.0
    
    # Last fallback: if there's a direct price column
    if 'price' in data.columns:
        return float(data['price'].iloc[-1])
    
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


def get_market_impact_estimate(data: OrderBookData, trade_amount: float, side: str) -> float:
    """
    Estimate market impact for a trade given the order book depth.
    This considers the volume available at each level.
    """
    if data.empty:
        raise ValueError("Cannot estimate market impact from empty data")
    
    remaining_amount = abs(trade_amount)
    total_cost = 0.0
    
    if side.lower() == 'buy':
        # For buying, we consume ask levels starting from level 1
        for level in range(1, 11):  # levels 1-10
            price_col = f'level-{level}-ask-price'
            volume_col = f'level-{level}-ask-volume'
            
            if price_col not in data.columns or volume_col not in data.columns:
                break
                
            available_volume = data[volume_col].iloc[-1]
            price = data[price_col].iloc[-1]
            
            if available_volume <= 0:
                continue
                
            consumed_volume = min(remaining_amount, available_volume)
            total_cost += consumed_volume * price
            remaining_amount -= consumed_volume
            
            if remaining_amount <= 0:
                break
    
    elif side.lower() == 'sell':
        # For selling, we consume bid levels starting from level 1
        for level in range(1, 11):  # levels 1-10
            price_col = f'level-{level}-bid-price'
            volume_col = f'level-{level}-bid-volume'
            
            if price_col not in data.columns or volume_col not in data.columns:
                break
                
            available_volume = data[volume_col].iloc[-1]
            price = data[price_col].iloc[-1]
            
            if available_volume <= 0:
                continue
                
            consumed_volume = min(remaining_amount, available_volume)
            total_cost += consumed_volume * price
            remaining_amount -= consumed_volume
            
            if remaining_amount <= 0:
                break
    
    if remaining_amount > 0:
        # Not enough liquidity - could raise warning or use last available price
        pass
    
    executed_amount = abs(trade_amount) - remaining_amount
    return total_cost / executed_amount if executed_amount > 0 else 0.0


def get_fee_for_trade(coin_from: Coin, coin_to: Coin, fees_graph: FeesGraph) -> float:
    """Get the fee rate for trading from one coin to another using the fees graph"""
    if coin_from not in fees_graph:
        raise ValueError(f"No fee information available for trading from {coin_from}")
    
    for target_coin, fee_rate in fees_graph[coin_from]:
        if target_coin == coin_to:
            return fee_rate
    
    raise ValueError(f"No direct trading path from {coin_from} to {coin_to}")


class Portfolio:
    
    def __init__(self, symbols: list[Coin], initial_amount: float) -> None:
        self.positions: Dict[Coin, float] = {'EURC': initial_amount}  # (euro stable coin)
        self.symbols = symbols
        for symbol in symbols:
            self.positions[symbol] = 0.0
    
    def get_value(self, market_data: MarketData) -> float:
        """Calculate total portfolio value in EURC"""
        total_value = self.positions.get('EURC', 0.0)
        
        for coin, amount in self.positions.items():
            if coin != 'EURC' and amount != 0:
                if coin in market_data:
                    price = estimate_price(market_data[coin])
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
 
    
    def execute_trade(self, coin_from: Coin, coin_to: Coin, amount: float, fees_graph: FeesGraph) -> bool:
        """Execute a trade between two coins using fees graph, returns True if successful"""
        if not self.can_execute_trade(coin_from, coin_to, amount, fees_graph):
            return False
        
        try:
            fee_rate = get_fee_for_trade(coin_from, coin_to, fees_graph)
            
            # Deduct from source coin (including fees)
            total_cost = amount * (1 + fee_rate)

            self.update_position(coin_from, -total_cost)
            self.update_position(coin_to, amount)

            return True
        
        except ValueError:
            return False