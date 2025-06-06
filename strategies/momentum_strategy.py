from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import pandas as pd


class MomentumStrategy(Strategy):
    """
    Momentum-based trading strategy that uses:
    - Short and long moving averages
    - Volume indicators
    - Bid-ask spread analysis
    - Portfolio rebalancing
    """

    def __init__(self, short_window=5, long_window=20, volume_threshold=1.5, target_allocation=0.4):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.volume_threshold = volume_threshold
        self.target_allocation = target_allocation  # Target ETH allocation as % of portfolio

    def _calculate_mid_price(self, data):
        """Calculate mid price from bid-ask levels"""
        if 'level-1-bid-price' in data.columns and 'level-1-ask-price' in data.columns:
            return (data['level-1-bid-price'] + data['level-1-ask-price']) / 2
        return None

    def _calculate_moving_averages(self, prices):
        """Calculate short and long moving averages"""
        if len(prices) < self.long_window:
            return None, None
        
        short_ma = prices.rolling(window=self.short_window).mean().iloc[-1]
        long_ma = prices.rolling(window=self.long_window).mean().iloc[-1]
        
        return short_ma, long_ma

    def _analyze_volume(self, data):
        """Analyze volume patterns"""
        if 'V-bid-5-levels' in data.columns and 'V-ask-5-levels' in data.columns:
            total_volume = data['V-bid-5-levels'] + data['V-ask-5-levels']
            if len(total_volume) >= 5:
                recent_avg_volume = total_volume.tail(5).mean()
                historical_avg_volume = total_volume.mean()
                
                if historical_avg_volume > 0:
                    volume_ratio = recent_avg_volume / historical_avg_volume
                    return volume_ratio
        return 1.0

    def _calculate_spread_signal(self, data):
        """Calculate signal based on bid-ask spread"""
        if 'spread' in data.columns and len(data) > 1:
            current_spread = data['spread'].iloc[-1]
            avg_spread = data['spread'].mean()
            
            if avg_spread > 0:
                spread_ratio = current_spread / avg_spread
                # Lower spread might indicate better liquidity/momentum
                return 1 / spread_ratio if spread_ratio > 0 else 1.0
        return 1.0

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate trading action based on momentum indicators
        """
        try:
            # Focus on ETH trading
            if 'ETH' not in data or data['ETH'].empty:
                return {}
            
            eth_data = data['ETH']
            
            # Calculate mid prices
            mid_prices = self._calculate_mid_price(eth_data)
            if mid_prices is None or len(mid_prices) < self.long_window:
                return {}
            
            # Calculate moving averages
            short_ma, long_ma = self._calculate_moving_averages(mid_prices)
            if short_ma is None or long_ma is None:
                return {}
            
            # Analyze volume
            volume_signal = self._analyze_volume(eth_data)
            
            # Calculate spread signal
            spread_signal = self._calculate_spread_signal(eth_data)
            
            # Get current positions
            current_eth = current_portfolio.get_position('ETH')
            current_eurc = current_portfolio.get_position('EURC')
            
            # Estimate current portfolio value
            current_eth_value = current_eth * mid_prices.iloc[-1] if len(mid_prices) > 0 else 0
            total_portfolio_value = current_eth_value + current_eurc
            
            # Calculate target ETH position
            target_eth_value = total_portfolio_value * self.target_allocation
            target_eth_amount = target_eth_value / mid_prices.iloc[-1] if mid_prices.iloc[-1] > 0 else 0
            
            # Generate signals
            momentum_signal = 1 if short_ma > long_ma else -1
            volume_boost = 1 if volume_signal > self.volume_threshold else 0.5
            
            # Combine signals
            combined_signal = momentum_signal * volume_boost * spread_signal
            
            # Calculate position adjustment
            base_adjustment = target_eth_amount - current_eth
            
            # Apply signal strength
            if combined_signal > 1.2:
                # Strong buy signal
                trade_amount = base_adjustment + 0.05 * total_portfolio_value / mid_prices.iloc[-1]
            elif combined_signal < -0.8:
                # Strong sell signal
                trade_amount = base_adjustment - 0.05 * total_portfolio_value / mid_prices.iloc[-1]
            else:
                # Rebalance to target
                trade_amount = base_adjustment * 0.5  # Gradual rebalancing
            
            # Limit trade size to reasonable bounds
            max_trade = 0.1 * total_portfolio_value / mid_prices.iloc[-1]
            trade_amount = max(-max_trade, min(max_trade, trade_amount))
            
            # Only trade if significant enough
            if abs(trade_amount) < 0.001:
                return {}
            
            return {"ETH": 0.1*trade_amount}
            
        except Exception as e:
            print(f"Error in MomentumStrategy.get_action(): {e}")
            return {}