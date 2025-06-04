# Strategy Development Documentation

The `strategy.py` module provides the abstract base class for implementing trading strategies in the backtesting framework.

## Strategy Abstract Base Class

### Overview

The `Strategy` class serves as the foundation for all trading strategies. It defines the interface that strategies must implement to work with the backtesting engine.

```python
from abc import ABC, abstractmethod
from .types import MarketData, Action, FeesGraph
from .portfolio import Portfolio

class Strategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Given market data and current portfolio, return the trading action.
        Must be implemented by concrete strategy classes.
        """
        raise NotImplementedError
```

### Abstract Method: get_action()

The core method that all strategies must implement.

**Method Signature:**
```python
def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action
```

**Parameters:**
- `data`: Windowed market data containing recent order book information for all coins
- `current_portfolio`: Current portfolio state with positions and balances
- `fees_graph`: Trading fee structure between coin pairs

**Returns:**
- `Action`: Dictionary mapping coin symbols to trade amounts
  - Positive values = buy amounts
  - Negative values = sell amounts  
  - Zero/missing = hold position

**Execution Context:**
- Called at each timestamp during backtesting
- Execution time is measured and may trigger timeouts
- Market data represents historical window up to current timestamp
- Portfolio reflects state before new trades execute

## Strategy Implementation Guidelines

### Basic Strategy Structure

```python
from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio

class MyStrategy(Strategy):
    def __init__(self, param1=default_value, param2=default_value):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        # Initialize any models, indicators, or state variables
        
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        # Implement strategy logic here
        action = {}
        
        # Example: Process each coin
        for coin in data.keys():
            if coin in current_portfolio.coins:
                # Analyze market data
                trade_amount = self._analyze_coin(data[coin], current_portfolio, coin)
                if abs(trade_amount) > 1e-8:  # Only include non-zero trades
                    action[coin] = trade_amount
        
        return action
    
    def _analyze_coin(self, coin_data, portfolio, coin):
        # Helper method for coin-specific analysis
        # Return positive for buy, negative for sell, 0 for hold
        pass
```

### Data Access Patterns

#### Accessing Order Book Data
```python
def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
    # Check if coin data is available
    if 'ETH' not in data or data['ETH'].empty:
        return {}
    
    eth_data = data['ETH']
    
    # Get latest price information
    latest_row = eth_data.iloc[-1]
    current_bid = latest_row['level-1-bid-price']
    current_ask = latest_row['level-1-ask-price']
    mid_price = (current_bid + current_ask) / 2
    
    # Access historical window
    if len(eth_data) >= 10:
        price_series = (eth_data['level-1-bid-price'] + eth_data['level-1-ask-price']) / 2
        price_change = (price_series.iloc[-1] - price_series.iloc[-10]) / price_series.iloc[-10]
    
    # Use features if available
    if 'spread' in eth_data.columns:
        current_spread = latest_row['spread']
    
    return {}
```

#### Portfolio State Analysis
```python
def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
    # Check current positions
    current_eth = current_portfolio.get_position('ETH')
    current_eurc = current_portfolio.get_position('EURC')
    
    # Calculate portfolio value
    total_value = current_portfolio.get_value(data)
    
    # Calculate position weights
    if total_value > 0:
        eth_weight = (current_eth * estimate_price(data['ETH'])) / total_value
    
    # Target-based rebalancing
    target_eth_weight = 0.5  # 50% ETH allocation
    weight_diff = target_eth_weight - eth_weight
    
    if abs(weight_diff) > 0.05:  # 5% threshold
        target_eth_value = total_value * target_eth_weight
        current_eth_value = current_eth * estimate_price(data['ETH'])
        trade_value = target_eth_value - current_eth_value
        trade_amount = trade_value / estimate_price(data['ETH'])
        return {'ETH': trade_amount}
    
    return {}
```

## Strategy Examples

### 1. Momentum Strategy

```python
class MomentumStrategy(Strategy):
    def __init__(self, lookback_periods=5, momentum_threshold=0.001):
        super().__init__()
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        action = {}
        
        for coin in ['ETH', 'XBT']:
            if coin not in data or len(data[coin]) < self.lookback_periods + 1:
                continue
                
            coin_data = data[coin]
            
            # Calculate momentum
            prices = (coin_data['level-1-bid-price'] + coin_data['level-1-ask-price']) / 2
            momentum = (prices.iloc[-1] - prices.iloc[-self.lookback_periods-1]) / prices.iloc[-self.lookback_periods-1]
            
            # Generate signal
            if momentum > self.momentum_threshold:
                action[coin] = 0.1  # Buy signal
            elif momentum < -self.momentum_threshold:
                action[coin] = -0.1  # Sell signal
        
        return action
```

### 2. Mean Reversion Strategy

```python
class MeanReversionStrategy(Strategy):
    def __init__(self, window=20, z_threshold=2.0):
        super().__init__()
        self.window = window
        self.z_threshold = z_threshold
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        action = {}
        
        for coin in ['ETH', 'XBT']:
            if coin not in data or len(data[coin]) < self.window:
                continue
            
            coin_data = data[coin]
            prices = (coin_data['level-1-bid-price'] + coin_data['level-1-ask-price']) / 2
            
            # Calculate z-score
            mean_price = prices.rolling(self.window).mean().iloc[-1]
            std_price = prices.rolling(self.window).std().iloc[-1]
            current_price = prices.iloc[-1]
            z_score = (current_price - mean_price) / std_price
            
            # Generate contrarian signal
            if z_score > self.z_threshold:
                action[coin] = -0.1  # Price too high, sell
            elif z_score < -self.z_threshold:
                action[coin] = 0.1   # Price too low, buy
        
        return action
```

### 3. Machine Learning Strategy

```python
import joblib
import numpy as np

class MLStrategy(Strategy):
    def __init__(self, model_path, features, target_position=0.5):
        super().__init__()
        self.model = joblib.load(model_path)
        self.features = features
        self.target_position = target_position
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        if 'ETH' not in data or data['ETH'].empty:
            return {}
        
        # Extract features from latest data
        latest_data = data['ETH'].iloc[-1]
        feature_vector = [latest_data[feature] for feature in self.features]
        
        # Get model prediction
        prediction = self.model.predict([feature_vector])[0]
        
        # Convert prediction to trading action
        current_eth = current_portfolio.get_position('ETH')
        
        if prediction == 1:  # Buy signal
            target_eth = self.target_position
        elif prediction == -1:  # Sell signal
            target_eth = 0.0
        else:  # Hold signal
            return {}
        
        trade_amount = target_eth - current_eth
        
        # Only trade if significant difference
        if abs(trade_amount) > 0.01:
            return {'ETH': trade_amount}
        
        return {}
```

## Performance Considerations

### Execution Time Limits
- The backtesting engine measures strategy execution time
- Default timeout is 0.1 seconds per `get_action()` call
- Strategies that exceed timeout may be skipped
- Optimize computation-heavy operations

### Memory Management
- Avoid storing large amounts of historical data in strategy state
- Use windowed data provided by the backtesting engine
- Clean up temporary variables and intermediate calculations

### Error Handling
```python
def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
    try:
        # Strategy logic here
        return self._calculate_action(data, current_portfolio, fees_graph)
    except Exception as e:
        # Log error and return safe default
        print(f"Strategy error: {e}")
        return {}  # Safe default: no action
```

## Testing and Validation

### Strategy Testing Framework
```python
def test_strategy():
    from backtesting.dataloader import OrderBookDataFromDf
    from backtesting.portfolio import Portfolio
    
    # Load test data
    sources = [('ETH', 'test_data/ETH_sample.parquet')]
    dataloader = OrderBookDataFromDf(sources)
    
    # Initialize strategy and portfolio
    strategy = MyStrategy()
    portfolio = Portfolio(['ETH'], 10000.0)
    fees_graph = {'EURC': [('ETH', 0.001)], 'ETH': [('EURC', 0.001)]}
    
    # Test on sample data
    test_data = dataloader.get_books_from_range(start_time, end_time)
    action = strategy.get_action(test_data, portfolio, fees_graph)
    
    print(f"Strategy action: {action}")
    
    # Validate action format
    assert isinstance(action, dict)
    for coin, amount in action.items():
        assert isinstance(coin, str)
        assert isinstance(amount, (int, float))
```

### Common Pitfalls

1. **Look-ahead bias**: Don't use future data in current decisions
2. **Overfitting**: Avoid strategies that work only on specific historical periods
3. **Transaction costs**: Consider fees and market impact in strategy logic
4. **Data quality**: Handle missing or invalid data gracefully
5. **Position limits**: Respect portfolio constraints and available capital

## Integration with Backtesting

Strategies are used in the backtesting framework as follows:

```python
from backtesting.backtest import Backtester, BacktestConfig
from strategies.my_strategy import MyStrategy

# Configure backtesting
config = BacktestConfig(
    initial_capital=10000.0,
    fees_graph=fees_graph,
    symbols=['ETH', 'XBT'],
    calibration_end_time=calibration_end,
    validation_start_time=validation_start
)

# Run backtest with multiple strategies
strategies = [
    MyStrategy(param1=value1),
    MyStrategy(param1=value2),  # Parameter sensitivity testing
]

backtester = Backtester(dataloader, config)
results = backtester.backtest(strategies)
```

The backtesting engine handles:
- Chronological data iteration
- Windowed market data creation
- Strategy execution timing
- Order scheduling and execution
- Performance metric calculation