# Backtesting Engine Documentation

The `backtest.py` module contains the core backtesting engine that orchestrates strategy testing with realistic execution modeling.

## Core Classes

### BacktestResult

A dataclass containing the results of a backtest run.

```python
@dataclass
class BacktestResult:
    portfolio_values: List[float]      # Portfolio value at each timestamp
    timestamps: List[float]            # Corresponding timestamps
    trades: List[Dict]                # All executed trades with metadata
    total_return: float               # Total return over the period
    sharpe_ratio: float               # Risk-adjusted return metric
    max_drawdown: float               # Maximum peak-to-trough decline
    win_rate: float                   # Percentage of profitable trades
    transaction_costs: float          # Total fees paid
    final_portfolio_value: float      # Final portfolio value
```

### BacktestConfig

Configuration object for backtesting parameters.

```python
@dataclass
class BacktestConfig:
    initial_capital: float                        # Starting capital in EURC
    fees_graph: FeesGraph                        # Trading fee structure
    symbols: List[Coin]                          # Coins to trade
    window_size: int = 10                        # Data window size
    
    # Time splits for validation
    calibration_end_time: Optional[TimeStep] = None
    validation_start_time: Optional[TimeStep] = None
    
    # Performance settings
    max_action_runtime: float = 0.1              # Strategy timeout (seconds)
    skip_on_timeout: bool = True                 # Skip slow strategies
    network_latency_delta: float = 0.05          # Execution delay (seconds)
```

## OrderProcessor

Handles realistic order execution with market impact and timing delays.

### Constructor
```python
def __init__(self, fees_graph: FeesGraph, dataloader: OrderBookDataLoader)
```

### Key Methods

#### `process_order(coin, amount, action_type, execution_timestamp, portfolio)`

Executes individual orders at their scheduled execution time.

**Features:**
- Gets market data at execution time (not decision time)
- Applies market impact based on order book depth
- Updates portfolio positions atomically
- Returns trade metadata including fees and delays

**Process:**
1. Retrieve market data at execution timestamp
2. Calculate effective price with market impact
3. Validate portfolio has sufficient funds
4. Execute trade and update positions
5. Return trade details with timing metadata

## Backtester Class

The main backtesting engine that coordinates all components.

### Constructor
```python
def __init__(self, dataloader: OrderBookDataLoader, config: BacktestConfig)
```

### Core Method: backtest()

```python
def backtest(self, strategies: List[Strategy]) -> Dict[str, Tuple[BacktestResult, BacktestResult]]
```

Executes backtesting for multiple strategies with calibration/validation splits.

**Parameters:**
- `strategies`: List of strategy instances to test

**Returns:**
- Dictionary mapping strategy names to (calibration_result, validation_result) tuples

## Execution Model

### Realistic Timing Simulation

The backtesting engine models realistic trading conditions:

1. **Decision Time**: Strategy receives market data and makes decision
2. **Computation Time**: Strategy execution time is measured
3. **Network Latency**: Additional delay added (configurable)
4. **Execution Time**: Order processed with market data at execution time

```
Timeline:
Decision_Time ----[Strategy Runtime]----[Network Latency]----Execution_Time
     |                                                              |
Market data provided                                     Order executed
Portfolio unchanged                                      Portfolio updated
```

### Order Scheduling and Execution

Orders are not executed immediately but scheduled for future execution:

```python
execution_timestamp = decision_timestamp + strategy_runtime + network_latency_delta
```

**Key Features:**
- Portfolio updates occur at execution time, not decision time
- Market conditions at execution may differ from decision time
- Accounts for realistic delays in automated trading systems
- Handles partial fills and insufficient liquidity

### Multi-Strategy Parallel Testing

All strategies are tested in parallel on the same data:
- Each strategy maintains separate portfolio state
- Identical market conditions for fair comparison
- Independent execution timing and delays
- Separate calibration and validation results

## Backtesting Workflow

### 1. Initialization Phase
```python
# Setup data and configuration
dataloader = OrderBookDataFromDf(sources)
config = BacktestConfig(
    initial_capital=10000.0,
    fees_graph=fees_graph,
    symbols=['ETH', 'XBT'],
    calibration_end_time=cal_end,
    validation_start_time=val_start
)

# Initialize strategies
strategies = [MomentumStrategy(), MeanReversionStrategy()]
backtester = Backtester(dataloader, config)
```

### 2. Chronological Iteration
```python
# For each timestamp in chronological order:
for timestamp, coin_indices in dataloader.chronological_iterator():
    # 1. Process pending orders (from previous decisions)
    process_pending_orders(timestamp)
    
    # 2. Create windowed market data
    market_data = create_windowed_data(timestamp, coin_indices)
    
    # 3. Execute strategies in parallel
    for strategy in strategies:
        # Measure execution time
        start_time = time.perf_counter()
        actions = strategy.get_action(market_data, portfolio, fees_graph)
        runtime = time.perf_counter() - start_time
        
        # Schedule orders for future execution
        execution_time = timestamp + runtime + network_latency
        schedule_orders(actions, execution_time)
        
        # Record portfolio value at decision time
        record_portfolio_value(timestamp, portfolio.get_value(market_data))
```

### 3. Order Processing
```python
# When orders reach execution time:
def process_pending_orders(current_timestamp):
    for order in pending_orders:
        if order.execution_timestamp <= current_timestamp:
            # Get market data at execution time
            execution_market_data = get_market_data_at(order.execution_timestamp)
            
            # Execute with market impact
            trade_result = order_processor.process_order(
                order.coin, order.amount, order.action_type,
                order.execution_timestamp, portfolio
            )
            
            # Record trade details
            if trade_result:
                trades.append(trade_result)
```

## Performance Metrics

### Calculated Metrics

#### Total Return
```python
total_return = (final_value - initial_value) / initial_value
```

#### Sharpe Ratio
```python
returns = np.diff(portfolio_values) / portfolio_values[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns)
```

#### Maximum Drawdown
```python
peak = portfolio_values[0]
max_drawdown = 0
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    max_drawdown = max(max_drawdown, drawdown)
```

#### Win Rate
```python
winning_trades = sum(1 for trade in trades if is_profitable(trade))
total_trades = len([trade for trade in trades if trade['action'] == 'sell'])
win_rate = winning_trades / total_trades if total_trades > 0 else 0
```

### Runtime Statistics

The backtester tracks strategy performance:
- Total execution time per strategy
- Number of action calls
- Timeout occurrences
- Average runtime per call

## Advanced Features

### Windowed Market Data

Strategies receive windowed historical data:
```python
def _create_windowed_market_data(self, current_timestep, coin_indices, window_size):
    windowed_data = {}
    for coin in self.config.symbols:
        current_idx = coin_indices[coin]
        start_idx = max(0, current_idx - window_size + 1)
        end_idx = current_idx + 1
        windowed_data[coin] = self.dataloader.dfs[coin].iloc[start_idx:end_idx]
    return windowed_data
```

### Error Handling and Resilience
- Graceful handling of strategy errors
- Timeout management for slow strategies
- Missing data handling
- Trade validation and rollback

### Memory Management
- Efficient data slicing using pandas views
- Minimal data copying during iteration
- Cleanup of processed orders
- Optimized timestamp indexing

## Usage Examples

### Basic Backtesting Setup

```python
from backtesting.dataloader import OrderBookDataFromDf
from backtesting.backtest import Backtester, BacktestConfig
from strategies.momentum_strategy import MomentumStrategy

# Load data
sources = [
    ('ETH', 'data/preprocessed/DATA_0/ETH_EUR.parquet'),
    ('XBT', 'data/preprocessed/DATA_0/XBT_EUR.parquet')
]
dataloader = OrderBookDataFromDf(sources)

# Configure backtesting
fees_graph = {
    'EURC': [('ETH', 0.001), ('XBT', 0.001)],
    'ETH': [('EURC', 0.001)],
    'XBT': [('EURC', 0.001)]
}

config = BacktestConfig(
    initial_capital=10000.0,
    fees_graph=fees_graph,
    symbols=['ETH', 'XBT'],
    window_size=20,
    calibration_end_time=1609459200000,
    validation_start_time=1609545600000,
    max_action_runtime=0.1,
    network_latency_delta=0.05
)

# Run backtest
strategies = [MomentumStrategy(lookback=5), MomentumStrategy(lookback=10)]
backtester = Backtester(dataloader, config)
results = backtester.backtest(strategies)

# Analyze results
for strategy_name, (cal_result, val_result) in results.items():
    print(f"\n{strategy_name}:")
    print(f"Calibration - Return: {cal_result.total_return:.3f}, Sharpe: {cal_result.sharpe_ratio:.3f}")
    print(f"Validation - Return: {val_result.total_return:.3f}, Sharpe: {val_result.sharpe_ratio:.3f}")
```

### Advanced Configuration

```python
# High-frequency trading simulation
config = BacktestConfig(
    initial_capital=100000.0,
    fees_graph=fees_graph,
    symbols=['ETH', 'XBT', 'LTC'],
    window_size=5,  # Small window for HFT
    max_action_runtime=0.01,  # Strict timing requirements
    skip_on_timeout=True,
    network_latency_delta=0.001,  # Low latency setup
    calibration_end_time=cal_end,
    validation_start_time=val_start
)

# Multiple strategy comparison
strategies = [
    MomentumStrategy(lookback=3, threshold=0.0005),
    MeanReversionStrategy(window=10, z_threshold=1.5),
    MLStrategy('models/rf_model.joblib', features_list),
]

results = backtester.backtest(strategies)
```

### Results Analysis

```python
import matplotlib.pyplot as plt

def analyze_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for strategy_name, (cal_result, val_result) in results.items():
        # Plot portfolio value evolution
        axes[0,0].plot(cal_result.timestamps, cal_result.portfolio_values, 
                      label=f"{strategy_name} (Cal)")
        axes[0,1].plot(val_result.timestamps, val_result.portfolio_values,
                      label=f"{strategy_name} (Val)")
        
        # Analyze trade patterns
        trade_returns = [trade.get('return', 0) for trade in val_result.trades]
        axes[1,0].hist(trade_returns, alpha=0.5, label=strategy_name)
        
        # Performance metrics
        metrics = {
            'Return': val_result.total_return,
            'Sharpe': val_result.sharpe_ratio,
            'Max DD': val_result.max_drawdown,
            'Win Rate': val_result.win_rate
        }
        axes[1,1].bar(range(len(metrics)), list(metrics.values()), 
                     alpha=0.7, label=strategy_name)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
```

## Performance Considerations

### Optimization Tips
- Use efficient data structures for large datasets
- Minimize strategy computation time
- Batch process orders when possible
- Consider parallel processing for independent strategies

### Scalability
- Memory usage scales with data size and window size
- Execution time depends on strategy complexity and data frequency
- Consider data sampling for initial strategy development
- Use profiling tools to identify bottlenecks