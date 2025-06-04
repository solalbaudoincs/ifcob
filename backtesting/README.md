# Backtesting Module

The backtesting module provides a comprehensive framework for testing trading strategies on historical cryptocurrency order book data. It simulates realistic trading conditions including execution delays, market impact, and transaction fees.

## Overview

This module implements a sophisticated backtesting engine that:
- Supports multiple trading strategies running in parallel
- Simulates realistic execution delays and network latency
- Includes market impact modeling based on order book depth
- Handles transaction fees and portfolio management
- Provides comprehensive performance metrics
- Supports time-based validation splits for strategy evaluation

## Architecture

The backtesting system consists of several key components:

```
backtesting/
├── types.py          # Core type definitions and abstract base classes
├── dataloader.py     # Data loading and market data management
├── portfolio.py      # Portfolio management and trade execution
├── strategy.py       # Abstract strategy base class
├── backtest.py       # Main backtesting engine
└── README.md         # This documentation
```

## Key Features

### 1. Realistic Execution Modeling
- **Execution Delays**: Simulates strategy computation time and network latency
- **Market Impact**: Models price impact based on order book depth
- **Asynchronous Orders**: Orders are scheduled for future execution, not immediate

### 2. Portfolio Management
- **Multi-Asset Support**: Handles multiple cryptocurrencies simultaneously
- **Fee Calculation**: Incorporates realistic trading fees
- **Position Tracking**: Maintains accurate portfolio positions over time

### 3. Strategy Framework
- **Abstract Base Class**: Standardized interface for strategy implementation
- **Flexible Actions**: Supports arbitrary buy/sell amounts per asset
- **Market Data Access**: Provides windowed historical data to strategies

### 4. Performance Analytics
- **Comprehensive Metrics**: Total return, Sharpe ratio, maximum drawdown, win rate
- **Split Testing**: Separate calibration and validation periods
- **Runtime Statistics**: Strategy execution time monitoring

## Quick Start

```python
from backtesting.dataloader import OrderBookDataFromDf
from backtesting.backtest import Backtester, BacktestConfig
from strategies.my_strategy import MyStrategy

# Setup data loader
sources = [
    ('ETH', 'data/preprocessed/DATA_0/ETH_EUR.parquet'),
    ('XBT', 'data/preprocessed/DATA_0/XBT_EUR.parquet')
]
dataloader = OrderBookDataFromDf(sources)

# Configure backtesting
config = BacktestConfig(
    initial_capital=10000.0,
    fees_graph={'EURC': [('ETH', 0.001), ('XBT', 0.001)], 
                'ETH': [('EURC', 0.001)], 
                'XBT': [('EURC', 0.001)]},
    symbols=['ETH', 'XBT'],
    calibration_end_time=1609459200000,
    validation_start_time=1609545600000
)

# Run backtest
backtester = Backtester(dataloader, config)
strategies = [MyStrategy()]
results = backtester.backtest(strategies)
```

## Documentation

For detailed documentation of each component:

- [Types and Interfaces](./TYPES.md) - Core type definitions and abstract classes
- [Data Loading](./DATALOADER.md) - Market data management and loading
- [Portfolio Management](./PORTFOLIO.md) - Position tracking and trade execution
- [Strategy Development](./STRATEGY.md) - Creating custom trading strategies
- [Backtesting Engine](./BACKTEST.md) - Main backtesting functionality
- [Performance Metrics](./METRICS.md) - Understanding backtest results

## Requirements

- Python 3.8+
- pandas
- numpy
- joblib (for model loading)
- tqdm (for progress bars)

## Contributing

When adding new features:
1. Follow the existing type annotations
2. Add comprehensive docstrings
3. Update relevant documentation
4. Include error handling for edge cases
5. Test with realistic market data

## License

This module is part of the IFCOB trading research project.