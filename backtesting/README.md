# Backtesting Module

This module provides a flexible, event-driven engine for backtesting trading strategies on historical cryptocurrency order-book data. It is designed for modularity, extensibility, and realistic simulation of trading conditions, including transaction costs and portfolio management.

## Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [File Descriptions](#file-descriptions)
- [Class & API Reference](#class--api-reference)
- [Usage Example](#usage-example)
- [Extending the Module](#extending-the-module)

---

## Overview
The backtesting module allows you to:
- Simulate trading strategies using historical order-book data
- Model realistic transaction costs and portfolio constraints
- Evaluate performance metrics such as Sharpe ratio, drawdown, and win rate
- Easily implement and test custom strategies

## Key Components
- **Backtest Engine** (`backtest.py`): Orchestrates the simulation, manages time, and records results. Provides `BacktestResult` and `BacktestConfig` dataclasses for results/configuration.
- **Strategy** (`strategy.py`): Abstract base class for user-defined trading logic. Users subclass `Strategy` and implement `get_action`.
- **Portfolio** (`portfolio.py`): Tracks asset balances, executes trades, computes portfolio value, and manages trade validation and fee application.
- **Order Processor** (`order_processor.py`): Handles order execution, price estimation, and fee application. Provides methods for processing buy/sell orders at specific timestamps.
- **Data Loader** (`dataloader.py`): Loads and slices historical order-book data for simulation. Implements `OrderBookDataFromDf` for loading from Parquet files.
- **Types** (`types.py`): Type aliases (e.g., `Coin`, `MarketData`, `Action`) and abstract base classes for data loaders and strategies. Ensures strong typing and extensibility.
- **`__init__.py`**: Marks the folder as a Python package.

## File Descriptions
- **`backtest.py`**: Main simulation loop, result/configuration dataclasses, and core backtesting logic. Handles metrics like Sharpe ratio, drawdown, win rate, and transaction costs. Supports configuration of initial capital, fee structure, window size, and more.
- **`strategy.py`**: Defines the `Strategy` abstract base class. All custom strategies must implement `get_action(data, current_portfolio, fees_graph)` to return trading actions for each timestep.
- **`portfolio.py`**: Implements the `Portfolio` class for managing asset positions, executing trades, and calculating portfolio value. Includes utility functions for price estimation and fee calculation.
- **`order_processor.py`**: Contains the `OrderProcessor` class for executing buy/sell orders, applying transaction costs, and updating the portfolio based on market data at execution time.
- **`dataloader.py`**: Implements `OrderBookDataFromDf`, a loader for Parquet order-book data. Provides time-sliced access and efficient range queries for simulation.
- **`types.py`**: Provides type aliases and abstract base classes for order book data loaders and strategies. Ensures type safety and extensibility.
- **`__init__.py`**: (Empty) Marks the folder as a Python package.

## Class & API Reference

### Backtest Engine (`backtest.py`)
- **BacktestResult**: Stores portfolio values, trades, returns, Sharpe ratio, drawdown, win rate, transaction costs, and more.
- **BacktestConfig**: Configures initial capital, fee structure, traded symbols, window size, calibration/validation splits, and timing constraints.
- **run_backtest** (not shown): Main function to run a backtest simulation.

### Strategy (`strategy.py`)
- **Strategy (ABC)**: Abstract base class. Implement `get_action(data, current_portfolio, fees_graph)` to return an `Action` dict (e.g., `{ 'ETH': 0.1 }`).

### Portfolio (`portfolio.py`)
- **Portfolio**: Manages asset balances, trade execution, and value calculation. Methods:
  - `get_position(coin)`: Get current balance for a coin.
  - `can_execute_trade(coin_from, coin_to, amount, fees_graph)`: Check if a trade is possible.
  - `execute_trade(coin_from, coin_to, price, amount, fees_graph, reverse=False)`: Execute a trade and update balances.
  - `get_value(market_data)`: Compute total portfolio value in EURC.
- **estimate_price**: Utility to estimate price from order book data.
- **get_fee_for_trade**: Utility to get fee rate for a trade.

### Order Processor (`order_processor.py`)
- **OrderProcessor**: Handles order execution at specific timestamps. Methods:
  - `process_order(coin, amount, action_type, execution_timestamp, portfolio)`: Execute a buy/sell order and update the portfolio.

### Data Loader (`dataloader.py`)
- **OrderBookDataFromDf**: Loads order book data from Parquet files. Methods:
  - `get_book_from_range(coin, start_time, end_time)`: Get data for a coin in a time range.
  - `get_books_from_range(start_time, end_time)`: Get all coins' data in a time range.
  - `get_coin_at_timestep(coin, time_step)`: Get data for a coin at a specific timestep.
  - `get_time_step_values()`: Get all available timestamps for each coin.

### Types (`types.py`)
- **Type Aliases**: `Coin`, `Filepath`, `OrderBookData`, `MarketData`, `TimeStep`, `Action`, `FeesGraph`.
- **OrderBookDataLoader (ABC)**: Abstract base for data loaders.
- **Strategy (ABC)**: Abstract base for strategies.

## Usage Example
```python
from backtesting import BacktestConfig, BacktestResult, Strategy, Portfolio, OrderProcessor, OrderBookDataFromDf

# 1. Load data
sources = [('XBT', 'data/features/DATA_1/XBT_EUR.parquet'), ('ETH', 'data/features/DATA_1/ETH_EUR.parquet')]
dataloader = OrderBookDataFromDf(sources)

# 2. Define your strategy
class MyStrategy(Strategy):
    def get_action(self, data, current_portfolio, fees_graph):
        # Example: always hold
        return {}

# 3. Set up portfolio and config
portfolio = Portfolio(['ETH', 'XBT'], 10000)
config = BacktestConfig(
    initial_capital=10000,
    fees_graph={'EURC': [('ETH', 0.001), ('XBT', 0.001)], 'ETH': [('EURC', 0.001)], 'XBT': [('EURC', 0.001)]},
    symbols=['ETH', 'XBT']
)

# 4. Run backtest (see backtest.py for details)
# results = run_backtest(config, dataloader, MyStrategy)
```

## Extending the Module
- **Add new strategies**: Subclass `Strategy` and implement `get_action`.
- **Custom data sources**: Implement a new `OrderBookDataLoader`.
- **Advanced portfolio logic**: Extend the `Portfolio` class.

See the code and docstrings in each file for further details and advanced usage.
