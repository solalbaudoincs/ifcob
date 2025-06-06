# Backtesting Module

This module provides a flexible and realistic backtesting framework for high-frequency crypto trading strategies. It supports asynchronous order execution, realistic latency, transaction cost modeling, and detailed performance metrics.

## Key Components

### Portfolio
- Manages asset positions and executes trades.
- Applies transaction fees and updates balances for all supported coins.
- Example usage:

```python
portfolio = Portfolio(['ETH', 'XBT'], 10000)
if portfolio.can_execute_trade('EURC', 'ETH', 100, fees_graph):
    portfolio.execute_trade('EURC', 'ETH', 2000, 1, fees_graph)
```

### OrderProcessor
- Handles the execution of buy and sell orders using market data and the fee structure.
- Ensures trades are executed at the correct simulated time and updates the portfolio accordingly.
- Example usage:

```python
processor = OrderProcessor(fees_graph, dataloader)
trade = processor.process_order('ETH', 1.0, 'buy', 1234567890, portfolio)
```

### Backtester
- Runs and evaluates trading strategies on historical data.
- Supports calibration/validation splits, realistic delays, and detailed reporting.
- Example usage:

```python
config = BacktestConfig(
    initial_capital=1e6,
    fees_graph=fees_graph,
    symbols=['XBT', 'ETH'],
    window_size=10,
    calibration_end_time=split_timestamp,
    validation_start_time=split_timestamp
)
backtester = Backtester(dataloader, config)
results = backtester.backtest([MyStrategy()])
```

## Type Aliases
- `MarketData`: `Dict[str, pd.DataFrame]` — Maps coin symbols to their market data.
- `Action`: `Dict[str, float]` — Maps coin symbols to trade amounts (positive for buy, negative for sell).
- `FeesGraph`: `Dict[str, List[Tuple[str, float]]]` — Maps coin symbols to a list of (target_coin, fee_rate) tuples.
- `Coin`: `str` — Symbol for a traded asset (e.g., 'ETH', 'XBT', 'EURC').
- `TimeStep`: `float` — Timestamp for a market data row.

## Strategy Interface
- All strategies must inherit from `Strategy` and implement `get_action(data, current_portfolio, fees_graph)`.
- Example:

```python
class MyStrategy(Strategy):
    def get_action(self, data, current_portfolio, fees_graph):
        # Implement trading logic here
        return {"ETH": 0.1}
```

## Documentation
- See `PORTFOLIO.md`, `ORDER_PROCESSOR.md`, `BACKTEST.md`, and `TYPES.md` for detailed API documentation and further examples.

## Notes
- All trade amounts are interpreted as the amount of the asset to buy (positive) or sell (negative).
- The portfolio is updated only if the trade passes the `can_execute_trade` check.
- Fees are always applied as specified in the `fees_graph`.