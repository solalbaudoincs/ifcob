# Strategies

This folder contains various trading strategies implemented for backtesting and evaluation. Each strategy is designed to operate on cryptocurrency market data, primarily focusing on ETH trading. Below is a summary of each strategy:

## 1. trend_following.py
- **TFCumulativeReturnStrategy**: Follows a simple trend-following logic based on the cumulative return of ETH. Buys or sells ETH depending on whether the cumulative return exceeds a volatility-based threshold.
- **TFSharpeRatioStrategy**: (See code for details) Similar to the above but uses the Sharpe ratio as the decision metric.

## 2. rf_pred_all_signed_strat_mateo.py
- **RFPredAllSignedStratMateo**: Utilizes a pre-trained Random Forest model to predict trading signals for ETH. The strategy buys, sells, or rebalances ETH based on the model's output signal (-1: sell, 0: rebalance, 1: buy).

## 3. momentum_strategy.py
- **MomentumStrategy**: Implements a momentum-based approach using short and long moving averages, volume analysis, and bid-ask spread. The strategy rebalances the portfolio based on detected momentum signals.

## 4. mateo_2_start.py
- **Mateo2StartStrategy**: Uses precomputed predictions from a Random Forest model (for benchmarking/cheating). Executes trades based on these predictions and manages a queue of buy orders.

## 5. linear_regression.py
- **LinearRegression**: Placeholder for a linear regression-based strategy (not yet implemented).

---

Each strategy inherits from the base `Strategy` class and implements a `get_action` method (or similar) to generate trading actions based on market data and portfolio state. See individual files for implementation details and customization options.
