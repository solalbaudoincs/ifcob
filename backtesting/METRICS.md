# Performance Metrics Documentation

This document explains the performance metrics calculated by the backtesting framework and how to interpret them.

## Core Performance Metrics

### Total Return

**Definition:** The percentage change in portfolio value from start to end of the period.

**Formula:**
```python
total_return = (final_portfolio_value - initial_capital) / initial_capital
```

**Interpretation:**
- `0.15` = 15% gain over the period
- `-0.05` = 5% loss over the period
- Measures absolute performance without risk adjustment

**Example:**
```python
# Starting with 10,000 EURC, ending with 11,500 EURC
total_return = (11500 - 10000) / 10000 = 0.15  # 15% return
```

### Sharpe Ratio

**Definition:** Risk-adjusted return metric that measures excess return per unit of risk.

**Formula:**
```python
returns = np.diff(portfolio_values) / portfolio_values[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns)
```

**Interpretation:**
- Higher values indicate better risk-adjusted performance
- `> 1.0`: Good performance
- `> 2.0`: Excellent performance
- `< 0`: Strategy loses money on average

**Notes:**
- Calculated using portfolio value changes at each timestamp
- Assumes zero risk-free rate for simplicity
- Annualized Sharpe can be calculated by multiplying by `sqrt(periods_per_year)`

### Maximum Drawdown

**Definition:** The largest peak-to-trough decline in portfolio value during the period.

**Formula:**
```python
peak = portfolio_values[0]
max_drawdown = 0
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    max_drawdown = max(max_drawdown, drawdown)
```

**Interpretation:**
- `0.10` = 10% maximum decline from peak
- Lower values indicate less downside risk
- Important for understanding worst-case scenarios

**Example:**
```python
# Portfolio goes from 10,000 -> 12,000 -> 9,000 -> 11,000
# Peak is 12,000, lowest point is 9,000
max_drawdown = (12000 - 9000) / 12000 = 0.25  # 25% drawdown
```

### Win Rate

**Definition:** Percentage of profitable trades (for sell orders only).

**Formula:**
```python
sell_trades = [trade for trade in trades if trade['action'] == 'sell']
profitable_trades = [trade for trade in sell_trades if is_profitable(trade)]
win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0
```

**Interpretation:**
- `0.60` = 60% of trades were profitable
- Higher win rates generally preferred but not always better
- Should be considered alongside average profit per trade

**Note:** Only calculated for sell trades since they represent completed round trips.

### Transaction Costs

**Definition:** Total fees paid during the backtesting period.

**Formula:**
```python
transaction_costs = sum(trade.get('fee', 0) for trade in trades)
```

**Interpretation:**
- Absolute amount in EURC terms
- Higher costs reduce net returns
- Important for comparing strategies with different trading frequencies

## Additional Metrics

### Volatility (Annualized)

```python
returns = np.diff(portfolio_values) / portfolio_values[:-1]
daily_vol = np.std(returns)
# Assuming daily returns, annualize with 365 days
annual_vol = daily_vol * np.sqrt(365)
```

### Calmar Ratio

Risk-adjusted return using maximum drawdown instead of volatility:

```python
calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
```

### Average Trade Return

```python
sell_trades = [trade for trade in trades if trade['action'] == 'sell']
avg_trade_return = np.mean([trade.get('return', 0) for trade in sell_trades])
```

## Trade-Level Metrics

### Trade Profitability Calculation

For each trade, the backtesting system calculates:

```python
def calculate_trade_profitability(buy_trade, sell_trade):
    """Calculate return for a buy-sell pair"""
    buy_cost = buy_trade['cost'] + buy_trade['fee']
    sell_proceeds = sell_trade['proceeds'] - sell_trade['fee']
    trade_return = (sell_proceeds - buy_cost) / buy_cost
    return trade_return
```

### Trade Timing Analysis

```python
# Execution delay analysis
for trade in trades:
    delay = trade['execution_timestamp'] - trade['decision_timestamp']
    runtime = trade['action_runtime']
    network_latency = trade['network_latency']
    print(f"Total delay: {delay:.3f}s (Runtime: {runtime:.3f}s, Network: {network_latency:.3f}s)")
```

## Performance Comparison

### Multi-Strategy Analysis

```python
def compare_strategies(results):
    comparison = {}
    
    for strategy_name, (cal_result, val_result) in results.items():
        comparison[strategy_name] = {
            'cal_return': cal_result.total_return,
            'val_return': val_result.total_return,
            'cal_sharpe': cal_result.sharpe_ratio,
            'val_sharpe': val_result.sharpe_ratio,
            'cal_max_dd': cal_result.max_drawdown,
            'val_max_dd': val_result.max_drawdown,
            'consistency': abs(cal_result.total_return - val_result.total_return),
            'trade_count': len(val_result.trades)
        }
    
    return comparison
```

### Statistical Significance Testing

```python
from scipy import stats

def test_performance_difference(returns1, returns2):
    """Test if two return series are significantly different"""
    t_stat, p_value = stats.ttest_ind(returns1, returns2)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## Visualization Examples

### Performance Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    strategies = list(results.keys())
    
    # 1. Cumulative Returns
    for strategy_name, (cal_result, val_result) in results.items():
        cumulative_returns = np.cumprod(1 + np.diff(val_result.portfolio_values) / val_result.portfolio_values[:-1])
        axes[0,0].plot(val_result.timestamps[1:], cumulative_returns, label=strategy_name)
    axes[0,0].set_title('Cumulative Returns')
    axes[0,0].legend()
    
    # 2. Risk-Return Scatter
    returns = [results[s][1].total_return for s in strategies]
    sharpes = [results[s][1].sharpe_ratio for s in strategies]
    axes[0,1].scatter(returns, sharpes)
    for i, strategy in enumerate(strategies):
        axes[0,1].annotate(strategy, (returns[i], sharpes[i]))
    axes[0,1].set_xlabel('Total Return')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].set_title('Risk-Return Profile')
    
    # 3. Drawdown Analysis
    for strategy_name, (cal_result, val_result) in results.items():
        portfolio_values = val_result.portfolio_values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        axes[0,2].plot(val_result.timestamps, drawdown, label=strategy_name)
    axes[0,2].set_title('Drawdown Over Time')
    axes[0,2].legend()
    
    # 4. Return Distribution
    for strategy_name, (cal_result, val_result) in results.items():
        returns = np.diff(val_result.portfolio_values) / val_result.portfolio_values[:-1]
        axes[1,0].hist(returns, alpha=0.7, bins=50, label=strategy_name)
    axes[1,0].set_title('Return Distribution')
    axes[1,0].legend()
    
    # 5. Win Rate vs Average Return
    win_rates = [results[s][1].win_rate for s in strategies]
    avg_returns = [np.mean([t.get('return', 0) for t in results[s][1].trades]) for s in strategies]
    axes[1,1].scatter(win_rates, avg_returns)
    for i, strategy in enumerate(strategies):
        axes[1,1].annotate(strategy, (win_rates[i], avg_returns[i]))
    axes[1,1].set_xlabel('Win Rate')
    axes[1,1].set_ylabel('Average Trade Return')
    axes[1,1].set_title('Win Rate vs Avg Return')
    
    # 6. Performance Metrics Heatmap
    metrics_data = []
    for strategy in strategies:
        val_result = results[strategy][1]
        metrics_data.append([
            val_result.total_return,
            val_result.sharpe_ratio,
            val_result.max_drawdown,
            val_result.win_rate
        ])
    
    sns.heatmap(metrics_data, 
                xticklabels=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                yticklabels=strategies,
                annot=True, fmt='.3f', ax=axes[1,2])
    axes[1,2].set_title('Performance Metrics Heatmap')
    
    plt.tight_layout()
    plt.show()
```

## Interpretation Guidelines

### Strategy Evaluation Checklist

1. **Profitability**: Is the total return positive and meaningful?
2. **Risk-Adjusted Returns**: Is the Sharpe ratio competitive?
3. **Downside Protection**: Is the maximum drawdown acceptable?
4. **Consistency**: Are calibration and validation results similar?
5. **Trade Efficiency**: Is the win rate reasonable given the strategy type?
6. **Cost Impact**: Are transaction costs manageable relative to returns?

### Red Flags

- **Overfitting**: Excellent calibration results but poor validation performance
- **High Transaction Costs**: Fees consuming significant portion of returns
- **Extreme Drawdowns**: Maximum drawdown > 50% indicates high risk
- **Low Win Rate with Low Average Returns**: Indicates poor trade selection
- **Inconsistent Performance**: Large differences between calibration and validation

### Performance Targets

**Conservative Strategy:**
- Total Return: 5-15% annually
- Sharpe Ratio: > 0.5
- Max Drawdown: < 20%
- Win Rate: > 50%

**Aggressive Strategy:**
- Total Return: 15-50% annually
- Sharpe Ratio: > 1.0
- Max Drawdown: < 40%
- Win Rate: > 45%

**High-Frequency Strategy:**
- Sharpe Ratio: > 2.0 (due to frequent trading)
- Max Drawdown: < 15%
- Win Rate: > 55%
- Low average return per trade but high frequency

## Runtime Performance Metrics

The backtesting framework also tracks strategy execution performance:

```python
def analyze_runtime_performance(backtester):
    for strategy_name, stats in backtester.runtime_stats.items():
        print(f"\n{strategy_name} Runtime Analysis:")
        print(f"  Average execution time: {stats['avg_runtime']:.4f}s")
        print(f"  Total execution time: {stats['total_runtime']:.4f}s")
        print(f"  Number of calls: {stats['action_count']}")
        print(f"  Timeout rate: {stats['timeout_count']/stats['action_count']*100:.1f}%")
        
        # Performance classification
        if stats['avg_runtime'] < 0.01:
            performance = "Excellent"
        elif stats['avg_runtime'] < 0.05:
            performance = "Good"
        elif stats['avg_runtime'] < 0.1:
            performance = "Acceptable"
        else:
            performance = "Slow"
        
        print(f"  Performance rating: {performance}")
```

This comprehensive metrics framework enables thorough evaluation of trading strategies across multiple dimensions of performance, risk, and operational efficiency.