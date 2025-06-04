# Portfolio Management Documentation

The `portfolio.py` module handles portfolio state management, trade execution, price estimation, and market impact modeling.

## Core Functions

### Price Estimation

#### `estimate_price(data: OrderBookData) -> float`

Estimates the current market price from order book data using level-1 bid-ask prices.

**Parameters:**
- `data`: Order book DataFrame with price/volume columns

**Returns:**
- Estimated mid-price as float

**Price Estimation Logic:**
1. **Primary method**: Uses level-1 bid-ask mid-price: `(bid + ask) / 2`
2. **Fallback 1**: Generic 'bid'/'ask' columns if level-1 not available
3. **Fallback 2**: Direct 'price' column if present

**Example:**
```python
current_price = estimate_price(market_data['ETH'])
print(f"ETH mid-price: {current_price}")
```

#### `get_bid_ask_spread(data: OrderBookData) -> float`

Calculates the bid-ask spread from level-1 order book data.

**Returns:**
- Spread value: `ask_price - bid_price`

**Usage:**
```python
spread = get_bid_ask_spread(market_data['ETH'])
spread_bps = (spread / current_price) * 10000  # Convert to basis points
```

### Market Impact Modeling

#### `get_market_impact_estimate(data: OrderBookData, trade_amount: float, side: str) -> float`

Models the effective execution price considering order book depth and market impact.

**Parameters:**
- `data`: Order book DataFrame
- `trade_amount`: Size of the trade (always positive)
- `side`: 'buy' or 'sell'

**Returns:**
- Volume-weighted average price (VWAP) considering market impact

**Market Impact Algorithm:**

**For Buy Orders:**
1. Consume ask levels starting from level-1 (best ask)
2. For each level, consume available volume at that price
3. Continue to deeper levels until trade is filled
4. Return VWAP across all consumed levels

**For Sell Orders:**
1. Consume bid levels starting from level-1 (best bid)
2. Similar process but consuming bid-side liquidity
3. Return VWAP of proceeds

**Example:**
```python
# Large buy order - will impact price
buy_price = get_market_impact_estimate(data, 10.0, 'buy')
mid_price = estimate_price(data)
impact = (buy_price - mid_price) / mid_price
print(f"Market impact: {impact:.4f}")
```

### Fee Management

#### `get_fee_for_trade(coin_from: Coin, coin_to: Coin, fees_graph: FeesGraph) -> float`

Retrieves the fee rate for trading between two coins using the fees graph.

**Parameters:**
- `coin_from`: Source coin
- `coin_to`: Target coin  
- `fees_graph`: Fee structure graph

**Returns:**
- Fee rate as decimal (e.g., 0.001 = 0.1%)

**Example:**
```python
fee_rate = get_fee_for_trade('EURC', 'ETH', fees_graph)
cost_with_fees = trade_amount * (1 + fee_rate)
```

## Portfolio Class

The main class for managing portfolio positions and executing trades.

### Constructor

```python
def __init__(self, coins: list[Coin], initial_amount: float) -> None
```

**Parameters:**
- `coins`: List of coin symbols to track
- `initial_amount`: Starting capital in EURC

**Initialization:**
- Sets EURC balance to `initial_amount`
- Initializes all other coin positions to 0.0
- Stores coin list for validation

### Portfolio State Management

#### `get_position(coin: Coin) -> float`

Returns current position for a specific coin.

**Example:**
```python
eth_balance = portfolio.get_position('ETH')
eurc_balance = portfolio.get_position('EURC')
```

#### `update_position(coin: Coin, amount: float) -> None`

Updates position for a specific coin (adds to existing position).

**Parameters:**
- `coin`: Coin symbol
- `amount`: Amount to add (can be negative)

**Usage:**
```python
portfolio.update_position('ETH', 0.5)   # Add 0.5 ETH
portfolio.update_position('EURC', -1000)  # Subtract 1000 EURC
```

#### `get_value(market_data: MarketData) -> float`

Calculates total portfolio value in EURC terms.

**Parameters:**
- `market_data`: Current market data for price estimation

**Returns:**
- Total portfolio value in EURC

**Calculation:**
```python
total_value = eurc_position
for coin, amount in positions:
    if coin != 'EURC' and amount != 0:
        price = estimate_price(market_data[coin])
        total_value += amount * price
```

### Trade Execution

#### `can_execute_trade(coin_from: Coin, coin_to: Coin, amount: float, fees_graph: FeesGraph) -> bool`

Checks if a trade can be executed given current positions and fees.

**Parameters:**
- `coin_from`: Source coin
- `coin_to`: Target coin
- `amount`: Trade amount
- `fees_graph`: Fee structure

**Returns:**
- True if trade is feasible, False otherwise

**Validation:**
- Checks sufficient balance in source coin
- Accounts for trading fees
- Validates fee graph contains required trading pair

#### `execute_trade(coin_from: Coin, coin_to: Coin, amount: float, fees_graph: FeesGraph) -> bool`

Executes a trade between two coins with fee deduction.

**Parameters:**
- `coin_from`: Source coin to sell
- `coin_to`: Target coin to buy
- `amount`: Amount of target coin to acquire
- `fees_graph`: Fee structure graph

**Returns:**
- True if trade executed successfully, False otherwise

**Execution Process:**
1. Validate trade feasibility using `can_execute_trade()`
2. Calculate total cost including fees: `cost = amount * (1 + fee_rate)`
3. Deduct cost from source coin position
4. Add amount to target coin position
5. Return success status

**Example:**
```python
# Buy 1.0 ETH using EURC
success = portfolio.execute_trade('EURC', 'ETH', 1.0, fees_graph)
if success:
    print("Trade executed successfully")
else:
    print("Trade failed - insufficient funds or invalid pair")
```

## Usage Examples

### Basic Portfolio Setup

```python
from backtesting.portfolio import Portfolio

# Initialize portfolio with 10,000 EURC
coins = ['ETH', 'XBT', 'LTC']
portfolio = Portfolio(coins, 10000.0)

print(f"Starting EURC: {portfolio.get_position('EURC')}")
print(f"Starting ETH: {portfolio.get_position('ETH')}")
```

### Price Analysis

```python
from backtesting.portfolio import estimate_price, get_bid_ask_spread, get_market_impact_estimate

# Analyze current market conditions
eth_data = market_data['ETH']
current_price = estimate_price(eth_data)
spread = get_bid_ask_spread(eth_data)
spread_pct = (spread / current_price) * 100

print(f"ETH Price: {current_price:.2f}")
print(f"Spread: {spread:.4f} ({spread_pct:.3f}%)")

# Estimate impact for different trade sizes
for size in [0.1, 0.5, 1.0, 5.0]:
    buy_price = get_market_impact_estimate(eth_data, size, 'buy')
    impact = ((buy_price - current_price) / current_price) * 10000  # basis points
    print(f"Buy {size} ETH - Impact: {impact:.1f} bps")
```

### Fee Structure Setup

```python
# Define trading fees (0.1% for all pairs)
fees_graph = {
    'EURC': [('ETH', 0.001), ('XBT', 0.001), ('LTC', 0.001)],
    'ETH': [('EURC', 0.001)],
    'XBT': [('EURC', 0.001)],
    'LTC': [('EURC', 0.001)]
}

# Check fee for specific trade
fee_rate = get_fee_for_trade('EURC', 'ETH', fees_graph)
print(f"EURC -> ETH fee: {fee_rate * 100}%")
```

### Portfolio Value Tracking

```python
# Track portfolio value over time
value_history = []

for timestamp, market_data in data_iterator:
    portfolio_value = portfolio.get_value(market_data)
    value_history.append((timestamp, portfolio_value))
    
    # Calculate returns
    if len(value_history) > 1:
        prev_value = value_history[-2][1]
        return_pct = (portfolio_value - prev_value) / prev_value
        print(f"Portfolio return: {return_pct:.4f}")
```

### Trade Execution with Validation

```python
def safe_execute_trade(portfolio, coin_from, coin_to, amount, fees_graph, market_data):
    """Execute trade with comprehensive validation and logging"""
    
    # Check current positions
    from_balance = portfolio.get_position(coin_from)
    print(f"Current {coin_from} balance: {from_balance}")
    
    # Validate trade feasibility
    if not portfolio.can_execute_trade(coin_from, coin_to, amount, fees_graph):
        print(f"Cannot execute trade: insufficient {coin_from}")
        return False
    
    # Get market conditions
    if coin_to in market_data:
        current_price = estimate_price(market_data[coin_to])
        impact_price = get_market_impact_estimate(market_data[coin_to], amount, 'buy' if coin_from == 'EURC' else 'sell')
        impact_pct = ((impact_price - current_price) / current_price) * 100
        print(f"Market impact: {impact_pct:.3f}%")
    
    # Execute trade
    success = portfolio.execute_trade(coin_from, coin_to, amount, fees_graph)
    
    if success:
        fee_rate = get_fee_for_trade(coin_from, coin_to, fees_graph)
        fee_amount = amount * fee_rate
        print(f"Trade executed: {amount} {coin_to}, fee: {fee_amount}")
        
        # Log new balances
        new_from_balance = portfolio.get_position(coin_from)
        new_to_balance = portfolio.get_position(coin_to)
        print(f"New balances - {coin_from}: {new_from_balance}, {coin_to}: {new_to_balance}")
    
    return success
```

## Performance Considerations

### Memory Efficiency
- Portfolio positions stored as simple dictionary
- Price calculations use vectorized pandas operations
- Market impact modeling processes levels sequentially

### Computational Efficiency  
- Fee lookups are O(1) dictionary operations
- Price estimation uses iloc[-1] for latest data
- Market impact calculation short-circuits when trade is filled

### Accuracy Considerations
- Mid-price estimation may not reflect actual execution price
- Market impact model assumes immediate execution across levels
- Does not account for partial fills or order placement strategies
- Fee calculation assumes percentage-based fees only