# Types and Interfaces Documentation

This file defines the core type aliases and abstract base classes used throughout the backtesting system.

## Type Aliases

### Basic Types
```python
Coin: TypeAlias = str
```
Represents a cryptocurrency symbol (e.g., 'ETH', 'XBT', 'EURC').

```python
Filepath: TypeAlias = str
```
Represents a file path to data files.

```python
TimeStep: TypeAlias = float
```
Represents a timestamp value (typically Unix timestamp in milliseconds).

### Market Data Types
```python
OrderBookData: TypeAlias = pd.DataFrame
```
A pandas DataFrame containing order book data with price/volume information and extracted features. Expected columns include:
- `timestamp`: Unix timestamp
- `level-{n}-bid-price/volume`: Bid prices and volumes for levels 1-10
- `level-{n}-ask-price/volume`: Ask prices and volumes for levels 1-10
- Feature columns: `bid-ask-imbalance-5-levels`, `spread`, `inst-return`, etc.

```python
MarketData: TypeAlias = dict[Coin, OrderBookData]
```
A dictionary mapping coin symbols to their respective order book DataFrames.

### Trading Types
```python
Action: TypeAlias = dict[Coin, float]
```
Represents trading actions returned by strategies. Positive values indicate buy amounts, negative values indicate sell amounts.

Example:
```python
action = {
    'ETH': 0.5,    # Buy 0.5 ETH
    'XBT': -0.1,   # Sell 0.1 XBT
    'LTC': 0.0     # Hold LTC position
}
```

```python
FeePrice: TypeAlias = float
```
Represents a fee rate as a decimal (e.g., 0.001 = 0.1% fee).

```python
FeesGraph: TypeAlias = dict[Coin, list[tuple[Coin, FeePrice]]]
```
Represents an oriented graph of transaction fees between coins. Each coin maps to a list of (target_coin, fee_rate) tuples.

Example:
```python
fees_graph = {
    'EURC': [('ETH', 0.001), ('XBT', 0.001)],
    'ETH': [('EURC', 0.001)],
    'XBT': [('EURC', 0.001)]
}
```

## Abstract Base Classes

### OrderBookDataLoader
Abstract base class for loading and managing market data.

#### Methods

##### `get_books_from_range(start_time: TimeStep, end_time: TimeStep) -> MarketData`
**Abstract method** that must be implemented by subclasses.

Retrieves market data for all symbols within the specified time range.

**Parameters:**
- `start_time`: Start timestamp (inclusive)
- `end_time`: End timestamp (inclusive)

**Returns:**
- Dictionary mapping coin symbols to their order book data

**Raises:**
- `ValueError`: If time range is invalid or outside available data

##### `get_time_step_values() -> dict[Coin, np.ndarray]`
**Abstract method** that must be implemented by subclasses.

Returns all available timestamps for each coin in the dataset.

**Returns:**
- Dictionary mapping coin symbols to sorted arrays of timestamps

**Example:**
```python
timesteps = loader.get_time_step_values()
print(timesteps['ETH'])  # Array of all ETH timestamps
```

##### `chronological_iterator()`
**Concrete method** that provides a generator for chronological iteration across multiple coin datasets.

Yields unique timestamps in chronological order along with current indices for all coins. This ensures that backtesting processes data in the correct temporal sequence across all assets.

**Yields:**
- `tuple`: (min_time, coin_indices)
  - `min_time`: Next chronological timestamp
  - `coin_indices`: Dict mapping coin names to current indices

**Features:**
- Ensures each timestamp is yielded only once
- Automatically removes coins that have exhausted their data
- Maintains chronological order across all datasets
- Handles coins with different timestamp frequencies

**Usage Example:**
```python
for timestamp, indices in dataloader.chronological_iterator():
    # Process data at this timestamp
    market_data = create_windowed_data(timestamp, indices)
    # ... backtesting logic
```

### Strategy
Abstract base class for trading strategies.

#### Methods

##### `get_action(data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action`
**Abstract method** that must be implemented by strategy subclasses.

Generates trading actions based on current market data and portfolio state.

**Parameters:**
- `data`: Current market data for all coins (windowed)
- `current_portfolio`: Current portfolio positions
- `fees_graph`: Trading fee structure

**Returns:**
- Dictionary specifying trading amounts for each coin

**Implementation Notes:**
- Should return `{}` or `{coin: 0.0}` for no action
- Positive values = buy, negative values = sell
- Strategy should consider current positions and fees
- Execution time is measured and may trigger timeouts

**Example Implementation:**
```python
def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
    # Simple momentum strategy
    if 'ETH' in data and len(data['ETH']) > 1:
        recent_return = (data['ETH']['level-1-bid-price'].iloc[-1] / 
                        data['ETH']['level-1-bid-price'].iloc[-2]) - 1
        
        if recent_return > 0.001:  # 0.1% positive return
            return {'ETH': 0.1}  # Buy 0.1 ETH
        elif recent_return < -0.001:  # 0.1% negative return
            return {'ETH': -0.1}  # Sell 0.1 ETH
    
    return {}  # Hold
```

## Usage Notes

### Data Format Requirements
Order book data is expected to have specific column naming conventions:
- Timestamps: `timestamp` column with Unix timestamps
- Order book levels: `level-{n}-{side}-{field}` format
  - `n`: Level number (1-10)
  - `side`: 'bid' or 'ask'
  - `field`: 'price' or 'volume'

### Error Handling
All abstract methods should include proper error handling:
- Validate time ranges and parameters
- Handle missing data gracefully
- Provide informative error messages
- Log warnings for non-critical issues

### Performance Considerations
- Use vectorized operations where possible
- Implement efficient timestamp indexing
- Consider memory usage for large datasets
- Cache frequently accessed data