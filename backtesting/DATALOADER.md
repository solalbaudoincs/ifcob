# Data Loading Documentation

The `dataloader.py` module provides concrete implementations for loading and managing historical order book data for backtesting.

## OrderBookDataFromDf

The primary implementation that loads order book data from parquet files into pandas DataFrames.

### Constructor

```python
def __init__(self, sources: list[tuple[Coin, Filepath]]) -> None
```

**Parameters:**
- `sources`: List of (coin_symbol, file_path) tuples specifying data sources

**Example:**
```python
sources = [
    ('ETH', 'data/preprocessed/DATA_0/ETH_EUR.parquet'),
    ('XBT', 'data/preprocessed/DATA_0/XBT_EUR.parquet')
]
dataloader = OrderBookDataFromDf(sources)
```

**Initialization Process:**
1. Loads each parquet file into a pandas DataFrame
2. Resets index to ensure timestamp is a column
3. Flattens multi-index columns if present
4. Stores DataFrames in `self.dfs` dictionary
5. Creates `self.coins` list for quick access to coin symbols

### Methods

#### `get_book_from_range(coin: Coin, start_time: TimeStep, end_time: TimeStep) -> pd.DataFrame`

Retrieves order book data for a single coin within a time range.

**Parameters:**
- `coin`: Cryptocurrency symbol
- `start_time`: Start timestamp (inclusive)
- `end_time`: End timestamp (inclusive)

**Returns:**
- DataFrame slice containing data within the specified time range

**Performance Features:**
- Uses `numpy.searchsorted()` for efficient timestamp-based slicing
- Assumes data is sorted by timestamp for optimal performance
- Returns a view of the original DataFrame (memory efficient)

**Error Handling:**
- Validates time range is within available data bounds
- Ensures start_time ≤ end_time
- Raises `ValueError` with descriptive messages for invalid ranges

**Example:**
```python
# Get ETH data for a specific hour
eth_data = dataloader.get_book_from_range('ETH', 1609459200000, 1609462800000)
print(f"Retrieved {len(eth_data)} rows of ETH data")
```

#### `get_books_from_range(start_time: TimeStep, end_time: TimeStep) -> MarketData`

Retrieves order book data for all coins within a time range.

**Parameters:**
- `start_time`: Start timestamp (inclusive)
- `end_time`: End timestamp (inclusive)

**Returns:**
- Dictionary mapping coin symbols to their respective DataFrames

**Implementation:**
- Calls `get_book_from_range()` for each coin
- Returns consistent MarketData format
- Handles coins with different data availability gracefully

**Example:**
```python
# Get data for all coins during a specific period
market_data = dataloader.get_books_from_range(1609459200000, 1609462800000)
for coin, data in market_data.items():
    print(f"{coin}: {len(data)} rows")
```

#### `get_time_step_values() -> dict[Coin, np.ndarray]`

Returns all available timestamps for each coin.

**Returns:**
- Dictionary mapping coin symbols to numpy arrays of timestamps

**Features:**
- Extracts timestamp values directly from DataFrame columns
- Returns numpy arrays for efficient processing
- Maintains original timestamp ordering

**Usage in Backtesting:**
- Used by `chronological_iterator()` to coordinate multi-coin iteration
- Enables efficient timestamp-based operations
- Supports validation of time range requests

**Example:**
```python
timesteps = dataloader.get_time_step_values()
print(f"ETH has {len(timesteps['ETH'])} timestamps")
print(f"First ETH timestamp: {timesteps['ETH'][0]}")
print(f"Last ETH timestamp: {timesteps['ETH'][-1]}")
```

## Data Format Requirements

### File Format
- **Parquet files**: Preferred for efficient storage and loading
- **Index handling**: Files may have timestamp as index or column
- **Multi-index support**: Handles hierarchical column structures

### Expected Columns

#### Required Columns
- `timestamp`: Unix timestamp in milliseconds (float64)

#### Order Book Levels (1-10)
- `level-{n}-bid-price`: Bid price at level n
- `level-{n}-bid-volume`: Bid volume at level n  
- `level-{n}-ask-price`: Ask price at level n
- `level-{n}-ask-volume`: Ask volume at level n

#### Feature Columns (Optional)
- `bid-ask-imbalance-5-levels`: Order book imbalance
- `spread`: Bid-ask spread
- `inst-return`: Instantaneous return
- `V-bid-5-levels`: Bid volume aggregated over 5 levels
- `V-ask-5-levels`: Ask volume aggregated over 5 levels
- `slope-bid-5-levels`: Bid price slope
- `slope-ask-5-levels`: Ask price slope

### Data Quality Requirements

#### Timestamp Consistency
- Timestamps must be sorted in ascending order
- No duplicate timestamps within a single coin's data
- Consistent timestamp units across all coins

#### Missing Data Handling
- Missing values should be handled at the preprocessing stage
- Order book levels should have valid price/volume pairs
- Feature columns may contain NaN values if properly documented

## Performance Considerations

### Memory Management
- DataFrames are loaded once at initialization
- Views are returned instead of copies where possible
- Consider data size when loading multiple large files

### Query Optimization
- Timestamp-based queries use binary search (`searchsorted`)
- Avoid frequent small range queries; prefer larger batches
- Index DataFrames by timestamp for repeated access patterns

### Scalability
- Current implementation loads all data into memory
- For very large datasets, consider implementing lazy loading
- Parquet format provides good compression and load speed

## Error Handling

### Common Errors
```python
# Time range outside available data
ValueError: Requested time range [start, end] is outside available data range

# Invalid time ordering  
ValueError: Start time cannot be after end time

# Missing coin data
KeyError: Coin 'SYMBOL' not found in loaded data
```

### Best Practices
- Always validate time ranges before querying
- Check data availability using `get_time_step_values()`
- Handle missing coins gracefully in strategy code
- Monitor memory usage with large datasets

## Usage Examples

### Basic Setup
```python
from backtesting.dataloader import OrderBookDataFromDf

# Load data for multiple coins
sources = [
    ('ETH', 'data/preprocessed/DATA_0/ETH_EUR.parquet'),
    ('XBT', 'data/preprocessed/DATA_0/XBT_EUR.parquet'),
    ('LTC', 'data/preprocessed/DATA_0/LTC_EUR.parquet')
]

dataloader = OrderBookDataFromDf(sources)
print(f"Loaded data for coins: {dataloader.coins}")
```

### Data Exploration
```python
# Check available time ranges
timesteps = dataloader.get_time_step_values()
for coin, times in timesteps.items():
    print(f"{coin}: {len(times)} records from {times[0]} to {times[-1]}")

# Sample recent data
recent_start = timesteps['ETH'][-1000]  # Last 1000 timestamps
recent_end = timesteps['ETH'][-1]
recent_data = dataloader.get_book_from_range('ETH', recent_start, recent_end)
print(f"Recent ETH data shape: {recent_data.shape}")
```

### Integration with Backtesting
```python
# Use with chronological iterator
for timestamp, indices in dataloader.chronological_iterator():
    if timestamp > target_end_time:
        break
    
    # Get windowed data around current timestamp
    window_start = timestamp - window_size_ms
    market_data = dataloader.get_books_from_range(window_start, timestamp)
    
    # Process with strategy
    # ...
```