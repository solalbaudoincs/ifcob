# Feature Generation System

This system provides a flexible and extensible framework for generating features from preprocessed order book data. The system allows you to generate individual features, specific sets of features, or all features at once.

## Overview

The feature generation system consists of:

1. **`feature_generator.py`** - Core module with feature classes and generation logic
2. **`generate_features.py`** - Command-line script for generating features
3. **`create_feature.py`** - Utility script for creating new feature templates

## Features Available

The system currently supports the following financial features:

### Order Book Imbalance Features
- **`bid-ask-imbalance-{n}-levels`** - Measures order book imbalance: (V_bid - V_ask)/(V_bid + V_ask)
- **`liquidity-ratio`** - Liquidity ratio: V_bid/V_ask

### Price Features
- **`spread`** - Bid-ask spread: level-1-ask-price - level-1-bid-price
- **`vwap-bid-{n}-levels`** - Volume-Weighted Average Price for bid side
- **`vwap-ask-{n}-levels`** - Volume-Weighted Average Price for ask side

### Volume Features
- **`V-bid-{n}-levels`** - Cumulative bid volume across multiple levels
- **`V-ask-{n}-levels`** - Cumulative ask volume across multiple levels
- **`rate-bid-volume-level-1`** - Rolling average of bid volume at level 1
- **`rate-ask-volume-level-1`** - Rolling average of ask volume at level 1

### Book Shape Features
- **`slope-bid-{n}-levels`** - Book slope for bid side: (P_N - P_1) / V_sum
- **`slope-ask-{n}-levels`** - Book slope for ask side: (P_N - P_1) / V_sum

### Statistical Features
- **`rate-inst-volatility`** - Instantaneous volatility (rolling variance of mid-price)
- **`rate-momentum`** - Momentum (change in mid-price over time window)
- **`rate-mid-price-trend`** - Trend indicator (rolling mean of mid-price)

## Usage

### Command Line Interface

#### Generate All Features
```bash
# Generate all features for ETH data version 2
python scripts/generate_features.py --coin ETH --data-version 2

# Generate features for multiple coins
python scripts/generate_features.py --coin ETH XBT --data-version 2
```

#### Generate Specific Features
```bash
# Generate only spread and imbalance features
python scripts/generate_features.py --coin ETH --data-version 2 --features spread bid-ask-imbalance-5-levels

# Regenerate a single feature (overwrite existing)
python scripts/generate_features.py --coin ETH --data-version 2 --features spread --overwrite
```

#### List Available Features
```bash
python scripts/generate_features.py --list-features
```

#### Advanced Options
```bash
# Custom input/output directories
python scripts/generate_features.py \\
    --coin ETH \\
    --data-version 2 \\
    --input-dir ./data/preprocessed \\
    --output-dir ./data/features \\
    --verbose

# Force overwrite existing files
python scripts/generate_features.py --coin ETH --data-version 2 --overwrite
```

### Python API

#### Basic Usage
```python
from feature_extraction.feature_generator import FeatureGenerator, load_and_generate_features

# Generate all features
features = load_and_generate_features('ETH', 2)

# Or use the generator directly
generator = FeatureGenerator()
df_cleaned = preprocess_data(df)  # Your preprocessed data
features = generator.generate_all_features(df_cleaned)
```

#### Generate Specific Features
```python
# Generate only specific features
feature_names = ['spread', 'bid-ask-imbalance-5-levels', 'rate-inst-volatility']
features = generator.generate_features(df_cleaned, feature_names)

# Generate a single feature
spread = generator.generate_feature('spread', df_cleaned)
```

#### Working with Individual Features
```python
from feature_extraction.feature_generator import SpreadFeature, BidAskImbalanceFeature

# Create and use individual features
spread_feature = SpreadFeature()
spread_values = spread_feature.generate(df_cleaned)

imbalance_feature = BidAskImbalanceFeature(n_levels=10)
imbalance_values = imbalance_feature.generate(df_cleaned)
```

## Adding New Features

### Method 1: Using the Template Generator

```bash
# Create a new feature template
python scripts/create_feature.py --name RSIFeature --description "Relative Strength Index"

# With custom parameters
python scripts/create_feature.py \\
    --name MACDFeature \\
    --description "MACD indicator" \\
    --parameters "fast_period: int = 12" "slow_period: int = 26" "signal_period: int = 9"
```

### Method 2: Manual Implementation

1. Create a new class inheriting from `BaseFeature`:

```python
class MyCustomFeature(BaseFeature):
    def __init__(self, window: int = 20):
        super().__init__(
            name="my-custom-feature",
            description="Description of my custom feature"
        )
        self.window = window
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        # Implement your feature logic here
        mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        return mid_price.rolling(window=self.window).std()
```

2. Register the feature in the `FeatureGenerator`:

```python
# In _register_default_features() method
self.register_feature(MyCustomFeature(window=20))
```

## Data Structure

### Input Data Format
The system expects preprocessed order book data with the following structure:

```
MultiIndex: (timestamp, row_id)
Columns: level-{i}-{side}-{price|volume}
Where:
- i: 1 to 10 (order book levels)
- side: 'bid' or 'ask'
- price|volume: price or volume data
```

### Output Format
Generated features are returned as a DataFrame with:
- Index: timestamp (unique timestamps from the input data)
- Columns: feature names
- Values: computed feature values

## File Structure

```
feature_extraction/
├── feature_generator.py          # Core feature generation module
└── feature_generation.ipynb      # Original notebook (reference)

scripts/
├── generate_features.py          # Main feature generation script
├── create_feature.py            # Feature template generator
└── README.md                    # This file

data/
├── preprocessed/                # Input: preprocessed order book data
│   └── DATA_{version}/
│       ├── ETH_EUR.parquet
│       └── XBT_EUR.parquet
└── features/                    # Output: generated features
    └── DATA_{version}/
        ├── ETH_EUR.parquet
        └── XBT_EUR.parquet
```

## Configuration

### Default Parameters
- **N_levels**: 5 (number of order book levels to consider)
- **Window**: 20 (rolling window size for statistical features)

### Customizing Parameters
You can customize parameters when creating features:

```python
# Custom imbalance with 10 levels
generator.register_feature(BidAskImbalanceFeature(n_levels=10))

# Custom volatility with 50-period window
generator.register_feature(VolatilityFeature(window=50))
```

## Performance Considerations

1. **Memory Usage**: Large datasets may require chunking for memory efficiency
2. **Rolling Windows**: Features with rolling windows will have NaN values for initial periods
3. **Parallel Processing**: The system can be extended to support parallel processing for multiple coins

## Error Handling

The system includes robust error handling:

- Missing data files are reported with clear error messages
- Invalid feature names are validated before processing
- Individual feature failures don't stop the entire process
- Warnings are issued for features that fail to generate

## Examples

### Complete Workflow Example

```python
import pandas as pd
from feature_extraction.feature_generator import FeatureGenerator, preprocess_data

# Load raw data
df = pd.read_parquet('data/preprocessed/DATA_2/ETH_EUR.parquet')

# Preprocess
df_cleaned = preprocess_data(df)

# Generate features
generator = FeatureGenerator()
features = generator.generate_all_features(df_cleaned)

# Save results
features.to_parquet('data/features/DATA_2/ETH_EUR.parquet')

print(f"Generated {len(features.columns)} features for {len(features)} timestamps")
```

### Batch Processing Multiple Coins

```bash
# Process all available coins
for coin in ETH XBT; do
    echo "Processing $coin..."
    python scripts/generate_features.py --coin $coin --data-version 2 --verbose
done
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure preprocessed data exists in the expected location
2. **MemoryError**: For large datasets, consider processing in smaller chunks
3. **KeyError**: Verify that required columns exist in the input data
4. **Import Error**: Ensure the feature_extraction directory is in your Python path

### Debugging

Enable verbose output to see detailed processing information:

```bash
python scripts/generate_features.py --coin ETH --data-version 2 --verbose
```

## Future Enhancements

Potential improvements to the system:

1. **Parallel Processing**: Support for multi-core processing
2. **Streaming Processing**: Real-time feature generation
3. **Feature Selection**: Automatic feature importance ranking
4. **Configuration Files**: YAML/JSON configuration for feature parameters
5. **Caching**: Intelligent caching of intermediate results
6. **Validation**: Automatic feature validation and quality checks
