# Preprocessing Module

This module provides functions for preprocessing cryptocurrency market data from CSV format to a structured wide format suitable for feature extraction and machine learning.

## Files

- `data_preprocessor.py`: Core preprocessing functions
- `preprocess_script.py`: Standalone command-line script
- `example_usage.py`: Usage examples
- `__init__.py`: Makes the module importable

## Installation

No additional installation required if you have pandas and numpy installed:

```bash
pip install pandas numpy
```

## Usage

### As an Imported Module

```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder

# Process single file
df = preprocess_crypto_data(
    input_file='../data/raw/DATA_0/XBT_EUR.csv',
    output_file='../data/preprocessed/DATA_0/XBT_EUR.parquet',
    coin='XBT'
)

# Process entire folder
results = preprocess_data_folder(
    input_folder='../data/raw/DATA_0',
    output_folder='../data/preprocessed/DATA_0',
    coins=['XBT', 'ETH']  # Optional: specify coins
)
```

### As a Standalone Script

```bash
# Process all CSV files in a folder
python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0

# Process specific coins
python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0 --coins XBT ETH

# Process single file
python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0 --single_file XBT

# Use custom block size
python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0 --block_size 30
```

## Functions

### `preprocess_crypto_data(input_file, output_file, coin, block_size=20)`

Processes a single cryptocurrency data file through the complete preprocessing pipeline.

**Parameters:**
- `input_file`: Path to input CSV file
- `output_file`: Path to output parquet file  
- `coin`: Cryptocurrency symbol (for logging)
- `block_size`: Size of each block for grouping rows (default: 20)

**Returns:** Preprocessed DataFrame in wide format

### `preprocess_data_folder(input_folder, output_folder, coins=None, block_size=20)`

Processes all cryptocurrency data files in a folder.

**Parameters:**
- `input_folder`: Path to input folder containing CSV files
- `output_folder`: Path to output folder for parquet files
- `coins`: List of coin symbols to process (optional)
- `block_size`: Size of each block for grouping rows (default: 20)

**Returns:** Dictionary with coin symbols as keys and processed DataFrames as values

### `preprocessing_vectorized(df, block_size=20)`

Applies vectorized preprocessing by creating blocks of data and aligning timestamps.

### `clean_duplicates(df)`

Removes duplicates based on timestamp, row_id, level, and side columns.

### `pivot_to_wide_format(df)`

Converts dataframe to wide format suitable for feature extraction.

## Data Format

### Input Format (CSV)
Expected columns: `timestamp`, `level`, `side`, `price`, `volume`

### Output Format (Parquet)
Wide format with columns like: `level-1-ask-price`, `level-1-ask-volume`, `level-1-bid-price`, etc.

The output has a MultiIndex with `timestamp` and `row_id` as indices.

## Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

This will demonstrate:
- Single file processing
- Folder processing
- Custom usage with individual functions
