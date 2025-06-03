# Scripts

This folder contains utility scripts for cryptocurrency data processing and analysis.

## Available Scripts

### `preprocess_all_data.py`

Batch preprocessing script that processes all XBT and ETH data files across multiple DATA folders (DATA_0, DATA_1, DATA_2).

**Features:**
- Automatically scans `data/raw/DATA_i` folders for i=0,1,2
- Processes both XBT_EUR.csv and ETH_EUR.csv files
- Outputs preprocessed data to `data/preprocessed/DATA_i/`
- Provides detailed progress reporting and error handling
- Configurable block sizes and processing options
- Skip existing files or force overwrite
- Performance metrics and processing statistics

**Usage:**

```bash
# Basic usage - process all available data
python scripts/preprocess_all_data.py

# Check which files are available without processing
python scripts/preprocess_all_data.py --check-only

# Use custom block size
python scripts/preprocess_all_data.py --block-size 30

# Force overwrite existing files
python scripts/preprocess_all_data.py --force

# Combine options
python scripts/preprocess_all_data.py --block-size 25 --force
```

**Command Line Options:**

- `--block-size`: Block size for preprocessing (default: 20)
- `--force`: Overwrite existing output files
- `--check-only`: Only check which files exist, do not process
- `--base-path`: Base path to the project (default: auto-detect)
- `--help`: Show help message and usage examples

**Example Output:**

```
🔄 Cryptocurrency Data Batch Preprocessor
============================================================
📅 Started at: 2025-06-03 14:30:00
📂 Project path: /path/to/ifcob
📂 Raw data path: /path/to/ifcob/data/raw
📂 Output path: /path/to/ifcob/data/preprocessed

🔍 Scanning for input files...
============================================================

📁 Checking DATA_0:
  ✅ XBT_EUR.csv (842.3 MB)
  ✅ ETH_EUR.csv (2567.1 MB)

📁 Checking DATA_1:
  ❌ XBT_EUR.csv (not found)
  ❌ ETH_EUR.csv (not found)

📁 Checking DATA_2:
  ❌ XBT_EUR.csv (not found)
  ❌ ETH_EUR.csv (not found)

📊 Summary: Found 2 files to process
📊 Total data size: 3409.4 MB

🔄 Processing 2 files with block_size=20...

[1/2] Processing DATA_0/XBT...
Processing XBT data from data\raw\DATA_0\XBT_EUR.csv
Loaded 17601860 rows
Applied vectorized preprocessing with block_size=20
Found 4336 duplicate rows, removing them...
After cleaning: 17597524 rows
Converted to wide format: (880093, 40)
Saved preprocessed data to data\preprocessed\DATA_0\XBT_EUR.parquet
  ✅ Success! Output shape: (880093, 40)
  ⏱️  Processing time: 45.2 seconds

[2/2] Processing DATA_0/ETH...
Processing ETH data from data\raw\DATA_0\ETH_EUR.csv
...
```

## Directory Structure

```
scripts/
├── __init__.py                 # Package initialization
├── README.md                   # This file
└── preprocess_all_data.py      # Batch preprocessing script
```

## Integration

The scripts use the preprocessing module from the main project:

```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder
```

Make sure the preprocessing module is properly installed or available in the Python path.

## Error Handling

The scripts include comprehensive error handling:
- File existence checks
- Processing error recovery
- Detailed error reporting
- Graceful failure handling
- Progress tracking

## Performance

The batch processor provides performance metrics:
- Processing time per file
- Processing rate (MB/s)
- Total processing time
- Memory usage optimization
- Progress indicators
