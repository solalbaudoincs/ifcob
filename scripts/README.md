# Scripts

This folder contains utility scripts for cryptocurrency data processing, feature engineering, model management, and backtesting.

## Available Scripts

### `preprocess_all_data.py`
Batch preprocessing script for all XBT and ETH data files across multiple DATA folders (DATA_0, DATA_1, DATA_2).

**Features:**
- Automatically scans `data/raw/DATA_i` folders for i=0,1,2
- Processes both XBT_EUR.csv and ETH_EUR.csv files
- Outputs preprocessed data to `data/preprocessed/DATA_i/`
- Detailed progress reporting and error handling
- Configurable block sizes and processing options
- Skip existing files or force overwrite
- Performance metrics and processing statistics

**Usage:**
```powershell
python scripts/preprocess_all_data.py [options]
```

---

### `generate_features.py`
Command-line tool to generate features from preprocessed order book data. Supports generating all, specific, or individual features, listing available features, and recomputing features.

**Features:**
- Generate features for a given coin and data version
- Select specific features to generate
- List available features
- Overwrite or recompute features

**Usage:**
```powershell
python scripts/generate_features.py --coin ETH --data-version 2
python scripts/generate_features.py --coin XBT --data-version 1 --features spread bid-ask-imbalance-5-levels
python scripts/generate_features.py --coin ETH --data-version 2 --list-features
```

---

### `create_feature.py`
Utility to generate new feature class templates and manage feature files (add/remove), including updating imports.

**Features:**
- Generate new feature class templates and files
- Automatically add/remove imports in `feature_extraction/features/__init__.py` and `feature_generator.py`
- Remove features and clean up all related imports and files

**Usage:**
```powershell
# Create a new feature
python scripts/create_feature.py --name MyNewFeature --description "Description of the feature"

# Remove a feature (deletes file and cleans up imports)
python scripts/create_feature.py --remove MyNewFeature
```

---

### `manage_models.py`
Script to train, test, tune, and compare models using the ModelManager.

**Features:**
- Train a model with optional hyperparameters: `--n_estimators`, `--max_depth`, `--learning_rate`
- Test a model and print/save performance report
- Compare models (hyperparameters must be set directly via CLI arguments)
- Select and display the best hyperparameters from previous runs

**Usage:**
```powershell
# Train a model (with optional hyperparameters)
python scripts/manage_models.py train --model random_forest_mateo --features <features_path> --target <target_path> [--n_estimators 5] [--max_depth 3] [--learning_rate 0.1]

# Test a model
python scripts/manage_models.py test --model random_forest_mateo --features <features_path> --target <target_path> --load <model_path>

# Compare models (set hyperparameters directly)
python scripts/manage_models.py compare --model random_forest_mateo --features <features_path> --target <target_path>

# Select best hyperparameters from previous runs
python scripts/manage_models.py select --model random_forest_mateo
```

---

### `run_backtest.py`
Script to run backtests using strategies from the `strategies` folder.

**Features:**
- Run single or multiple strategies on specified data
- Support for custom data sources and initial capital
- Save results and trade logs to output directories
- Profile execution for performance analysis

**Usage:**
```powershell
# Run a single strategy
python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --window-size 5

# Run multiple strategies
python scripts/run_backtest.py --strategy Mateo2StartStrategy --strategy RFPredAllSignedStratMateoCheating --data-index 2

# Custom data and parameters
python scripts/run_backtest.py --strategy RFPredAllSignedStratMateo --data-index 1 --data-sources XBT:data/custom/XBT.parquet ETH:data/custom/ETH.parquet --initial-capital 500000

# Save results
python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --output-dir backtest_results --save-trades

# Profile execution
python scripts/run_backtest.py --strategy Mateo2StartStrategy --data-index 2 --profile
```

---

### `usal_commands.txt`
A text file with useful shell commands or notes for project maintenance and development. You can add your own frequently used commands here for quick reference.

---

## Directory Structure

```
scripts/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── preprocess_all_data.py      # Batch preprocessing script
├── generate_features.py        # Feature generation script
├── create_feature.py           # Feature template generator
├── manage_models.py            # Model training/testing script
├── run_backtest.py             # Backtesting runner script
├── usal_commands.txt           # Useful shell commands
```

## Integration

The scripts use the preprocessing, feature extraction, and prediction modules from the main project:

```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder
from feature_extraction.feature_generator import FeatureGenerator
from prediction_model.model_manager import ModelManager
```

Make sure the required modules are properly installed or available in the Python path. If you encounter import errors, check your `PYTHONPATH` or run scripts from the project root.

## Error Handling & Performance

- Comprehensive error handling: file existence checks, error recovery, detailed reporting, and progress tracking.
- Performance metrics: processing time, rate, memory usage, and progress indicators for batch and feature generation scripts.

---

## Extending & Troubleshooting

- To add new scripts, follow the structure and docstring conventions used here.
- For more details on each script, see the inline comments and docstrings in the respective files.
- If you encounter issues, check the logs and error messages for troubleshooting tips.

## Usage Details

Below are detailed usage instructions and example scenarios for each script in this folder:

---

### preprocess_all_data.py
Batch preprocesses all XBT and ETH data files across multiple DATA folders (DATA_0, DATA_1, DATA_2). Use this script to prepare raw CSV data for feature extraction and modeling.

**Example:**
```powershell
python scripts/preprocess_all_data.py --block-size 100000 --skip-existing
```
- `--block-size`: Set the number of rows to process at a time (improves memory usage).
- `--skip-existing`: Skip files that have already been processed.

---

### generate_features.py
Generates features from preprocessed order book data. You can generate all features, a subset, or just list available features.

**Examples:**
```powershell
# Generate all features for ETH, data version 2
python scripts/generate_features.py --coin ETH --data-version 2

# Generate specific features for XBT
python scripts/generate_features.py --coin XBT --data-version 1 --features spread bid-ask-imbalance-5-levels

# List all available features for ETH
python scripts/generate_features.py --coin ETH --data-version 2 --list-features

# Recompute a single feature and overwrite
python scripts/generate_features.py --coin ETH --data-version 2 --features spread --overwrite
```
- `--coin`: Specify one or more coins (e.g., ETH, XBT).
- `--data-version`: Select the data version (e.g., 1, 2).
- `--features`: List of features to generate.
- `--list-features`: List all available features for the coin/version.
- `--overwrite`: Overwrite existing feature files.

---

### create_feature.py
Quickly create or remove feature class templates. This script helps you scaffold new features or clean up old ones.

**Examples:**
```powershell
# Create a new feature template
python scripts/create_feature.py --name MyNewFeature --description "Description of the feature"

# Remove a feature and clean up imports
python scripts/create_feature.py --remove MyNewFeature
```
- `--name`: Name of the new feature class.
- `--description`: Short description for the feature docstring.
- `--remove`: Remove a feature by class name.

---

### manage_models.py
Train, test, tune, and compare models using the ModelManager. Supports direct CLI hyperparameter setting and best hyperparameter selection.

**Examples:**
```powershell
# Train a model with custom hyperparameters
python scripts/manage_models.py train --model random_forest_mateo --features data/features/DATA_0/XBT_EUR.parquet --target data/features/DATA_0/ETH_EUR.parquet --n_estimators 150 --max_depth 4

# Test a trained model
python scripts/manage_models.py test --model random_forest_mateo --features data/features/DATA_0/XBT_EUR.parquet --target data/features/DATA_0/ETH_EUR.parquet --load predictors/mateo/model.joblib

# Compare models with different hyperparameters
python scripts/manage_models.py compare --model random_forest_mateo --features data/features/DATA_0/XBT_EUR.parquet --target data/features/DATA_0/ETH_EUR.parquet --n_estimators 100 --max_depth 3

# Select the best hyperparameters from previous runs
python scripts/manage_models.py select --model random_forest_mateo
```
- `--model`: Model name (see ModelManager.MODELS for options).
- `--features`, `--target`: Paths to feature and target data.
- `--n_estimators`, `--max_depth`, `--learning_rate`: Model hyperparameters.
- `--load`: Path to a saved model for testing.

---

### run_backtest.py
Run backtests using strategies from the `strategies` folder. Supports single/multiple strategies, custom data, saving results, and profiling.

**Examples:**
```powershell
# Run a single strategy
python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --window-size 5

# Run multiple strategies
python scripts/run_backtest.py --strategy Mateo2StartStrategy --strategy RFPredAllSignedStratMateoCheating --data-index 2

# Use custom data sources and set initial capital
python scripts/run_backtest.py --strategy RFPredAllSignedStratMateo --data-index 1 --data-sources XBT:data/custom/XBT.parquet ETH:data/custom/ETH.parquet --initial-capital 500000

# Save results and trade logs
python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --output-dir backtest_results --save-trades

# Profile execution for performance analysis
python scripts/run_backtest.py --strategy Mateo2StartStrategy --data-index 2 --profile
```
- `--strategy`: Name(s) of the strategy class(es) to run.
- `--data-index`: Which data folder to use (e.g., 1 for DATA_1).
- `--window-size`: Window size parameter for the strategy.
- `--data-sources`: Custom data file paths for each coin.
- `--initial-capital`: Starting capital for the backtest.
- `--output-dir`: Directory to save results.
- `--save-trades`: Save trade logs.
- `--profile`: Enable profiling for performance analysis.

---

### usal_commands.txt
A text file with useful shell commands or notes for project maintenance and development. Add your own frequently used commands for quick reference.

---

## Directory Structure

```
scripts/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── preprocess_all_data.py      # Batch preprocessing script
├── generate_features.py        # Feature generation script
├── create_feature.py           # Feature template generator
├── manage_models.py            # Model training/testing script
├── run_backtest.py             # Backtesting runner script
├── usal_commands.txt           # Useful shell commands
```

## Integration

The scripts use the preprocessing, feature extraction, and prediction modules from the main project:

```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder
from feature_extraction.feature_generator import FeatureGenerator
from prediction_model.model_manager import ModelManager
```

Make sure the required modules are properly installed or available in the Python path. If you encounter import errors, check your `PYTHONPATH` or run scripts from the project root.

## Error Handling & Performance

- Comprehensive error handling: file existence checks, error recovery, detailed reporting, and progress tracking.
- Performance metrics: processing time, rate, memory usage, and progress indicators for batch and feature generation scripts.

---

## Extending & Troubleshooting

- To add new scripts, follow the structure and docstring conventions used here.
- For more details on each script, see the inline comments and docstrings in the respective files.
- If you encounter issues, check the logs and error messages for troubleshooting tips.
