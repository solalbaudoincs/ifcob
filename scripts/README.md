# Scripts

This folder contains utility scripts for cryptocurrency data processing, feature engineering, and model management.

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
python scripts/preprocess_all_data.py [options]
```

---

### `generate_features.py`
Feature generation script for preprocessed order book data. Supports generating all, specific, or individual features.

**Features:**
- Generate features for a given coin and data version
- Select specific features to generate
- List available features
- Overwrite existing features

**Usage:**
```bash
python scripts/generate_features.py --coin ETH --data-version 2
python scripts/generate_features.py --coin XBT --data-version 1 --features spread bid-ask-imbalance-5-levels
python scripts/generate_features.py --coin ETH --data-version 2 --list-features
```

---

### `create_feature.py`

Feature template generator and feature management utility.

**Features:**
- Generate new feature class templates and files
- Automatically add imports to `feature_extraction/features/__init__.py` and `feature_generator.py`
- Remove features and clean up all related imports and files

**Usage:**
```bash
# Create a new feature
python scripts/create_feature.py --name MyNewFeature --description "Description of the feature"

# Remove a feature (deletes file and cleans up imports)
python scripts/create_feature.py --remove MyNewFeature
```

---

### `manage_models.py`
Script to train, test, tune, and compare models using the ModelManager.

**Features:**
- Train models
- Test models
- Compare/tune hyperparameters

**Usage:**
```bash
python scripts/manage_models.py train --model random_forest_mateo --features <features_path> --target <target_path>
python scripts/manage_models.py test --model random_forest_mateo --features <features_path> --target <target_path> --load <model_path>
python scripts/manage_models.py compare --model random_forest_mateo --features <features_path> --target <target_path> --param_grid '{"n_estimators": [50, 100], "max_depth": [3, 5]}'
```

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
```

## Integration

The scripts use the preprocessing and prediction modules from the main project:

```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder
from prediction_model.model_manager import ModelManager
```

Make sure the required modules are properly installed or available in the Python path.

## Error Handling

The scripts include comprehensive error handling:
- File existence checks
- Processing error recovery
- Detailed error reporting
- Graceful failure handling
- Progress tracking

## Performance

The batch processor and feature generator provide performance metrics:
- Processing time per file
- Processing rate (MB/s)
- Total processing time
- Memory usage optimization
- Progress indicators
