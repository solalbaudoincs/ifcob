# Prediction Model

This folder contains the core components for training, evaluating, and managing machine learning models for financial prediction tasks. It includes data preprocessing utilities, model management classes, and example usage in a Jupyter notebook.

## Contents

- **data_preprocess.py**: Implements the `DataPreprocessor` class for preparing and synchronizing features and targets, handling missing values, encoding labels, and splitting data into train/test sets.
- **model_manager.py**: Provides a unified interface for managing, training, testing, and comparing machine learning models. Includes:
  - `BaseModel`: Abstract base class for all models.
  - `RandomForestMateoModel`, `RandomForestMateoModel2`, `RandomForestModel`, `XGBoostModel`, `AdaBoostModel`: Implementations of various ML models with methods for training, prediction, and reporting.
  - `ModelManager`: Factory and utility class to instantiate models and prepare data.
- **model_cumulative_finance.ipynb**: Jupyter notebook demonstrating the workflow: loading data, preprocessing, training a Random Forest model, evaluating performance, and visualizing results.
- **__init__.py**: Empty file to mark the directory as a Python package.

## Usage Example

1. **Data Preparation**
   - Use `DataPreprocessor` to align and split your features and target data.

2. **Model Training & Evaluation**
   - Instantiate a model via `ModelManager.get_model('model_name', **params)`.
   - Train and evaluate using the model's `train_and_report` method.

3. **Notebook Example**
   - See `model_cumulative_finance.ipynb` for a full workflow example, including data loading, preprocessing, model training, evaluation, and visualization.

## Adding New Models
- Inherit from `BaseModel` and register your model class in `ModelManager.MODELS`.

## Requirements
- See the root `requirements.txt` for dependencies (scikit-learn, xgboost, pandas, numpy, etc.).

---
For more details, refer to the docstrings in each file and the notebook for practical examples.
