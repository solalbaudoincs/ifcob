"""
model_manager.py
================
Unified interface for managing, training, testing, and comparing ML models based on existing notebooks.

How to add a new model:
- Implement a new class inheriting from BaseModel, and register it in ModelManager.MODELS.

Features:
- Train a model on a dataset
- Test a model on a dataset
- Change hyperparameters
- Compare different hyperparameters
- Easily extendable for new models

Author: Your Name
Date: 2025-06-04
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from prediction_model.data_preprocess import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BaseModel:
    """
    Abstract base class for all models.
    """
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None
        self.preprocessor = DataPreprocessor()

    def train(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}

class RandomForestMateoModel(BaseModel):
    """
    Random Forest model as implemented in model_mateo_clean.ipynb.
    """
    DEFAULT_FEATURES = [
        "bid-ask-imbalance-5-levels",
        "spread",
        "inst-return",
        "V-bid-5-levels",
        "V-ask-5-levels",
        "slope-bid-5-levels",
        "slope-ask-5-levels"
    ]
    DEFAULT_TARGET = "return-all-signed-for-5-ms"
    DEFAULT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/XBT_EUR.parquet')
    DEFAULT_TARGET_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/ETH_EUR.parquet')
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../predictors/mateo/rf_model_5ms_clean.joblib')

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        self.feature_columns = hyperparams.get('feature_columns', self.DEFAULT_FEATURES)
        self.target_column = hyperparams.get('target_column', self.DEFAULT_TARGET)
        self.model = RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 3),
            class_weight=hyperparams.get('class_weight', 'balanced'),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=hyperparams.get('n_jobs', -1)
        )

    @classmethod
    def get_default_paths(cls):
        return cls.DEFAULT_FEATURES_PATH, cls.DEFAULT_TARGET_PATH, cls.DEFAULT_MODEL_PATH

    def train_and_report(self, X_train, X_test, y_train, y_test, save_path=None):
        # Train
        print("Entraînement du modèle Random Forest...")
        self.model.fit(X_train, y_train)
        print("Modèle Random Forest entraîné!")
        # Predict
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\n=== RÉSULTATS DU MODÈLE RANDOM FOREST ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Différence (overfitting): {train_acc - test_acc:.4f}")
        actual_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        print(f"\nClasses présentes: {actual_labels}")
        print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
        print(classification_report(y_test, y_pred, labels=actual_labels))
        print("\n=== CONFUSION MATRIX (TEST SET) ===")
        print(confusion_matrix(y_test, y_pred, labels=actual_labels))
        print("Classes dans y_train:", np.unique(y_train))
        print("Classes dans y_test:", np.unique(y_test))
        print("Classes dans y_pred:", np.unique(y_pred))
        # Feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n=== IMPORTANCE DES FEATURES ===")
        print(feature_importance)
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"\nModèle sauvegardé dans '{save_path}'")
        print("\n=== ANALYSE TERMINÉE ===")

class ModelManager:
    """
    Manages available models and provides unified interface.
    """
    MODELS = {
        'random_forest_mateo': RandomForestMateoModel,
        # Add new models here
    }

    @staticmethod
    def get_model(name: str, **kwargs) -> BaseModel:
        if name not in ModelManager.MODELS:
            raise ValueError(f"Unknown model: {name}")
        return ModelManager.MODELS[name](**kwargs)

    @staticmethod
    def prepare_data(features_path=None, target_path=None, feature_columns=None, target_column=None, test_size=0.2, random_state=42):
        if features_path is None or target_path is None:
            features_path, target_path, _ = RandomForestMateoModel.get_default_paths()
        features_df = pd.read_parquet(features_path)
        target_df = pd.read_parquet(target_path)
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            features_df=features_df,
            target_df=target_df,
            feature_columns=feature_columns or RandomForestMateoModel.DEFAULT_FEATURES,
            target_column=target_column or RandomForestMateoModel.DEFAULT_TARGET,
            test_size=test_size
        )
        return X_train, X_test, y_train, y_test
