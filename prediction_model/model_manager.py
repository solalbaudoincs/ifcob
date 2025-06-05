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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

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
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import os
        import json
        # Génération d'un nom de dossier basé sur target_column, max_depth, n_estimators
        def get_model_id():
            import re
            params = self.model.get_params() if hasattr(self.model, 'get_params') else getattr(self, 'hyperparams', {})
            target = getattr(self, 'target_column', None)
            n_estimators = params.get('n_estimators', None)
            max_depth = params.get('max_depth', None)
            def clean(val):
                if val is None:
                    return 'NA'
                return re.sub(r'[^a-zA-Z0-9_]', '_', str(val))
            parts = [
                f"target-{clean(target)}" if target else None,
                f"depth-{max_depth}" if max_depth is not None else None,
                f"nest-{n_estimators}" if n_estimators is not None else None
            ]
            parts = [p for p in parts if p]
            return "_".join(parts)
        # Si save_path n'est pas précisé, on génère un chemin par défaut
        if not save_path:
            model_id = get_model_id()
            model_dir = os.path.join('predictors', 'mateo', model_id)
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, 'model.joblib')
        else:
            # Si save_path est un .joblib, on prend le nom du dossier parent et on ajoute le model_id
            base_dir = os.path.dirname(save_path)
            model_id = get_model_id()
            model_dir = os.path.join(base_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, 'model.joblib')
        self.model.fit(X_train, y_train)
        print("Modèle Random Forest entraîné!")
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\n=== RÉSULTATS DU MODÈLE RANDOM FOREST ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Différence (overfitting): {train_acc - test_acc:.4f}")
        actual_labels = sorted(set(list(y_test) + list(y_pred)))
        print(f"\nClasses présentes: {actual_labels}")
        print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
        print(classification_report(y_test, y_pred, labels=actual_labels))
        print("\n=== CONFUSION MATRIX (TEST SET) ===")
        print(confusion_matrix(y_test, y_pred, labels=actual_labels))
        print("Classes dans y_train:", set(y_train))
        print("Classes dans y_test:", set(y_test))
        print("Classes dans y_pred:", set(y_pred))
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n=== IMPORTANCE DES FEATURES ===")
        print(feature_importance)
        joblib.dump(self.model, save_path)
        print(f"\nModèle sauvegardé dans '{save_path}'")
        # Récupération des hyperparamètres explicites pour la sauvegarde des perfs
        params = dict(getattr(self, 'hyperparams', {}))
        if hasattr(self, 'model') and hasattr(self.model, 'get_params'):
            params.update(self.model.get_params())
        if hasattr(self, 'feature_columns'):
            params['feature_columns'] = self.feature_columns
        if hasattr(self, 'target_column'):
            params['target_column'] = self.target_column
        # Nettoyage des valeurs None ou NaN
        import math
        def is_valid(v):
            if v is None:
                return False
            if isinstance(v, float) and (math.isnan(v) or v is None):
                return False
            return True
        params = {k: v for k, v in params.items() if is_valid(v)}
        # Sauvegarde des performances dans le même dossier
        perf_data = {
            'model': 'random_forest_mateo',
            'hyperparameters': params,
            'performance': {
                'accuracy': test_acc,
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred, labels=actual_labels).tolist()
            }
        }
        perf_file = os.path.join(model_dir, 'perf.json')
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f, indent=2)
        print(f"Performance sauvegardée dans '{perf_file}'")
        print("\n=== ANALYSE TERMINÉE ===")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    """
    XGBoost model basé sur le notebook Xgboost_notebook.ipynb.
    """
    DEFAULT_FEATURES = [
        "V-bid-5-levels",
        "V-ask-5-levels",
        "slope-bid-5-levels",
        "slope-ask-5-levels"
    ]
    DEFAULT_TARGET = "return-vs-volatility-5-ms"
    DEFAULT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/XBT_EUR.parquet')
    DEFAULT_TARGET_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/ETH_EUR.parquet')
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../predictors/xgboost/xgb_model_5ms_clean.joblib')

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        self.feature_columns = hyperparams.get('feature_columns', self.DEFAULT_FEATURES)
        self.target_column = hyperparams.get('target_column', self.DEFAULT_TARGET)
        self.model = xgb.XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 150),
            max_depth=hyperparams.get('max_depth', 5),
            learning_rate=hyperparams.get('learning_rate', 0.15),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            random_state=hyperparams.get('random_state', 42),
            eval_metric=hyperparams.get('eval_metric', 'logloss'),
            use_label_encoder=False,
            n_jobs=-1
        )

    @classmethod
    def get_default_paths(cls):
        return cls.DEFAULT_FEATURES_PATH, cls.DEFAULT_TARGET_PATH, cls.DEFAULT_MODEL_PATH

    def train_and_report(self, X_train, X_test, y_train, y_test, save_path=None):
        print("Entraînement du modèle XGBoost...")
        # Calcul des poids d'échantillons pour équilibrer les classes
        from sklearn.utils.class_weight import compute_class_weight
        unique = sorted(set(y_train))
        class_weights = compute_class_weight('balanced', classes=np.array(unique), y=y_train)
        class_weight_dict = dict(zip(unique, class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        self.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=True)
        print("Modèle XGBoost entraîné!")
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print("\n=== RÉSULTATS DU MODÈLE XGBOOST ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Différence (overfitting): {train_acc - test_acc:.4f}")
        print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
        print(classification_report(y_test, y_pred))
        print("\n=== CONFUSION MATRIX (TEST SET) ===")
        print(confusion_matrix(y_test, y_pred))
        # Feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            n_features = len(importances)
            # Robust feature names: use self.feature_columns if correct length, else generic names
            if hasattr(self, 'feature_columns') and self.feature_columns is not None and len(self.feature_columns) == n_features:
                feature_names = self.feature_columns
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            print("\n=== IMPORTANCE DES FEATURES ===")
            print(feature_importance)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"\nModèle sauvegardé dans '{save_path}'")
        print("\n=== ANALYSE TERMINÉE ===")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class AdaBoostModel(BaseModel):
    """
    AdaBoost model basé sur le notebook adaboost_notebook_clean.ipynb.
    """
    DEFAULT_FEATURES = [
        "V-bid-5-levels",
        "V-ask-5-levels",
        "slope-bid-5-levels",
        "slope-ask-5-levels"
    ]
    DEFAULT_TARGET = "return-all-signed-for-5-ms"
    DEFAULT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/XBT_EUR.parquet')
    DEFAULT_TARGET_PATH = os.path.join(os.path.dirname(__file__), '../data/features/DATA_0/ETH_EUR.parquet')
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../predictors/adaboost/ada_model_5ms_clean.joblib')

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        self.feature_columns = hyperparams.get('feature_columns', self.DEFAULT_FEATURES)
        self.target_column = hyperparams.get('target_column', self.DEFAULT_TARGET)
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=hyperparams.get('max_depth', 3)),
            n_estimators=hyperparams.get('n_estimators', 100),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            random_state=hyperparams.get('random_state', 42)
        )

    @classmethod
    def get_default_paths(cls):
        return cls.DEFAULT_FEATURES_PATH, cls.DEFAULT_TARGET_PATH, cls.DEFAULT_MODEL_PATH

    def train_and_report(self, X_train, X_test, y_train, y_test, save_path=None):
        print("Entraînement du modèle AdaBoost...")
        from sklearn.utils.class_weight import compute_class_weight
        unique = sorted(set(y_train))
        class_weights = compute_class_weight('balanced', classes=np.array(unique), y=y_train)
        sample_weights = np.array([class_weights[label] for label in y_train])
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        print("Modèle AdaBoost entraîné!")
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification report (test):")
        print(classification_report(y_test, y_pred))
        print("\nConfusion matrix (test):")
        print(confusion_matrix(y_test, y_pred))
        # Feature importances
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n=== IMPORTANCE DES FEATURES ===")
            print(feature_importance)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"\nModèle sauvegardé dans '{save_path}'")
        print("\n=== ANALYSE TERMINÉE ===")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class ModelManager:
    """
    Manages available models and provides unified interface.
    """
    MODELS = {
        'random_forest_mateo': RandomForestMateoModel,
        'xgboost': XGBoostModel,
        'adaboost': AdaBoostModel,
        # Add new models here
    }

    @staticmethod
    def get_model(name: str, **kwargs) -> BaseModel:
        if name not in ModelManager.MODELS:
            raise ValueError(f"Unknown model: {name}")
        return ModelManager.MODELS[name](**kwargs)

    @staticmethod
    def prepare_data(features_path=None, target_path=None, feature_columns=None, target_column=None, test_size=0.2, random_state=42, n_samples=None):
        if features_path is None or target_path is None:
            features_path, target_path, _ = RandomForestMateoModel.get_default_paths()
        features_df = pd.read_parquet(features_path)
        target_df = pd.read_parquet(target_path)
        preprocessor = DataPreprocessor()
        if n_samples is not None:
            preprocessor.n_samples_limit = n_samples
        else:
            preprocessor.n_samples_limit = None
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            features_df=features_df,
            target_df=target_df,
            feature_columns=feature_columns or RandomForestMateoModel.DEFAULT_FEATURES,
            target_column=target_column or RandomForestMateoModel.DEFAULT_TARGET,
            test_size=test_size
        )
        return X_train, X_test, y_train, y_test
