import pandas as pd
import numpy as np
import sys
import os

# Pour permettre l'import du modèle depuis le dossier adaboost
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prediction_model.adaboost.adaboost_model import CryptoAdaBoostClassifier

DATA_VERSION = "DATA_0"
FEATURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/features/{DATA_VERSION}'))
TARGETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/features/{DATA_VERSION}'))

PAIRS = ["XBT_EUR", "ETH_EUR"]

for pair in PAIRS:
    print(f"\n=== Entraînement pour {pair} ===")
    features_path = os.path.join(FEATURES_DIR, f"{pair}.parquet")
    targets_path = os.path.join(TARGETS_DIR, f"{pair}.parquet")

    if not os.path.exists(features_path):
        print(f"Fichier features introuvable: {features_path}")
        continue
    if not os.path.exists(targets_path):
        print(f"Fichier targets introuvable: {targets_path}")
        continue

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(targets_path)

    # Si l'index s'appelle 'timestamp', on le remet en colonne
    if X.index.name == 'timestamp' or X.index.name is not None:
        X = X.reset_index()
    if y.index.name == 'timestamp' or y.index.name is not None:
        y = y.reset_index()

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # Liste des features à utiliser (hors timestamp)
    features = [col for col in X.columns if col != 'timestamp']

    clf = CryptoAdaBoostClassifier(n_estimators=20, max_depth=3, random_state=42)
    results = clf.prepare_and_fit(X, y, test_size=0.2, validate=False, features=features)

    print("Accuracy:", results['accuracy'])
    print("Classification report:\n", results['classification_report'])
