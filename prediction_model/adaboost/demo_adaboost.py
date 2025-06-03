import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Adapter l'import selon l'endroit où tu exécutes ce script
from prediction_model.adaboost.adaboost_model import CryptoAdaBoostClassifier

# Génère des données factices
np.random.seed(42)
n = 500
timestamps = np.arange(n) * 60.0 + 1704067200.0
X = pd.DataFrame({
    'timestamp': timestamps,
    'level-1-bid-price': np.random.normal(0.01, 0.005, n),
    'level-1-bid-volume': np.random.normal(10, 2, n),
    'level-1-ask-price': np.random.normal(0.02, 0.005, n),
    'level-1-ask-volume': np.random.normal(10, 2, n),
    'spread': np.random.normal(0.01, 0.002, n),
    'bid-ask-imbalance-5-levels': np.random.normal(0, 0.1, n),
    'vwap-bid-5-levels': np.random.normal(0.01, 0.005, n),
    'vwap-ask-5-levels': np.random.normal(0.02, 0.005, n)
})
y = pd.DataFrame({
    'timestamp': timestamps,
    'price': 50000 + np.cumsum(np.random.normal(0, 100, n)),
    'volume': np.random.exponential(10, n)
})

# Initialise et entraîne le modèle
clf = CryptoAdaBoostClassifier(n_estimators=10, max_depth=2, random_state=42)
results = clf.prepare_and_fit(X, y, test_size=0.2, validate=False)

print("Accuracy:", results['accuracy'])
print("Classification report:\n", results['classification_report'])

# Prédiction sur de nouvelles données (ici, les 5 premières lignes)
X_test = X.iloc[:50]
y_test = y.iloc[:50]
preds, probs = clf.predict(X_test, y_test)
print("Prédictions:", preds)
print("Probabilités:\n", probs)
