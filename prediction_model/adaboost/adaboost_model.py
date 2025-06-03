# filepath: c:\Users\enzo.cAo\Documents\ST-Finance\EI LOB\ifcob\prediction_model\adaboost\adaboost_model.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from ..dataset.convert_to_target import AdaBoostDataPreprocessor

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib et/ou Seaborn non disponibles. Les fonctions de visualisation seront désactivées.")


class CryptoAdaBoostClassifier:
    """
    Modèle AdaBoost spécialisé pour la classification de signaux de trading crypto
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 1.0,
                 max_depth: int = 3,
                 random_state: int = 42):
        """
        Initialise le modèle AdaBoost
        
        Args:
            n_estimators: Nombre d'estimateurs faibles
            learning_rate: Taux d'apprentissage
            max_depth: Profondeur maximale des arbres de décision
            random_state: Graine aléatoire
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Créer l'estimateur de base
        base_estimator = DecisionTreeClassifier(
            max_depth=max_depth, 
            random_state=random_state
        )
        
        # Créer le modèle AdaBoost
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            algorithm='SAMME'  # Meilleur pour la classification multi-classe
        )
        
        # Préprocesseur de données
        self.preprocessor = AdaBoostDataPreprocessor()
        
        # Variables pour stocker les données d'entraînement
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_fitted = False
        
    def prepare_and_fit(self, 
                       input_df: pd.DataFrame, 
                       target_df: pd.DataFrame,
                       test_size: float = 0.2,
                       validate: bool = True,
                       features: list = None) -> Dict:
        """
        Prépare les données et entraîne le modèle
        
        Args:
            input_df: DataFrame avec les features
            target_df: DataFrame avec timestamp, price, volume
            test_size: Proportion des données pour le test
            validate: Si True, effectue une validation croisée
            features: Liste des features à sélectionner (optionnel)
        
        Returns:
            Dictionnaire avec les métriques de performance
        """
        print("Préparation des données...")
        X, y_encoded = self.preprocessor.prepare_data(input_df, target_df, features=features)
        
        print("Division train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded  # Préserver la distribution des classes
        )
        
        print(f"Entraînement du modèle AdaBoost avec {self.X_train.shape[0]} échantillons...")
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        
        # Évaluation
        results = self._evaluate_model(validate=validate)
        
        return results
    
    def _evaluate_model(self, validate: bool = True) -> Dict:
        """
        Évalue les performances du modèle
        """
        results = {}
        
        # Prédictions sur l'ensemble de test
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Métriques de base
        results['accuracy'] = accuracy_score(self.y_test, y_pred)
        results['classification_report'] = classification_report(
            self.y_test, y_pred, 
            target_names=self.preprocessor.label_encoder.classes_,
            output_dict=True
        )
        
        # Matrice de confusion
        results['confusion_matrix'] = confusion_matrix(self.y_test, y_pred)
        
        # Validation croisée
        if validate and hasattr(self, 'X_train'):
            cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train, 
                cv=5, scoring='accuracy'
            )
            results['cv_scores'] = cv_scores
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
        
        # Importance des features
        if hasattr(self.model, 'feature_importances_'):
            results['feature_importances'] = self.model.feature_importances_
        
        # Probabilités de prédiction pour analyse
        results['prediction_probabilities'] = y_pred_proba
        results['predictions'] = y_pred
        results['true_labels'] = self.y_test
        
        self._print_evaluation_summary(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict):
        """
        Affiche un résumé des résultats d'évaluation
        """
        print("\n" + "="*50)
        print("RÉSUMÉ DE L'ÉVALUATION DU MODÈLE ADABOOST")
        print("="*50)
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        if 'cv_mean' in results:
            print(f"Validation croisée: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        print("\nRapport de classification:")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  {class_name}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        print("\nMatrice de confusion:")
        print(results['confusion_matrix'])
    
    def predict(self, input_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fait des prédictions sur de nouvelles données
        
        Returns:
            Tuple (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Préparer les nouvelles données
        X, _ = self.preprocessor.prepare_data(input_df, target_df)
        
        # Prédictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_signals(self, input_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne les prédictions sous forme de signaux de trading lisibles
        """
        predictions, probabilities = self.predict(input_df, target_df)
        
        # Convertir les prédictions en signaux
        signal_mapping = self.preprocessor.get_label_mapping()
        signals = [signal_mapping[pred] for pred in predictions]
        
        # Créer le DataFrame de résultats
        results_df = pd.DataFrame({
            'signal_predicted': signals,
            'buy_probability': probabilities[:, signal_mapping.get(0, 0)],
            'hold_probability': probabilities[:, signal_mapping.get(1, 1)],
            'sell_probability': probabilities[:, signal_mapping.get(2, 2)]
        })
        
        return results_df
    
    def optimize_hyperparameters(self, 
                                input_df: pd.DataFrame, 
                                target_df: pd.DataFrame,
                                param_grid: Dict = None) -> Dict:
        """
        Optimise les hyperparamètres du modèle
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.5, 1.0, 1.5],
                'estimator__max_depth': [2, 3, 4]
            }
        
        print("Préparation des données pour l'optimisation...")
        X, y_encoded, _ = self.preprocessor.prepare_data(input_df, target_df)
        
        print("Recherche des meilleurs hyperparamètres...")
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y_encoded)
        
        # Mettre à jour le modèle avec les meilleurs paramètres
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Meilleurs paramètres: {results['best_params']}")
        print(f"Meilleur score: {results['best_score']:.4f}")
        
        return results
    def plot_feature_importance(self, feature_names: list = None, top_n: int = 15):
        """
        Affiche l'importance des features
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib non disponible. Impossible d'afficher le graphique.")
            return
            
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'afficher l'importance des features")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Trier par importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Importance des Features - AdaBoost Crypto Trading")
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, results: Dict):
        """
        Affiche la matrice de confusion
        """
        if not PLOTTING_AVAILABLE:
            print("Seaborn/Matplotlib non disponible. Impossible d'afficher le graphique.")
            return
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=self.preprocessor.label_encoder.classes_,
            yticklabels=self.preprocessor.label_encoder.classes_
        )
        plt.title('Matrice de Confusion - AdaBoost Crypto Trading')
        plt.ylabel('Vrai Label')
        plt.xlabel('Label Prédit')
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filepath: str):
        """
        Charge un modèle pré-entraîné
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Modèle chargé depuis {filepath}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer des données d'exemple
    np.random.seed(42)
    
    # DataFrame d'input avec features crypto
    input_data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min').astype(int) // 10**9,
        'bid_ask_spread': np.random.normal(0.01, 0.005, 1000),
        'bid_ask_imbalance': np.random.normal(0, 0.1, 1000),
        'depth_imbalance': np.random.normal(0, 0.2, 1000),
        'volatility': np.random.exponential(0.02, 1000),
        'order_flow': np.random.normal(0, 1, 1000),
        'volume_weighted_price': 50000 + np.random.normal(0, 1000, 1000)
    }
    input_df = pd.DataFrame(input_data)
    
    # DataFrame target avec prix et volume
    target_data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min').astype(int) // 10**9,
        'price': 50000 + np.cumsum(np.random.normal(0, 100, 1000)),
        'volume': np.random.exponential(10, 1000)
    }
    target_df = pd.DataFrame(target_data)
    
    # Créer et entraîner le modèle
    crypto_adaboost = CryptoAdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        max_depth=3
    )
    
    # Entraîner le modèle
    results = crypto_adaboost.prepare_and_fit(input_df, target_df)
    
    # Afficher les résultats
    crypto_adaboost.plot_confusion_matrix(results)
    
    # Faire des prédictions sur de nouvelles données
    new_predictions = crypto_adaboost.predict_signals(input_df.iloc[-100:], target_df.iloc[-100:])
    print("\nPrédictions sur les 100 derniers échantillons:")
    print(new_predictions.head())