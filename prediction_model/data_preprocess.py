# Preprocessing pour AdaBoost compatible avec le notebook adaboost_notebook.ipynb

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder



class DataPreprocessor:
    """
    Classe principale pour préparer les données pour AdaBoost
    Compatible avec la logique du notebook adaboost_notebook.ipynb
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, 
                    features_df: pd.DataFrame, 
                    target_df: pd.DataFrame,
                    feature_columns: list[str],
                    features_columns_target: list[str] = None,
                    target_column: str = "return-all-signed-for-5-ms",
                    test_size: float = 0.2,
                    target_lag: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:        
        """
        Prépare les données pour l'entraînement AdaBoost selon la méthode du notebook et effectue un split train/test.

        Args:
            features_df: DataFrame avec les features principales (ex: XBT data)
            target_df: DataFrame avec le target (ex: ETH data)  
            feature_columns: Liste des colonnes features à sélectionner depuis features_df
            features_columns_target: Liste des colonnes features à sélectionner depuis target_df (optionnel)
            target_column: Nom de la colonne target à utiliser ou convertir
            test_size: Proportion du jeu de test (float entre 0 et 1)
            target_lag: Décalage (en lignes) à appliquer sur les features du target_df (default=1)

        Returns:
            Tuple (X_train, X_test, y_train, y_test) où:
            - X_train: features combinées pour l'entraînement
            - X_test: features combinées pour le test
            - y_train: labels encodés pour l'entraînement
            - y_test: labels encodés pour le test
        """
        
        print("=== PRÉPARATION DES DONNÉES ADABOOST ===")
        
        # Étape 1: Extraction du target
        print(f"Étape 1: Extraction du target '{target_column}'...")
        target = target_df[target_column]
        print(f"Target shape: {target.shape}")
        
        # Étape 2: Sélection des features du features_df
        print("Étape 2: Sélection des features du features_df...")
        print(f"Features sélectionnées: {feature_columns}")
        pre_features = features_df[feature_columns]
        print(f"Features shape avant dropna: {pre_features.shape}")
        pre_features = pre_features.dropna()
        print(f"Features shape après dropna: {pre_features.shape}")
        # Étape 2b: Sélection des features du target_df (si spécifiées)
        target_features = None
        if features_columns_target is not None and len(features_columns_target) > 0:
            print("Étape 2b: Sélection des features du target_df avec lag...")
            print(f"Features target sélectionnées: {features_columns_target}")
            target_features = target_df[features_columns_target].shift(target_lag)
            print(f"Target features shape après shift (lag={target_lag}): {target_features.shape}")
            target_features = target_features.dropna()
            print(f"Target features shape après dropna: {target_features.shape}")
        
        # Étape 3: Alignement temporel (méthode exacte du notebook)
        print("Étape 3: Alignement temporel avec np.searchsorted...")
        target_timestamp = target.index.values
        feature_timestamp = pre_features.index.values
        filtered_indices = np.searchsorted(feature_timestamp, target_timestamp, side="left")
        
        # Filtrer les indices qui dépassent la taille du DataFrame des features
        filtered_indices = filtered_indices[filtered_indices < len(pre_features)]
        pre_features_filtered = pre_features.iloc[filtered_indices]
        
        print(f"Features après filtrage: {pre_features_filtered.shape}")
        print(f"Indices filtrés: {len(filtered_indices)}")
        
        # Étape 4: Création des DataFrames nettoyés avec les colonnes sélectionnées
        print("Étape 4: Création des DataFrames nettoyés...")
        data_clean = pre_features_filtered.reset_index(drop=True)
        
        # Si des features du target_df sont spécifiées, les ajouter après synchronisation
        if target_features is not None:
            print("Étape 4b: Ajout des features du target_df après synchronisation et lag...")
            # Synchroniser les target_features avec les mêmes indices que les features principales
            target_features_aligned = target_features.iloc[:len(filtered_indices)].reset_index(drop=True)
            print(f"Target features après alignement: {target_features_aligned.shape}")
            # Concaténer les features
            data_clean = pd.concat([data_clean, target_features_aligned], axis=1)
            print(f"Features combinées shape: {data_clean.shape}")
            print(f"Colonnes finales: {list(data_clean.columns)}")
        
        # target_clean contient le target aligné sur les mêmes indices
        target_clean = target.iloc[:len(filtered_indices)].reset_index(drop=True)

        # Drop des NaN sur les features et le target (alignement strict)
        combined = pd.concat([data_clean, target_clean], axis=1)
        combined = combined.dropna()
        data_clean = combined.iloc[:, :-1]
        target_clean = combined.iloc[:, -1]

        print(f"Données après dropna et filtrage NaN: {data_clean.shape}")
        print(f"Target après dropna et filtrage NaN: {target_clean.shape}")
        print("Distribution des classes:")
        print(target_clean.value_counts().sort_index())
        print("Proportions des classes:")
        print(target_clean.value_counts(normalize=True).sort_index())
        
        # Étape 5: Préparation finale
        print("Étape 5: Conversion en arrays numpy...")
        X = data_clean.values
        y_labels = target_clean.values
        
        # Encoder les labels  
        y_encoded = self.label_encoder.fit_transform(y_labels)
        
        print("\n=== RÉSULTAT FINAL ===")
        print(f"Features (X): {X.shape}")
        print(f"Labels encodés (y): {y_encoded.shape}")
        print(f"Mapping des labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Étape 6: Split train/test en gardant l'ordre chronologique
        print("Étape 6: Split train/test chronologique...")
        # Calcul de l'index de séparation
        split_index = int(len(X) * (1 - test_size))

        # Division manuelle en gardant l'ordre
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y_encoded[:split_index]
        y_test = y_encoded[split_index:]

        # Limitation du nombre de points si demandé (après split)
        if hasattr(self, 'n_samples_limit') and self.n_samples_limit is not None:
            n = self.n_samples_limit
            X_train = X_train[:n]
            X_test = X_test[:n]
            y_train = y_train[:n]
            y_test = y_test[:n]

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test
    
    def get_label_mapping(self) -> dict[int, str]:
        """
        Retourne le mapping entre les labels encodés et les signaux
        """
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}