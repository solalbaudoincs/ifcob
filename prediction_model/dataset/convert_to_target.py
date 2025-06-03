# filepath: c:\Users\enzo.cAo\Documents\ST-Finance\EI LOB\ifcob\prediction_model\dataset\convert_to_target.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TradingSignalConverter:
    """
    Convertit les données de prix/volume en signaux de trading (buy, sell, hold)
    """
    
    def __init__(self, 
                 price_threshold: float = 0.02,  # 2% de variation
                 volume_threshold: float = 1.5,   # 1.5x le volume moyen
                 lookforward_periods: int = 5):   # Nombre de périodes pour calculer le signal
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.lookforward_periods = lookforward_periods
        
    def _calculate_price_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule les signaux basés sur les variations de prix futures
        """
        # Calculer le prix futur (look-forward)
        future_price = df['price'].shift(-self.lookforward_periods)
        current_price = df['price']
        
        # Calculer le pourcentage de variation
        price_change = (future_price - current_price) / current_price
        
        # Générer les signaux
        signals = pd.Series('hold', index=df.index)
        signals[price_change > self.price_threshold] = 'buy'
        signals[price_change < -self.price_threshold] = 'sell'
        
        return signals
    
    def _calculate_volume_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule les signaux basés sur les variations de volume
        """
        # Volume moyen mobile
        volume_ma = df['volume'].rolling(window=20).mean()
        current_volume = df['volume']
        
        # Signaux basés sur le volume
        volume_ratio = current_volume / volume_ma
        
        signals = pd.Series('hold', index=df.index)
        signals[volume_ratio > self.volume_threshold] = 'buy'
        signals[volume_ratio < 1/self.volume_threshold] = 'sell'
        
        return signals
    
    def _combine_signals(self, price_signals: pd.Series, volume_signals: pd.Series) -> pd.Series:
        """
        Combine les signaux de prix et de volume
        """
        combined = pd.Series('hold', index=price_signals.index)
        
        # Si les deux signaux concordent, on les garde
        combined[(price_signals == 'buy') & (volume_signals == 'buy')] = 'buy'
        combined[(price_signals == 'sell') & (volume_signals == 'sell')] = 'sell'
        
        # Si un seul signal est fort, on le prend en compte avec moins de poids
        combined[(price_signals == 'buy') & (volume_signals == 'hold')] = 'buy'
        combined[(price_signals == 'sell') & (volume_signals == 'hold')] = 'sell'
        
        return combined
    
    def convert_to_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit un DataFrame avec timestamp, price, volume en signaux de trading
        """
        df = df.copy()
        
        # Calculer les différents types de signaux
        price_signals = self._calculate_price_signals(df)
        volume_signals = self._calculate_volume_signals(df)
        
        # Combiner les signaux
        final_signals = self._combine_signals(price_signals, volume_signals)
        
        # Créer le DataFrame de sortie
        result_df = df[['timestamp']].copy()
        result_df['signal'] = final_signals
        
        # Supprimer les dernières lignes où on ne peut pas calculer le signal futur
        result_df = result_df.iloc[:-self.lookforward_periods]
        
        return result_df


class DataSynchronizer:
    """
    Synchronise les DataFrames asynchrones en utilisant merge_asof
    """
    
    def __init__(self, forward_tolerance: float = 60.0, backward_tolerance: float = 30.0):
        # Ici, forward_tolerance et backward_tolerance sont des floats (secondes, par ex.)
        self.forward_tolerance = forward_tolerance
        self.backward_tolerance = backward_tolerance
    
    def synchronize_dataframes(self, 
                             input_df: pd.DataFrame, 
                             target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Synchronise les DataFrames input et target en utilisant merge_asof
        
        Args:
            input_df: DataFrame avec les features (doit contenir 'timestamp')
            target_df: DataFrame avec les labels (doit contenir 'timestamp', 'price', 'volume')
        
        Returns:
            Tuple des DataFrames synchronisés
        """
        
        # Merge asof pour prendre les valeurs futures du target (forward)
        # On inverse l'ordre pour avoir un merge forward
        
        # 1) merge_asof en plaçant target_df à gauche
        synchronized_df = pd.merge_asof(
            left=target_df,
            right=input_df,
            on='timestamp',
            direction='backward',
            tolerance=self.backward_tolerance
        )
        
        # Séparer les colonnes en fonction de l’origine
        input_columns = [col for col in input_df.columns if col in synchronized_df.columns]
        target_columns = [col for col in target_df.columns if col in synchronized_df.columns]

        # Reconstruire les DataFrames synchronisés
        sync_input = synchronized_df[input_columns]
        sync_target = synchronized_df[target_columns]
        
        return sync_input, sync_target


class FeatureSelector:
    """
    Sélectionne les features importantes du DataFrame d'input
    """
    
    def __init__(self, important_features: list = None):
        if important_features is None:
            # Liste réelle des features importantes pour les order books crypto
            self.important_features = [
                'level-1-bid-price', 'level-1-bid-volume', 'level-2-bid-price', 'level-2-bid-volume',
                'level-3-bid-price', 'level-3-bid-volume', 'level-4-bid-price', 'level-4-bid-volume',
                'level-5-bid-price', 'level-5-bid-volume', 'level-6-bid-price', 'level-6-bid-volume',
                'level-7-bid-price', 'level-7-bid-volume', 'level-8-bid-price', 'level-8-bid-volume',
                'level-9-bid-price', 'level-9-bid-volume', 'level-10-bid-price', 'level-10-bid-volume',
                'level-1-ask-price', 'level-1-ask-volume', 'level-2-ask-price', 'level-2-ask-volume',
                'level-3-ask-price', 'level-3-ask-volume', 'level-4-ask-price', 'level-4-ask-volume',
                'level-5-ask-price', 'level-5-ask-volume', 'level-6-ask-price', 'level-6-ask-volume',
                'level-7-ask-price', 'level-7-ask-volume', 'level-8-ask-price', 'level-8-ask-volume',
                'level-9-ask-price', 'level-9-ask-volume', 'level-10-ask-price', 'level-10-ask-volume',
                'V-bid-5-levels', 'V-ask-5-levels', 'bid-ask-imbalance-5-levels', 'spread',
                'slope-bid-5-levels', 'slope-ask-5-levels', 'vwap-bid-5-levels', 'vwap-ask-5-levels',
                'avg-vwap-diff-5-levels', 'liquidity-ratio', 'rate-inst-volatility', 'rate-momentum',
                'rate-mid-price-trend', 'rate-vwap-diff-5-levels', 'rate-bid-volume-level-1', 'rate-ask-volume-level-1'
            ]
        else:
            self.important_features = important_features

    def select_features(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """
        Sélectionne les features importantes du DataFrame.
        Si 'features' est fourni, utilise cette liste, sinon utilise les features importantes par défaut.
        """
        if features is not None:
            available_features = [col for col in features if col in df.columns]
        else:
            available_features = [col for col in self.important_features if col in df.columns]
        
        if not available_features:
            print("Aucune feature importante trouvée. Utilisation de toutes les colonnes numériques.")
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in available_features:
                available_features.remove('timestamp')
        
        # Garder timestamp + features sélectionnées
        selected_columns = ['timestamp'] + available_features
        return df[selected_columns]


class AdaBoostDataPreprocessor:
    """
    Classe principale pour préparer les données pour AdaBoost
    """
    
    def __init__(self, 
                 feature_selector: FeatureSelector = None,
                 data_synchronizer: DataSynchronizer = None,
                 signal_converter: TradingSignalConverter = None):
        
        self.feature_selector = feature_selector or FeatureSelector()
        self.data_synchronizer = data_synchronizer or DataSynchronizer()
        self.signal_converter = signal_converter or TradingSignalConverter()
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        
    def prepare_data(self, 
                    input_df: pd.DataFrame, 
                    target_df: pd.DataFrame,
                    features: list = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement AdaBoost

        Args:
            input_df: DataFrame avec les features
            target_df: DataFrame avec timestamp, price, volume, etc.
            features: Liste des features à sélectionner (optionnel)

        Returns:
            Tuple (X, y_encoded, y_onehot) où:
            - X: features pour l'entraînement
            - y_encoded: labels encodés (0, 1, 2 pour buy, hold, sell)
            - y_onehot: labels en one-hot encoding
        """
        print("Étape 1: Sélection des features importantes...")
        selected_input = self.feature_selector.select_features(input_df, features=features)
        
        print("Étape 2: Synchronisation des DataFrames...")
        sync_input, sync_target = self.data_synchronizer.synchronize_dataframes(
            selected_input, target_df
        )
        
        print("Étape 3: Conversion en signaux de trading...")
        signals_df = self.signal_converter.convert_to_signals(sync_target)
        
        print("Étape 4: Alignement final des données...")
        # Aligner les timestamps entre input synchronisé et signaux
        final_df = pd.merge(sync_input, signals_df, on='timestamp', how='inner')
        
        # Préparer les features (X)
        feature_columns = [col for col in final_df.columns if col not in ['timestamp', 'signal']]
        X = final_df[feature_columns].values
        
        # Préparer les labels (y)
        y_labels = final_df['signal'].values
        
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y_labels)
        
        print(f"Données finales: {X.shape[0]} échantillons, {X.shape[1]} features")
        print(f"Distribution des signaux: {pd.Series(y_labels).value_counts().to_dict()}")
        
        return X, y_encoded
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Retourne le mapping entre les labels encodés et les signaux
        """
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer des données d'exemple
    np.random.seed(42)
    
    # DataFrame d'input avec features
    input_data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min').astype(int) // 10**9,
        'bid_ask_spread': np.random.normal(0.01, 0.005, 1000),
        'bid_ask_imbalance': np.random.normal(0, 0.1, 1000),
        'depth_imbalance': np.random.normal(0, 0.2, 1000),
        'volatility': np.random.exponential(0.02, 1000)
    }
    input_df = pd.DataFrame(input_data)
    
    # DataFrame target avec prix et volume
    target_data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min').astype(int) // 10**9,
        'price': 50000 + np.cumsum(np.random.normal(0, 100, 1000)),
        'volume': np.random.exponential(10, 1000)
    }
    target_df = pd.DataFrame(target_data)
    
    # Préparer les données
    preprocessor = AdaBoostDataPreprocessor()
    X, y_encoded, y_onehot = preprocessor.prepare_data(input_df, target_df)
    
    print(f"Shape des features: {X.shape}")
    print(f"Shape des labels encodés: {y_encoded.shape}")
    print(f"Shape des labels one-hot: {y_onehot.shape}")
    print(f"Mapping des labels: {preprocessor.get_label_mapping()}")