# filepath: c:\Users\enzo.cAo\Documents\ST-Finance\EI LOB\ifcob\prediction_model\adaboost\adaboost_model.py

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import data structures from dataset module
from ..dataset.dataset_class import InputData, LabelData, PredictionData, DataStructure


class AdaBoostModel:
    """AdaBoost model for cryptocurrency price prediction"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 1.0, 
                 random_state: int = 42, max_depth: int = 3):
        """
        Initialize the AdaBoost model
        
        Args:
            n_estimators: Number of weak learners
            learning_rate: Learning rate
            random_state: Random state for reproducibility
            max_depth: Maximum depth of decision trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_depth = max_depth
        
        # Initialize separate models for different prediction targets
        self.price_model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
        self.volume_model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
        self.feature_names = []
        self.is_trained = False
        
    def _extract_features(self, training_data_list: List[InputData]) -> np.ndarray:
        """
        Extract features from training data into a numpy array
        
        Args:
            training_data_list: List of InputData objects
            
        Returns:
            2D numpy array with features
        """
        if not training_data_list:
            raise ValueError("Training data list is empty")
        
        # Get all possible feature names from first sample
        sample = training_data_list[0]
        feature_names = []
        
        # Add features from the features dict
        for key in sample.features.keys():
            if isinstance(sample.features[key], (int, float)):
                feature_names.append(f"features_{key}")
            elif isinstance(sample.features[key], dict):
                for subkey in sample.features[key].keys():
                    if isinstance(sample.features[key][subkey], (int, float)):
                        feature_names.append(f"features_{key}_{subkey}")
        
        # Add features from the context dict
        for key in sample.context.keys():
            if isinstance(sample.context[key], (int, float)):
                feature_names.append(f"context_{key}")
            elif isinstance(sample.context[key], dict):
                for subkey in sample.context[key].keys():
                    if isinstance(sample.context[key][subkey], (int, float)):
                        feature_names.append(f"context_{key}_{subkey}")
        
        self.feature_names = feature_names
        
        # Extract features for all samples
        features_matrix = []
        for data in training_data_list:
            row = []
            
            # Extract from features
            for key in sample.features.keys():
                if isinstance(data.features[key], (int, float)):
                    row.append(float(data.features[key]))
                elif isinstance(data.features[key], dict):
                    for subkey in sample.features[key].keys():
                        if isinstance(data.features[key][subkey], (int, float)):
                            row.append(float(data.features[key][subkey]))
            
            # Extract from context
            for key in sample.context.keys():
                if isinstance(data.context[key], (int, float)):
                    row.append(float(data.context[key]))
                elif isinstance(data.context[key], dict):
                    for subkey in sample.context[key].keys():
                        if isinstance(data.context[key][subkey], (int, float)):
                            row.append(float(data.context[key][subkey]))
            
            features_matrix.append(row)
        
        return np.array(features_matrix)
    
    def _extract_labels(self, label_data_list: List[LabelData]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract labels from label data
        
        Args:
            label_data_list: List of LabelData objects
            
        Returns:
            Tuple of (prices, volumes) arrays
        """
        prices = []
        volumes = []
        
        for label in label_data_list:
            # Extract price (assuming it's in the data dict)
            price = label.data.get('price', 0.0)
            volume = label.data.get('volume', 0.0)
            
            prices.append(float(price))
            volumes.append(float(volume))
        
        return np.array(prices), np.array(volumes)
    
    def train(self, data_structure: DataStructure) -> None:
        """
        Train the AdaBoost model
        
        Args:
            data_structure: DataStructure containing training and label data
        """
        print("Extracting features from training data...")
        X = self._extract_features(data_structure.training_data)
        
        print("Extracting labels from label data...")
        prices, volumes = self._extract_labels(data_structure.label_data)
        
        # Ensure we have the same number of samples
        min_samples = min(len(X), len(prices), len(volumes))
        X = X[:min_samples]
        prices = prices[:min_samples]
        volumes = volumes[:min_samples]
        
        print(f"Training on {min_samples} samples with {X.shape[1]} features...")
        
        # Train models
        print("Training price prediction model...")
        self.price_model.fit(X, prices)
        
        print("Training volume prediction model...")
        self.volume_model.fit(X, volumes)
        
        self.is_trained = True
        print("Training completed successfully!")
    
    def predict(self, data_structure: DataStructure) -> List[PredictionData]:
        """
        Make predictions using the trained model
        
        Args:
            data_structure: DataStructure containing training data for prediction
            
        Returns:
            List of PredictionData objects with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Making predictions...")
        X = self._extract_features(data_structure.training_data)
        
        # Make predictions
        predicted_prices = self.price_model.predict(X)
        predicted_volumes = self.volume_model.predict(X)
        
        # Create prediction data objects
        predictions = []
        for i, training_data in enumerate(data_structure.training_data):
            prediction = PredictionData(
                timestamp=training_data.timestamp,
                predicted_data={
                    'price': predicted_prices[i],
                    'volume': predicted_volumes[i]
                }
            )
            predictions.append(prediction)
        
        print(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _find_closest_higher_timestamp(self, target_timestamp: float, 
                                     label_data: List[LabelData]) -> Optional[LabelData]:
        """
        Find the closest higher timestamp in label data
        
        Args:
            target_timestamp: Target timestamp to find
            label_data: List of LabelData objects
            
        Returns:
            LabelData object with closest higher timestamp or None
        """
        # Sort label data by timestamp
        sorted_labels = sorted(label_data, key=lambda x: x.timestamp)
        
        # Find closest higher timestamp
        for label in sorted_labels:
            if label.timestamp >= target_timestamp:
                return label
        
        return None
    
    def evaluate(self, predictions: List[PredictionData], 
                 label_data: List[LabelData]) -> Dict[str, float]:
        """
        Evaluate model predictions against label data
        
        Args:
            predictions: List of PredictionData objects
            label_data: List of LabelData objects
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model performance...")
        
        matched_predictions = []
        matched_labels = []
        
        # Match predictions with closest higher timestamps in labels
        for prediction in predictions:
            closest_label = self._find_closest_higher_timestamp(
                prediction.timestamp, label_data
            )
            
            if closest_label is not None:
                matched_predictions.append(prediction)
                matched_labels.append(closest_label)
        
        print(f"Matched {len(matched_predictions)} predictions with labels")
        
        if not matched_predictions:
            raise ValueError("No predictions could be matched with labels")
          # Extract predicted and actual values
        pred_prices = [pred.predicted_data.get('price', 0) for pred in matched_predictions]
        actual_prices = [label.data.get('price', 0) for label in matched_labels]
        
        pred_volumes = [pred.predicted_data.get('volume', 0) for pred in matched_predictions]
        actual_volumes = [label.data.get('volume', 0) for label in matched_labels]
        
        # Calculate metrics
        price_mse = mean_squared_error(actual_prices, pred_prices)
        price_mae = mean_absolute_error(actual_prices, pred_prices)
        price_r2 = r2_score(actual_prices, pred_prices)
        
        volume_mse = mean_squared_error(actual_volumes, pred_volumes)
        volume_mae = mean_absolute_error(actual_volumes, pred_volumes)
        volume_r2 = r2_score(actual_volumes, pred_volumes)
        
        metrics = {
            'price_mse': price_mse,
            'price_mae': price_mae,
            'price_r2': price_r2,
            'price_rmse': np.sqrt(price_mse),
            'volume_mse': volume_mse,
            'volume_mae': volume_mae,
            'volume_r2': volume_r2,
            'volume_rmse': np.sqrt(volume_mse),
            'matched_samples': len(matched_predictions)
        }
        
        print("Evaluation completed!")
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Tuple[float, float]]:
        """
        Get feature importance from trained models
        
        Returns:
            Dictionary with feature names and importance scores (price_model, volume_model)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        price_importance = self.price_model.feature_importances_
        volume_importance = self.volume_model.feature_importances_
        
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_dict[feature_name] = (price_importance[i], volume_importance[i])
        
        return importance_dict


# Example usage and helper functions