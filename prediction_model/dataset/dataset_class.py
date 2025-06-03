from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class InputData:
    """Data structure for training features at a specific timestamp"""
    timestamp: float
    features: Dict[str, Any]  # Features of the first coin
    context: Dict[str, Any]   # Context data of Bitcoin


@dataclass
class LabelData:
    """Data structure for label information at a specific timestamp"""
    timestamp: float
    data: Dict[str, Any]  # Contains price, volume, etc.


@dataclass
class PredictionData:
    """Data structure for model predictions at a specific timestamp"""
    timestamp: float
    predicted_data: Dict[str, Any]  # Contains predicted price, volume, etc.


@dataclass
class DataStructure:
    """Main data structure containing training and label data"""
    training_data: List[InputData]
    label_data: List[LabelData]