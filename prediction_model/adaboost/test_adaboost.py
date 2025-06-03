import pytest
import numpy as np
from .adaboost_model import AdaBoostModel
from ..dataset.dataset_class import InputData, LabelData, DataStructure

@pytest.fixture
def sample_data():
    """Create sample data for testing purposes"""
    training_data = []
    label_data = []
    for i in range(100):
        timestamp = 1000.0 + i * 10.0
        training = InputData(
            timestamp=timestamp,
            features={
                'price_feature': 50000.0 + np.random.normal(0, 1000),
                'volume_feature': 1000.0 + np.random.normal(0, 100),
                'technical_indicators': {
                    'rsi': np.random.uniform(20, 80),
                    'macd': np.random.normal(0, 10)
                }
            },
            context={
                'btc_price': 60000.0 + np.random.normal(0, 2000),
                'market_sentiment': np.random.uniform(0, 1),
                'macro_indicators': {
                    'vix': np.random.uniform(10, 30),
                    'dxy': np.random.uniform(90, 110)
                }
            }
        )
        training_data.append(training)
        label = LabelData(
            timestamp=timestamp + 5.0,
            data={
                'price': 50000.0 + np.random.normal(0, 1000),
                'volume': 1000.0 + np.random.normal(0, 100),
                'bid_ask_spread': np.random.uniform(0.1, 1.0)
            }
        )
        label_data.append(label)
    return DataStructure(training_data=training_data, label_data=label_data)

def test_adaboost_model_train_predict_evaluate(sample_data):
    model = AdaBoostModel(n_estimators=10, learning_rate=0.5, max_depth=2)
    model.train(sample_data)
    assert model.is_trained
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data.training_data)
    metrics = model.evaluate(predictions, sample_data.label_data)
    for key in [
        'price_mse', 'price_mae', 'price_r2', 'price_rmse',
        'volume_mse', 'volume_mae', 'volume_r2', 'volume_rmse', 'matched_samples']:
        assert key in metrics
    assert isinstance(metrics['price_mse'], float)
    assert isinstance(metrics['volume_mse'], float)
    assert metrics['matched_samples'] > 0

def test_feature_importance_shape(sample_data):
    model = AdaBoostModel(n_estimators=5, learning_rate=1.0, max_depth=1)
    model.train(sample_data)
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == len(model.feature_names)
    for v in importance.values():
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert all(isinstance(x, float) for x in v)

def test_prediction_value_types(sample_data):
    model = AdaBoostModel(n_estimators=3, learning_rate=1.0, max_depth=1)
    model.train(sample_data)
    predictions = model.predict(sample_data)
    for pred in predictions:
        assert isinstance(pred.timestamp, float)
        assert 'price' in pred.predicted_data
        assert 'volume' in pred.predicted_data
        assert isinstance(pred.predicted_data['price'], float)
        assert isinstance(pred.predicted_data['volume'], float)

def test_error_on_untrained_predict(sample_data):
    model = AdaBoostModel()
    with pytest.raises(ValueError, match="Model must be trained before making predictions"):
        model.predict(sample_data)

def test_error_on_untrained_feature_importance(sample_data):
    model = AdaBoostModel()
    with pytest.raises(ValueError, match="Model must be trained before getting feature importance"):
        model.get_feature_importance()

def test_error_on_empty_training():
    model = AdaBoostModel()
    from ..dataset.dataset_class import DataStructure
    empty_data = DataStructure(training_data=[], label_data=[])
    with pytest.raises(ValueError, match="Training data list is empty"):
        model.train(empty_data)
