from ..backtesting.strategy import Strategy
import joblib

class RFPredAllSignedStratMateo(Strategy):
    """
    Strategy that uses the RF predictions to make trades.
    It uses the predictions of all assets to make trades.
    It is a signed strategy, meaning that it takes into account the sign of the prediction.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"../data/models/mateo/rf_model_{window_size}ms.joblib")


    def get_action(self, data, current_portfolio, fees):
        # Implement the logic for making trades based on RF predictions

        pass  # Replace with actual implementation