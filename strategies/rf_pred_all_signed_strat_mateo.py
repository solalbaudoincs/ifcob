from ..backtesting.strategy import Strategy
from ..backtesting.types import MarketData, Coin, Action, FeesGraph
from ..backtesting.portolio import Portfolio
import joblib

class RFPredAllSignedStratMateo(Strategy):
    """
    Strategy that uses the RF predictions to make trades.
    It uses the predictions of all assets to make trades.
    It is a signed strategy, meaning that it takes into account the sign of the prediction.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"../predictors/mateo/rf_model_{window_size}ms.joblib")
        self.target_eth = 10.0


    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        # Implement the logic for making trades based on RF predictions
        last_signal = data["XBT"].iloc[-1]
        prediction = self.model.predict([last_signal])[0]
        if prediction == -1:
            # sell signal
            return {"ETH" : -0.1}
        elif prediction == 0 and current_portfolio["ETH"] < 9.9:
            # Hold signal
            return {"ETH": 0.01}
        elif prediction == 0 and current_portfolio["ETH"] > 10.1:
            # Hold signal
            return {"ETH": -0.01}
        elif prediction == 1:
            # buy signal
            return {"ETH": 0.1}