from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
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
        last_signal = data["XBT"].iloc[-1][["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]]
        prediction = self.model.predict([last_signal])[0]
        if prediction == -1:
            # sell signal
            return {"ETH" : -0.1}
        elif prediction == 0:
            # Hold signal
            return {"ETH": self.target_eth - current_portfolio["ETH"]}
        elif prediction == 1:
            # buy signal
            return {"ETH": 0.1}