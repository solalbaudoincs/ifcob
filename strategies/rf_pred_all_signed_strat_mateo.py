from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import joblib
import pandas as pd

class RFPredAllSignedStratMateo(Strategy):
    """
    Strategy that uses the RF predictions to make trades.
    It uses the predictions of all assets to make trades.
    It is a signed strategy, meaning that it takes into account the sign of the prediction.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"predictors/mateo/rf_model_{window_size}ms.joblib")
        self.target_eth = 10.0


    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        # Implement the logic for making trades based on RF predictions
        """features = pd.DataFrame()
        features["V-bid-5-levels"] = sum([rows[f"level-{i}-bid-volume"] for i in range(1, 6)])
        features["V-ask-5-levels"] = sum([rows[f"level-{i}-ask-volume"] for i in range(1, 6)])
        features["bid-ask-imbalance-5-levels"] = features["V-bid-5-levels"] - features["V-ask-5-levels"]
        features["spread"] = rows[f"level-1-ask-price"] - rows[f"level-1-bid-price"]
        mid_price = (rows[f"level-1-ask-price"] + rows[f"level-1-bid-price"]) / 2
        features["inst-return"] = mid_price.diff()/rows.index.diff()
        features["slope-bid-5-levels"] = (rows["level-5-bid-price"] - rows["level-1-bid-price"])/ features["V-bid-5-levels"]
        features["slope-ask-5-levels"] = (rows["level-5-ask-price"] - rows["level-1-ask-price"])/ features["V-ask-5-levels"]"""
        # Reorder features columns to match the model's expected input
        entry = data["XBT"][["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]].iloc[[-1]]
        prediction = self.model.predict(entry)[0]
        if prediction == -1:
            # sell signal
            return {"ETH" : -0.1}
        elif prediction == 0:
            # Hold signal
            return {"ETH": self.target_eth - current_portfolio.get_position("ETH")}
        elif prediction == 1:
            # buy signal
            return {"ETH": 0.1}
        else:
            raise ValueError(f"Unexpected prediction value: {prediction}. Expected -1, 0, or 1.")