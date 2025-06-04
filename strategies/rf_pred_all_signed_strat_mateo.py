from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import joblib
import pandas as pd
import time

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
        start_time = time.time()
        # Reorder features columns to match the model's expected input
        entry = data["XBT"][["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]].iloc[[-1]]
        prediction = self.model.predict(entry)[0]
        elapsed = time.time() - start_time
        print(f"get_action execution time: {elapsed:.6f} seconds")
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

class RFPredAllSignedStratMateoCheating(Strategy):
    """
    Strategy that uses the RF predictions to make trades.
    It uses the predictions of all assets to make trades.
    It is a signed strategy, meaning that it takes into account the sign of the prediction.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"predictors/mateo/rf_model_{window_size}ms.joblib")
        self.target_eth = 10.0
        self.btc_df = pd.read_parquet("data/features/DATA_0/XBT_EUR.parquet")
        self.prediction = self.model.predict(self.btc_df[["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]])
        self.prediction = pd.Series(self.prediction, index=self.btc_df.index)
        print(self.prediction.shape)

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        # Reorder features columns to match the model's expected input
        prediction = self.prediction.loc[data["XBT"].index[-1]]  # Get the prediction for the last timestamp
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
