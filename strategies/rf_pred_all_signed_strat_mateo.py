from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import joblib
import pandas as pd
import time

class RFPredAllSignedStratMateo(Strategy):
    """
    Random Forest Prediction-Based Strategy for ETH Trading.

    This strategy uses a pre-trained Random Forest model to generate trading signals for ETH based on XBT market features.
    The model predicts a signal (-1: sell, 0: hold/rebalance, 1: buy), and the strategy acts accordingly:
      - If prediction == -1 and ETH is held, issues a sell order for 0.1 ETH.
      - If prediction == 0, rebalances ETH holdings to a target amount (default: 100.0 ETH).
      - If prediction == 1 and ETH holdings < 200, issues a buy order for 0.1 ETH.

    Args:
        window_size (int): The window size (in ms) for the model features.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"predictors/mateo/rf_model_{window_size}ms.joblib")
        self.target_eth = 100.0


    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate a trading action for ETH based on the latest XBT features and the RF model prediction.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell.
                - {"ETH": -0.1} for sell
                - {"ETH": 0.1} for buy
                - {"ETH": self.target_eth - current_portfolio.get_position("ETH")} for rebalance/hold
        """
        start_time = time.time()
        # Reorder features columns to match the model's expected input
        entry = data["XBT"][["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]].iloc[[-1]]
        prediction = self.model.predict(entry)[0]
        elapsed = time.time() - start_time
        print(f"get_action execution time: {elapsed:.6f} seconds")
        if prediction == -1 and current_portfolio.get_position("ETH") > 0:
            # sell signal
            return {"ETH" : -0.1}
        elif prediction == 0:
            # Hold signal
            return {"ETH": self.target_eth - current_portfolio.get_position("ETH")}
        elif prediction == 1 and current_portfolio.get_position("ETH") < 200:
            # buy signal
            return {"ETH": 0.1}
        else:
            raise ValueError(f"Unexpected prediction value: {prediction}. Expected -1, 0, or 1.")

class RFPredAllSignedStratMateoCheating(Strategy):
    """
    Random Forest Prediction-Based Strategy for ETH Trading (Cheating Version).

    This strategy precomputes all predictions for the XBT dataset and uses the correct prediction for each timestamp.
    It acts as follows:
      - If prediction == -1 and ETH is held, issues a sell order for 0.1 ETH.
      - If prediction == 0 and ETH holdings < target_eth, issues a buy order up to target_eth (max 0.2 ETH per step).
      - If prediction not in (-1, 0, 1), raises an error.
      - Otherwise, holds (returns empty action).

    Args:
        window_size (int): The window size (in ms) for the model features.
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.model = joblib.load(f"predictors/mateo/rf_model_{window_size}ms.joblib")
        self.target_eth = 10.0
        self.btc_df = pd.read_parquet("data/features/DATA_1/XBT_EUR.parquet")
        self.prediction = self.model.predict(self.btc_df[["bid-ask-imbalance-5-levels", "spread", "inst-return", "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"]])
        self.prediction = pd.Series(self.prediction, index=self.btc_df.index)
        print(self.prediction.shape)

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Generate a trading action for ETH using precomputed predictions for the current timestamp.

        Args:
            data (MarketData): Dictionary of DataFrames for each symbol, containing recent market features.
            current_portfolio (Portfolio): The current portfolio state.
            fees_graph (FeesGraph): The transaction fee structure.

        Returns:
            Action: Dictionary of {symbol: amount} to trade. Positive for buy, negative for sell, or empty dict for hold.
        """
        # Reorder features columns to match the model's expected input
        prediction = self.prediction.loc[data["XBT"].index[-1]]  # Get the prediction for the last timestamp
        if prediction == -1 and current_portfolio.get_position("ETH") > 0:
            # sell signal
            return {"ETH" : -0.1}
        elif prediction == 0 and current_portfolio.get_position("ETH") < self.target_eth:
            # Hold signal
            return {"ETH": min(0.2, self.target_eth - current_portfolio.get_position("ETH"))}
        #elif prediction == 1 and current_portfolio.get_position("ETH") < 200:
            # buy signal
        #    return {"ETH": 0.1}
        elif prediction not in (-1, 0, 1):
            raise ValueError(f"Unexpected prediction value: {prediction}. Expected -1, 0, or 1.")
        return {}

class GenericSignalBasedStrategy(Strategy):
    """
    Stratégie de trading générique fondée sur des signaux -1, 0, 1 d’un modèle IA.

    - 1 → acheter buy_qty d'ETH
    - 0 → ne rien faire
    - -1 → vendre sell_qty d'ETH (si position suffisante)

    Paramètres :
    - model_path (str): chemin vers le modèle joblib
    - buy_qty (float): quantité à acheter si signal == 1
    - sell_qty (float): quantité à vendre si signal == -1
    - features (list): liste des colonnes à utiliser
    """

    def __init__(self, model_path: str, buy_qty: float = 0.1, sell_qty: float = 0.1,
                 features: list = None):
        super().__init__()
        self.model = joblib.load(model_path)
        self.buy_qty = buy_qty
        self.sell_qty = sell_qty
        self.features = features or [
            "bid-ask-imbalance-5-levels",
            "spread",
            "inst-return",
            "V-bid-5-levels",
            "V-ask-5-levels",
            "slope-bid-5-levels",
            "slope-ask-5-levels"
        ]

    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        xbt_data = data["XBT"]
        latest_features = xbt_data[self.features].iloc[[-1]]  # Shape (1, n_features)

        signal = self.model.predict(latest_features)[0]

        eth_position = current_portfolio.get_position("ETH")

        if signal == 1:
            return {"ETH": self.buy_qty}
        elif signal == -1 and eth_position >= self.sell_qty:
            return {"ETH": -self.sell_qty}
        else:
            return {}
