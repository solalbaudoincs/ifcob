from backtesting.strategy import Strategy
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.portfolio import Portfolio
import joblib
import pandas as pd
import time
import heapq

class Mateo2StartStrategy(Strategy):
    """
    Random Forest Prediction-Based Strategy for ETH Trading (Cheating Version).

    This strategy precomputes all predictions for the XBT dataset and uses the correct prediction for each timestamp.
    It acts as follows:
      - If prediction == -1 and ETH is held, issues a sell order for 0.1 ETH.
      - If prediction == 0 and ETH holdings < target_eth, issues a buy order up to target_eth (max 0.2 ETH per step).
      - If prediction not in (-1, 0, 1), raises an error.
      - Otherwise, holds (returns empty action).

    Args:
        model_path (str): Path to the pre-trained model file.
    """

    def __init__(self, model_path="predictors/mateo/target-avg_10ms_of_mid_price_itincreases_after_200ms_with_threshold_5_depth-5_nest-100/model.joblib"):
        super().__init__()
        self.model = joblib.load(model_path)
        self.target_eth = 10.0
        self.btc_df = pd.read_parquet("data/features/DATA_1/XBT_EUR.parquet")
        self.prediction = self.model.predict(self.btc_df[[
            "slope-bid-5-levels",
            "slope-ask-5-levels",
            "avg-250ms-of-slope-ask-5-levels",
            "avg-250ms-of-slope-bid-5-levels",
            "avg-250ms-of-V-bid-5-levels",
            "avg-250ms-of-V-ask-5-levels",
            "avg-250ms-of-liquidity-ratio-5-levels",
        ]])
        self.prediction = pd.Series(self.prediction, index=self.btc_df.index)
        self.buy_orders = []
        heapq.heapify(self.buy_orders)
        #print(self.prediction.shape)
    
    def program_trade(self, eth_amount: float, timestamp : float):
        self.buy_orders.append((timestamp, eth_amount))

    def needed_trade_amount(self, current_timestamp):
        adujstment = 0.0
        for ts, amt in self.buy_orders:
            if ts < current_timestamp:
                adujstment += amt
                heapq.heappop(self.buy_orders)
            else:
                break
        return adujstment

                

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
        orders = {"ETH": 0.0}
        # Reorder features columns to match the model's expected input
        current_timestamp = data["XBT"].index[-1]
        prediction = self.prediction.loc[current_timestamp]  # Get the prediction for the last timestamp
        if prediction == 1:
            orders["ETH"] += 0.01
            self.program_trade(-0.01, current_timestamp+200)
        orders["ETH"] += self.needed_trade_amount(current_timestamp)
        return orders
        


