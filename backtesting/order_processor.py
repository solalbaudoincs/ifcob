from .types import OrderBookDataLoader, Coin, TimeStep, FeesGraph
from .portfolio import Portfolio, estimate_price, get_fee_for_trade
import pandas as pd
from typing import Dict, Optional


class OrderProcessor:
    """
    OrderProcessor handles the execution of buy and sell orders for a portfolio using market data and a fee structure.

    Methods:
        - process_order(coin, amount, action_type, execution_timestamp, portfolio): Processes a buy or sell order at a given timestamp.
        - _process_buy_order(...): Internal method to process a buy order.
        - _process_sell_order(...): Internal method to process a sell order.

    Example usage:
        processor = OrderProcessor(fees_graph, dataloader)
        trade = processor.process_order('ETH', 1.0, 'buy', 1234567890, portfolio)
    """

    def __init__(self, fees_graph: FeesGraph, dataloader: OrderBookDataLoader):
        self.fees_graph = fees_graph
        self.dataloader = dataloader
        self.timesteps = dataloader.get_time_step_values()

    def process_order(self, coin: Coin, amount: float, action_type: str,
                      execution_timestamp: TimeStep, portfolio: Portfolio) -> Optional[Dict]:
        """
        Process an individual order at execution_timestamp (timestamp + runtime + delta).
        Gets market data at the actual execution time, not decision time.
        Updates portfolio at execution time.

        Args:
            coin (Coin): The coin to buy or sell.
            amount (float): The amount of the coin to buy (for 'buy') or sell (for 'sell').
            action_type (str): Either 'buy' or 'sell'.
            execution_timestamp (TimeStep): The timestamp at which the order is executed.
            portfolio (Portfolio): The portfolio to update.

        Returns:
            Optional[Dict]: A dictionary with trade details if successful, or None if the trade could not be executed.
        """
        try:
            # Get market data at execution time
            last_timestep_before_execution = self.timesteps[coin].searchsorted(execution_timestamp, side='left')
            execution_market_data = self.dataloader.get_coin_at_timestep(
                coin, last_timestep_before_execution
            )

            if execution_market_data.empty:
                print(
                    f"No market data available for {coin} at execution time {execution_timestamp}")
                return None

            if action_type == 'buy':
                return self._process_buy_order(coin, amount, execution_market_data, execution_timestamp, portfolio)
            elif action_type == 'sell':
                return self._process_sell_order(coin, amount, execution_market_data, execution_timestamp, portfolio)
            else:
                raise ValueError(f"Unknown action type: {action_type}")

        except Exception as e:
            print(
                f"Order processing failed for {coin} at {execution_timestamp}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_market_data_at_timestamp(self, coin: Coin, timestamp: TimeStep) -> pd.DataFrame:
        """Get market data for a specific coin at the exact execution timestamp"""
        try:
            # Get a small range around the timestamp to find the closest data point
            start_range = 0  # 1ms before
            end_range = timestamp + 0.001    # 1ms after

            market_data = self.dataloader.get_books_from_range(
                start_range, end_range)

            if coin in market_data and not market_data[coin].empty:
                # Find the closest timestamp
                coin_data = market_data[coin]
                if 'timestamp' in coin_data.columns:
                    closest_idx = (
                        coin_data['timestamp'] - timestamp).abs().idxmin()
                    return coin_data.loc[[closest_idx]]
                else:
                    # If no timestamp column, return the last row
                    return coin_data.tail(1)
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Could not get market data for {coin} at {timestamp}: {e}")
            return pd.DataFrame()

    def _process_buy_order(self, coin: Coin, amount: float, coin_data: pd.DataFrame,
                           execution_timestamp: TimeStep, portfolio: Portfolio) -> Optional[Dict]:
        """
        Process a buy order with market impact and fees at execution time.

        Args:
            coin (Coin): The coin to buy.
            amount (float): The amount of the coin to buy.
            coin_data (pd.DataFrame): Market data for the coin at execution time.
            execution_timestamp (TimeStep): The timestamp at which the order is executed.
            portfolio (Portfolio): The portfolio to update.

        Returns:
            Optional[Dict]: Trade details if successful, None otherwise.
        """
        try:
            price = estimate_price(coin_data, 'ask')

            cost = amount * price

            # Check if portfolio can execute trade at execution time
            if portfolio.can_execute_trade('EURC', coin, cost, self.fees_graph):
                fee_rate = get_fee_for_trade('EURC', coin, self.fees_graph)

                # CRITICAL: Portfolio is updated HERE at execution time
                success = portfolio.execute_trade('EURC', coin, price, amount, self.fees_graph)

                if success:
                    return {
                        'execution_timestamp': execution_timestamp,
                        'coin': coin,
                        'action': 'buy',
                        'amount': amount,
                        'effective_price': price,
                        'cost': cost,
                        'fee': cost * fee_rate,
                        'fee_rate': fee_rate
                    }
            else:
                print(f"Buy order failed for {coin}: volume : {amount}, coin_data : {coin_data}, portfolio : {portfolio}, XBT_price : {price}, coin_price : {price}, total cost : {cost}€")
        except ValueError as e:
            print(f"Buy order failed for {coin}: {e}")

        return None

    def _process_sell_order(self, coin: Coin, amount: float, coin_data: pd.DataFrame,
                            execution_timestamp: TimeStep, portfolio: Portfolio) -> Optional[Dict]:
        """
        Process a sell order with market impact and fees at execution time.

        Args:
            coin (Coin): The coin to sell.
            amount (float): The amount of the coin to sell.
            coin_data (pd.DataFrame): Market data for the coin at execution time.
            execution_timestamp (TimeStep): The timestamp at which the order is executed.
            portfolio (Portfolio): The portfolio to update.

        Returns:
            Optional[Dict]: Trade details if successful, None otherwise.
        """
        try:
            
            price = estimate_price(coin_data, 'bid')

            # Check if portfolio has enough coins to sell at execution time
            if portfolio.can_execute_trade(coin, 'EURC', amount, self.fees_graph):
                fee_rate = get_fee_for_trade(coin, 'EURC', self.fees_graph)

                # CRITICAL: Portfolio is updated HERE at execution time
                success = portfolio.execute_trade(coin, 'EURC', price, amount, self.fees_graph, reverse=True)

                if success:
                    proceeds = amount * price
                    return {
                        'execution_timestamp': execution_timestamp,
                        'coin': coin,
                        'action': 'sell',
                        'amount': amount,
                        'effective_price': price,
                        'proceeds': proceeds,
                        'fee': proceeds * fee_rate,
                        'fee_rate': fee_rate
                    }
            
        except ValueError as e:
            print(f"Sell order failed for {coin}: {e}")

        return None