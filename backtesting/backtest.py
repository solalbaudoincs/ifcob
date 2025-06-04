from .types import OrderBookDataLoader, MarketData, Coin, TimeStep, Action, FeesGraph
from .strategy import Strategy
from .portolio import Portfolio, estimate_price, get_fee_for_trade, get_market_impact_estimate
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from tqdm.rich import tqdm
import time


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    portfolio_values: List[float]
    timestamps: List[float]
    trades: List[Dict]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    transaction_costs: float
    final_portfolio_value: float


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float
    fees_graph: FeesGraph
    symbols: List[Coin]
    window_size: int = 10  # Number of last rows to include in windowed market data

    # Validation splits
    calibration_end_time: Optional[TimeStep] = None  # End of in-sample data
    # Start of out-of-sample data
    validation_start_time: Optional[TimeStep] = None

    # Runtime and latency management
    # Max time allowed for get_action (seconds)
    max_action_runtime: float = 0.1
    skip_on_timeout: bool = True         # Skip timestep if action takes too long
    network_latency_delta: float = 0.05  # Network latency in seconds (delta)


class OrderProcessor:
    """Separate order processing component as shown in architecture"""

    def __init__(self, fees_graph: FeesGraph, dataloader: OrderBookDataLoader):
        self.fees_graph = fees_graph
        self.dataloader = dataloader

    def process_order(self, coin: Coin, amount: float, action_type: str,
                      execution_timestamp: TimeStep, portfolio: Portfolio) -> Optional[Dict]:
        """
        Process individual order at execution_timestamp (timestamp + runtime + delta)
        Gets market data at the actual execution time, not decision time
        Updates portfolio at execution time
        """
        try:
            # Get market data at execution time
            execution_market_data = self._get_market_data_at_timestamp(
                coin, execution_timestamp)

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
            return None

    def _get_market_data_at_timestamp(self, coin: Coin, timestamp: TimeStep) -> pd.DataFrame:
        """Get market data for a specific coin at the exact execution timestamp"""
        try:
            # Get a small range around the timestamp to find the closest data point
            start_range = timestamp - 0.001  # 1ms before
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
        """Process buy order with market impact and fees at execution time"""
        try:
            effective_price = get_market_impact_estimate(
                coin_data, amount, 'buy')
            if effective_price == 0:
                effective_price = estimate_price(coin_data)

            cost = amount * effective_price

            # Check if portfolio can execute trade at execution time
            if portfolio.can_execute_trade('EURC', coin, cost, self.fees_graph):
                fee_rate = get_fee_for_trade('EURC', coin, self.fees_graph)

                # CRITICAL: Portfolio is updated HERE at execution time
                success = portfolio.execute_trade(
                    'EURC', coin, cost, self.fees_graph)

                if success:
                    return {
                        'execution_timestamp': execution_timestamp,
                        'coin': coin,
                        'action': 'buy',
                        'amount': amount,
                        'effective_price': effective_price,
                        'cost': cost,
                        'fee': cost * fee_rate,
                        'fee_rate': fee_rate
                    }
        except ValueError as e:
            print(f"Buy order failed for {coin}: {e}")

        return None

    def _process_sell_order(self, coin: Coin, amount: float, coin_data: pd.DataFrame,
                            execution_timestamp: TimeStep, portfolio: Portfolio) -> Optional[Dict]:
        """Process sell order with market impact and fees at execution time"""
        try:
            effective_price = get_market_impact_estimate(
                coin_data, amount, 'sell')
            if effective_price == 0:
                effective_price = estimate_price(coin_data)

            # Check if portfolio has enough coins to sell at execution time
            if portfolio.can_execute_trade(coin, 'EURC', amount, self.fees_graph):
                fee_rate = get_fee_for_trade(coin, 'EURC', self.fees_graph)

                # CRITICAL: Portfolio is updated HERE at execution time
                success = portfolio.execute_trade(
                    coin, 'EURC', amount, self.fees_graph)

                if success:
                    proceeds = amount * effective_price
                    return {
                        'execution_timestamp': execution_timestamp,
                        'coin': coin,
                        'action': 'sell',
                        'amount': amount,
                        'effective_price': effective_price,
                        'proceeds': proceeds,
                        'fee': proceeds * fee_rate,
                        'fee_rate': fee_rate
                    }
        except ValueError as e:
            print(f"Sell order failed for {coin}: {e}")

        return None


class Backtester:

    def __init__(self, dataloader: OrderBookDataLoader, config: BacktestConfig) -> None:
        self.dataloader = dataloader
        self.config = config
        self.order_processor = OrderProcessor(config.fees_graph, dataloader)
        self.runtime_stats = {}  # Track runtime statistics per strategy
        self.pending_orders = []  # Queue for orders awaiting execution

    def backtest(self, strategies: List[Strategy]) -> Dict[str, Tuple[BacktestResult, BacktestResult]]:
        """
        Tests multiple strategies in parallel with realistic execution delays.
        Portfolios are updated at execution time (timestamp + runtime + delta), not decision time.
        """
        if not self.config.calibration_end_time or not self.config.validation_start_time:
            raise ValueError(
                "Calibration and validation times must be set for validation split")

        # Initialize tracking for each strategy
        strategy_states = {}
        for i, strategy in enumerate(strategies):
            strategy_name = getattr(
                strategy, '__class__.__name__', f'Strategy_{i}')
            strategy_states[strategy_name] = {
                'strategy': strategy,
                'calibration_portfolio': Portfolio(self.config.symbols, self.config.initial_capital),
                'validation_portfolio': Portfolio(self.config.symbols, self.config.initial_capital),
                'calibration_trades': [],
                'validation_trades': [],
                'calibration_history': [],
                'validation_history': []
            }
            # Initialize runtime tracking
            self.runtime_stats[strategy_name] = {
                'total_runtime': 0.0,
                'action_count': 0,
                'timeout_count': 0,
                'avg_runtime': 0.0
            }

        unique_timesteps = len(set().union(
            *self.dataloader.get_time_step_values().values()))

        # Iterate through each timestep in chronological order
        for current_timestep, coin_indices in tqdm(self.dataloader.chronological_iterator(),
                                                   total=unique_timesteps,
                                                   unit_scale="timesteps",
                                                   desc="Backtesting"):
            try:
                # FIRST: Process pending orders that should execute at or before current_timestep
                # This ensures portfolios are updated with completed trades before making new decisions
                self._process_pending_orders(current_timestep, strategy_states)

                # Create windowed market data for current timestep
                windowed_market_data = self._create_windowed_market_data(
                    current_timestep, coin_indices, self.config.window_size)

                # Skip if no data available
                if not windowed_market_data or all(data.empty for data in windowed_market_data.values()):
                    continue

                # Test all strategies in parallel at this timestep
                for strategy_name, state in strategy_states.items():
                    strategy = state['strategy']

                    # Process calibration phase
                    if current_timestep <= self.config.calibration_end_time:
                        portfolio = state['calibration_portfolio']
                        history = state['calibration_history']

                        # Get strategy actions with runtime measurement
                        actions, action_runtime = self._get_timed_action(
                            strategy, windowed_market_data, portfolio, self.config.fees_graph, strategy_name
                        )

                        # Skip execution if action took too long
                        if self.config.skip_on_timeout and action_runtime > self.config.max_action_runtime:
                            print(
                                f"Warning: {strategy_name} action timeout at {current_timestep} ({action_runtime:.4f}s)")
                            continue

                        # Schedule orders for execution at timestamp + runtime + delta
                        # Portfolio will be updated when orders execute, NOT NOW
                        execution_timestamp = current_timestep + \
                            action_runtime + self.config.network_latency_delta
                        self._schedule_orders(actions, execution_timestamp, current_timestep, action_runtime,
                                              strategy_name, 'calibration')

                        # Record portfolio value at decision time (before new trades execute)
                        portfolio_value = portfolio.get_value(
                            windowed_market_data)
                        history.append((current_timestep, portfolio_value))

                    # Process validation phase
                    if current_timestep >= self.config.validation_start_time:
                        portfolio = state['validation_portfolio']
                        history = state['validation_history']

                        # Get strategy actions with runtime measurement
                        actions, action_runtime = self._get_timed_action(
                            strategy, windowed_market_data, portfolio, self.config.fees_graph, strategy_name
                        )

                        # Skip execution if action took too long
                        if self.config.skip_on_timeout and action_runtime > self.config.max_action_runtime:
                            print(
                                f"Warning: {strategy_name} action timeout at {current_timestep} ({action_runtime:.4f}s)")
                            continue

                        # Schedule orders for execution at timestamp + runtime + delta
                        execution_timestamp = current_timestep + \
                            action_runtime + self.config.network_latency_delta
                        self._schedule_orders(actions, execution_timestamp, current_timestep, action_runtime,
                                              strategy_name, 'validation')

                        # Record portfolio value at decision time (before new trades execute)
                        portfolio_value = portfolio.get_value(
                            windowed_market_data)
                        history.append((current_timestep, portfolio_value))

            except Exception as e:
                print(f"Error at timestep {current_timestep}: {e}")

        # Process any remaining pending orders
        self._process_remaining_orders(strategy_states)

        # Print runtime statistics
        self._print_runtime_stats()

        # Build results for each strategy
        results = self._build_results(strategy_states)
        return results

    def _schedule_orders(self, actions: Action, execution_timestamp: TimeStep,
                         decision_timestamp: TimeStep, action_runtime: float,
                         strategy_name: str, phase: str):
        """Schedule orders for later execution"""
        for coin, amount_to_trade in actions.items():
            if abs(amount_to_trade) < 1e-8:  # Skip negligible trades
                continue

            order = {
                'execution_timestamp': execution_timestamp,
                'decision_timestamp': decision_timestamp,
                'action_runtime': action_runtime,
                'strategy_name': strategy_name,
                'phase': phase,
                'coin': coin,
                'amount': amount_to_trade
            }
            self.pending_orders.append(order)

    def _process_pending_orders(self, current_timestep: TimeStep, strategy_states: dict):
        """Process all orders scheduled for execution at or before current_timestep"""
        orders_to_remove = []

        for i, order in enumerate(self.pending_orders):
            if order['execution_timestamp'] <= current_timestep:
                # Execute the order - this will update the portfolio
                executed_trade = self._execute_scheduled_order(
                    order, strategy_states)
                if executed_trade:
                    # Add to appropriate trades list
                    strategy_name = order['strategy_name']
                    phase = order['phase']
                    if phase == 'calibration':
                        strategy_states[strategy_name]['calibration_trades'].append(
                            executed_trade)
                    else:
                        strategy_states[strategy_name]['validation_trades'].append(
                            executed_trade)

                orders_to_remove.append(i)

        # Remove processed orders
        for i in reversed(orders_to_remove):
            self.pending_orders.pop(i)

    def _execute_scheduled_order(self, order: dict, strategy_states: dict) -> Optional[Dict]:
        """Execute a single scheduled order using OrderProcessor"""
        strategy_name = order['strategy_name']
        phase = order['phase']

        # Get the appropriate portfolio
        if phase == 'calibration':
            portfolio = strategy_states[strategy_name]['calibration_portfolio']
        else:
            portfolio = strategy_states[strategy_name]['validation_portfolio']

        coin = order['coin']
        amount = order['amount']
        execution_timestamp = order['execution_timestamp']

        # Determine action type and execute using OrderProcessor
        # The OrderProcessor will update the portfolio at execution time
        if amount > 0:  # Buy
            trade = self.order_processor.process_order(
                coin, amount, 'buy', execution_timestamp, portfolio
            )
        else:  # Sell
            trade = self.order_processor.process_order(
                coin, abs(amount), 'sell', execution_timestamp, portfolio
            )

        if trade:
            # Add metadata about the delay
            trade['decision_timestamp'] = order['decision_timestamp']
            trade['action_runtime'] = order['action_runtime']
            trade['network_latency'] = self.config.network_latency_delta
            trade['total_delay'] = execution_timestamp - \
                order['decision_timestamp']

        return trade

    def _process_remaining_orders(self, strategy_states: dict):
        """Process any orders that haven't been executed yet at the end of backtest"""
        remaining_orders = len(self.pending_orders)
        if remaining_orders > 0:
            print(f"Processing {remaining_orders} remaining orders...")

            for order in self.pending_orders:
                executed_trade = self._execute_scheduled_order(
                    order, strategy_states)
                if executed_trade:
                    strategy_name = order['strategy_name']
                    phase = order['phase']
                    if phase == 'calibration':
                        strategy_states[strategy_name]['calibration_trades'].append(
                            executed_trade)
                    else:
                        strategy_states[strategy_name]['validation_trades'].append(
                            executed_trade)

        self.pending_orders.clear()

    def _get_timed_action(self, strategy: Strategy, market_data: MarketData,
                          portfolio: Portfolio, fees_graph: FeesGraph,
                          strategy_name: str) -> Tuple[Action, float]:
        """Execute strategy.get_action() and measure its runtime"""
        start_time = time.perf_counter()

        try:
            actions = strategy.get_action(market_data, portfolio, fees_graph)
        except Exception as e:
            print(f"Error in {strategy_name}.get_action(): {e}")
            actions = {}

        end_time = time.perf_counter()
        runtime = end_time - start_time

        # Update runtime statistics
        stats = self.runtime_stats[strategy_name]
        stats['total_runtime'] += runtime
        stats['action_count'] += 1
        stats['avg_runtime'] = stats['total_runtime'] / stats['action_count']

        if runtime > self.config.max_action_runtime:
            stats['timeout_count'] += 1

        return actions, runtime

    def _print_runtime_stats(self):
        """Print runtime statistics for all strategies"""
        print("\n=== Runtime Statistics ===")
        for strategy_name, stats in self.runtime_stats.items():
            print(f"\n{strategy_name}:")
            print(f"  Total runtime: {stats['total_runtime']:.4f}s")
            print(f"  Action calls: {stats['action_count']}")
            print(f"  Average runtime: {stats['avg_runtime']:.4f}s")
            print(f"  Timeouts: {stats['timeout_count']}")
            if stats['action_count'] > 0:
                timeout_rate = (stats['timeout_count'] /
                                stats['action_count']) * 100
                print(f"  Timeout rate: {timeout_rate:.1f}%")
        print(
            f"\nNetwork latency delta: {self.config.network_latency_delta:.4f}s")
        print(f"Pending orders at end: {len(self.pending_orders)}")

    def _build_results(self, strategy_states: dict) -> Dict[str, Tuple[BacktestResult, BacktestResult]]:
        """Build BacktestResult objects for each strategy"""
        results = {}
        for strategy_name, state in strategy_states.items():
            # Calculate calibration metrics
            calibration_metrics = self._calculate_metrics_from_data(
                state['calibration_history'], state['calibration_trades'])

            calibration_result = BacktestResult(
                portfolio_values=[v for _, v in state['calibration_history']],
                timestamps=[t for t, _ in state['calibration_history']],
                trades=state['calibration_trades'].copy(),
                total_return=calibration_metrics['total_return'],
                sharpe_ratio=calibration_metrics['sharpe_ratio'],
                max_drawdown=calibration_metrics['max_drawdown'],
                win_rate=calibration_metrics['win_rate'],
                transaction_costs=sum(trade.get('fee', 0)
                                      for trade in state['calibration_trades']),
                final_portfolio_value=state['calibration_history'][-1][1] if state['calibration_history'] else self.config.initial_capital
            )

            # Calculate validation metrics
            validation_metrics = self._calculate_metrics_from_data(
                state['validation_history'], state['validation_trades'])

            validation_result = BacktestResult(
                portfolio_values=[v for _, v in state['validation_history']],
                timestamps=[t for t, _ in state['validation_history']],
                trades=state['validation_trades'].copy(),
                total_return=validation_metrics['total_return'],
                sharpe_ratio=validation_metrics['sharpe_ratio'],
                max_drawdown=validation_metrics['max_drawdown'],
                win_rate=validation_metrics['win_rate'],
                transaction_costs=sum(trade.get('fee', 0)
                                      for trade in state['validation_trades']),
                final_portfolio_value=state['validation_history'][-1][1] if state['validation_history'] else self.config.initial_capital
            )

            results[strategy_name] = (calibration_result, validation_result)

        return results

    def _calculate_metrics_from_data(self, portfolio_history: List[Tuple[float, float]],
                                     trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from portfolio history and trades"""
        if len(portfolio_history) < 2:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

        values = [v for _, v in portfolio_history]
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] - values[0]) / values[0]
        sharpe_ratio = np.mean(returns) / \
            np.std(returns) if np.std(returns) > 0 else 0.0

        # Max drawdown calculation
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

        # Win rate calculation
        winning_trades = sum(1 for trade in trades
                             if trade['action'] == 'sell' and trade.get('proceeds', 0) > 0)
        total_sells = sum(1 for trade in trades if trade['action'] == 'sell')
        win_rate = winning_trades / total_sells if total_sells > 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }

    def _create_windowed_market_data(self, current_timestep: TimeStep, coin_indices: dict, window_size: int) -> MarketData:
        """
        Create windowed market data containing the k last rows for each coin up to current_timestep.
        Uses coin indices from chronological_iterator for efficient processing.
        """
        windowed_data = {}

        for coin in self.config.symbols:
            try:
                if coin not in coin_indices:
                    # If coin has no more data, return empty DataFrame
                    windowed_data[coin] = pd.DataFrame()
                    continue

                current_idx = coin_indices[coin]

                # Calculate window start index (ensure it's not negative)
                start_idx = max(0, current_idx - window_size + 1)
                end_idx = current_idx + 1  # Include current index

                # Get the DataFrame slice directly using indices from dataloader
                df = self.dataloader.dfs[coin]
                windowed_data[coin] = df.iloc[start_idx:end_idx]

            except Exception as e:
                print(
                    f"Warning: Could not get windowed data for {coin} at timestep {current_timestep}: {e}")
                windowed_data[coin] = pd.DataFrame()

        return windowed_data
