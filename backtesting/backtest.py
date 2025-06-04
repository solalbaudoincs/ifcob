from .types import OrderBookDataLoader, MarketData, Coin, TimeStep, Action, FeesGraph
from .strategy import Strategy
from .portfolio import Portfolio, estimate_price, get_fee_for_trade
from .order_processor import OrderProcessor
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from tqdm.rich import tqdm
import time
import json
import os
from datetime import datetime

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
    network_latency_delta: float = 0.001  # Network latency in seconds (delta)


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
                'validation_history': [],
                'recent_actions': []  # Track last 10 actions for status updates
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

        iteration_count = 0  # Track iterations for periodic updates

        # Iterate through each timestep in chronological order
        for current_timestep, coin_indices in tqdm(self.dataloader.chronological_iterator(),
                                                   total=unique_timesteps,
                                                   unit_scale="timesteps",
                                                   desc="Backtesting"):
            try:
                iteration_count += 1
                
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

                        # Track recent actions for status updates
                        if actions:
                            action_summary = {
                                'timestamp': current_timestep,
                                'phase': 'calibration',
                                'actions': dict(actions),
                                'portfolio_value': portfolio.get_value(windowed_market_data)
                            }
                            state['recent_actions'].append(action_summary)
                            # Keep only last 10 actions
                            if len(state['recent_actions']) > 10:
                                state['recent_actions'].pop(0)

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

                        # Track recent actions for status updates
                        if actions:
                            action_summary = {
                                'timestamp': current_timestep,
                                'phase': 'validation',
                                'actions': dict(actions),
                                'portfolio_value': portfolio.get_value(windowed_market_data)
                            }
                            state['recent_actions'].append(action_summary)
                            # Keep only last 10 actions
                            if len(state['recent_actions']) > 10:
                                state['recent_actions'].pop(0)

                        # Skip execution if action took too long
                        if self.config.skip_on_timeout and action_runtime > self.config.max_action_runtime:
                            print(
                                f"Warning: {strategy_name} action timeout at {current_timestep} ({action_runtime:.4f}s)")
                            continue

                        # Schedule orders for execution at timestamp + runtime + delta
                        execution_timestamp = current_timestep + action_runtime + self.config.network_latency_delta
                        self._schedule_orders(actions, execution_timestamp, current_timestep, action_runtime,
                                              strategy_name, 'validation')

                        # Record portfolio value at decision time (before new trades execute)
                        portfolio_value = portfolio.get_value(windowed_market_data)
                        history.append((current_timestep, portfolio_value))

                # Print status update every 10,000 iterations
                if iteration_count % 10000 == 0:
                    self._print_status_update(iteration_count, current_timestep, strategy_states, windowed_market_data)

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
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

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
                print(f"Could not get windowed data for {coin} at timestep {current_timestep}: {e}")
                windowed_data[coin] = pd.DataFrame()

        return windowed_data

    def save_results_to_files(self, results: Dict[str, Tuple[BacktestResult, BacktestResult]], 
                            output_dir: str = "backtest_results", 
                            timestamp_suffix: bool = True,
                            save_json: bool = True,
                            save_csv: bool = True) -> Dict[str, str]:
        """
        Save backtest results to files in JSON and/or CSV format.
        
        Args:
            results: Dictionary of strategy results from backtest()
            output_dir: Directory to save files (will be created if doesn't exist)
            timestamp_suffix: Whether to add timestamp to filenames
            save_json: Whether to save results in JSON format
            save_csv: Whether to save results in CSV format
            
        Returns:
            Dictionary mapping file types to saved file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp suffix if requested
        if timestamp_suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{timestamp}"
        else:
            suffix = ""
        
        saved_files = {}
        
        if save_json:
            json_path = self._save_results_as_json(results, output_dir, suffix)
            saved_files['json'] = json_path
            
        if save_csv:
            csv_paths = self._save_results_as_csv(results, output_dir, suffix)
            saved_files.update(csv_paths)
            
        return saved_files
    
    def _save_results_as_json(self, results: Dict[str, Tuple[BacktestResult, BacktestResult]], 
                             output_dir: str, suffix: str) -> str:
        """Save results as JSON file with complete backtest information."""
        json_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "backtest_config": {
                    "initial_capital": self.config.initial_capital,
                    "symbols": self.config.symbols,
                    "window_size": self.config.window_size,
                    "calibration_end_time": self.config.calibration_end_time,
                    "validation_start_time": self.config.validation_start_time,
                    "max_action_runtime": self.config.max_action_runtime,
                    "network_latency_delta": self.config.network_latency_delta
                },
                "runtime_stats": self.runtime_stats
            },
            "strategies": {}
        }
        
        # Convert results to JSON-serializable format
        for strategy_name, (cal_result, val_result) in results.items():
            json_data["strategies"][strategy_name] = {
                "calibration": self._backtest_result_to_dict(cal_result),
                "validation": self._backtest_result_to_dict(val_result)
            }
        
        # Save JSON file
        json_filepath = os.path.join(output_dir, f"backtest_results{suffix}.json")
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"✅ Saved complete backtest results to: {json_filepath}")
        return json_filepath
    
    def _save_results_as_csv(self, results: Dict[str, Tuple[BacktestResult, BacktestResult]], 
                            output_dir: str, suffix: str) -> Dict[str, str]:
        """Save results as CSV files (summary metrics, trades, and portfolio evolution)."""
        saved_files = {}
        
        # 1. Summary metrics CSV
        summary_data = []
        for strategy_name, (cal_result, val_result) in results.items():
            summary_data.append({
                'strategy': strategy_name,
                'phase': 'calibration',
                'total_return': cal_result.total_return,
                'sharpe_ratio': cal_result.sharpe_ratio,
                'max_drawdown': cal_result.max_drawdown,
                'win_rate': cal_result.win_rate,
                'transaction_costs': cal_result.transaction_costs,
                'final_portfolio_value': cal_result.final_portfolio_value,
                'total_trades': len(cal_result.trades)
            })
            summary_data.append({
                'strategy': strategy_name,
                'phase': 'validation',
                'total_return': val_result.total_return,
                'sharpe_ratio': val_result.sharpe_ratio,
                'max_drawdown': val_result.max_drawdown,
                'win_rate': val_result.win_rate,
                'transaction_costs': val_result.transaction_costs,
                'final_portfolio_value': val_result.final_portfolio_value,
                'total_trades': len(val_result.trades)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"backtest_summary{suffix}.csv")
        summary_df.to_csv(summary_path, index=False)
        saved_files['summary_csv'] = summary_path
        print(f"✅ Saved summary metrics to: {summary_path}")
        
        # 2. Portfolio evolution CSV
        portfolio_data = []
        for strategy_name, (cal_result, val_result) in results.items():
            # Calibration phase
            for timestamp, value in zip(cal_result.timestamps, cal_result.portfolio_values):
                portfolio_data.append({
                    'strategy': strategy_name,
                    'phase': 'calibration',
                    'timestamp': timestamp,
                    'portfolio_value': value
                })
            # Validation phase
            for timestamp, value in zip(val_result.timestamps, val_result.portfolio_values):
                portfolio_data.append({
                    'strategy': strategy_name,
                    'phase': 'validation',
                    'timestamp': timestamp,
                    'portfolio_value': value
                })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_path = os.path.join(output_dir, f"portfolio_evolution{suffix}.csv")
        portfolio_df.to_csv(portfolio_path, index=False)
        saved_files['portfolio_csv'] = portfolio_path
        print(f"✅ Saved portfolio evolution to: {portfolio_path}")
        
        # 3. Trades CSV
        trades_data = []
        for strategy_name, (cal_result, val_result) in results.items():
            # Calibration trades
            for trade in cal_result.trades:
                trade_record = trade.copy()
                trade_record['strategy'] = strategy_name
                trade_record['phase'] = 'calibration'
                trades_data.append(trade_record)
            # Validation trades
            for trade in val_result.trades:
                trade_record = trade.copy()
                trade_record['strategy'] = strategy_name
                trade_record['phase'] = 'validation'
                trades_data.append(trade_record)
        
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
            trades_path = os.path.join(output_dir, f"trades_details{suffix}.csv")
            trades_df.to_csv(trades_path, index=False)
            saved_files['trades_csv'] = trades_path
            print(f"✅ Saved trade details to: {trades_path}")
        
        return saved_files
    
    def _backtest_result_to_dict(self, result: BacktestResult) -> dict:
        """Convert BacktestResult to dictionary for JSON serialization."""
        result_dict = asdict(result)
        
        # Handle numpy types that aren't JSON serializable
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result_dict[key] = value.item()
        
        return result_dict

    def _print_status_update(self, iteration: int, current_timestep: TimeStep, 
                           strategy_states: dict, windowed_market_data: MarketData):
        """Print detailed status update every 10,000 iterations"""
        print(f"\n{'='*80}")
        print(f"STATUS UPDATE - Iteration {iteration:,} (Timestep: {current_timestep})")
        print(f"{'='*80}")
        
        for strategy_name, state in strategy_states.items():
            print(f"\n📊 {strategy_name}:")
            
            # Determine current phase and portfolio
            if current_timestep <= self.config.calibration_end_time:
                current_phase = "CALIBRATION"
                current_portfolio = state['calibration_portfolio']
                current_history = state['calibration_history']
                current_trades = state['calibration_trades']
            else:
                current_phase = "VALIDATION"
                current_portfolio = state['validation_portfolio']
                current_history = state['validation_history']
                current_trades = state['validation_trades']
            
            print(f"   Phase: {current_phase}")
            
            # Portfolio composition and value
            current_value = current_portfolio.get_value(windowed_market_data)
            initial_value = self.config.initial_capital
            pnl = current_value - initial_value
            pnl_percent = (pnl / initial_value) * 100
            
            print(f"   💰 Portfolio Value: ${current_value:,.2f} (PnL: ${pnl:+,.2f} / {pnl_percent:+.2f}%)")
            
            # Calculate cumulative metrics
            if len(current_history) > 1:
                values = [v for _, v in current_history]
                
                # Cumulative return
                cumulative_return = (values[-1] - values[0]) / values[0] * 100
                
                # Cumulative Sharpe ratio
                returns = np.diff(values) / np.array(values[:-1])
                if len(returns) > 1 and np.std(returns) > 0:
                    cumulative_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
                else:
                    cumulative_sharpe = 0.0
                
                # Maximum drawdown so far
                peak = values[0]
                max_drawdown = 0.0
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                print(f"   📈 Performance Metrics:")
                print(f"      📊 Cumulative Return: {cumulative_return:+.2f}%")
                print(f"      ⚡ Cumulative Sharpe: {cumulative_sharpe:.4f}")
                print(f"      📉 Max Drawdown: {max_drawdown:.2%}")
                print(f"      📏 Data Points: {len(current_history)}")
            else:
                print(f"   📈 Performance Metrics: Insufficient data (need >1 data points)")
            
            # Portfolio composition
            print(f"   🏦 Holdings:")
            
            # Show EURC (base currency) first
            eurc_amount = current_portfolio.positions.get('EURC', 0)
            eurc_percent = (eurc_amount / current_value * 100) if current_value > 0 else 0
            print(f"      💵 EURC: ${eurc_amount:,.2f} ({eurc_percent:.1f}%)")
            
            # Show other coin holdings
            for coin in self.config.symbols:
                amount = current_portfolio.positions.get(coin, 0)
                if amount >= 0:
                    # Try to get current price for valuation
                    try:
                        if coin in windowed_market_data:
                            price = estimate_price(windowed_market_data[coin], "bid")
                            value = amount * price
                            value_percent = (value / current_value * 100) if current_value > 0 else 0
                            print(f"      🪙 {coin}: {amount:.6f} (≈${value:,.2f} @ ${price:.4f}, {value_percent:.1f}%)")
                        else:
                            print(f"      🪙 {coin}: {amount:.6f} (couldn't acess coin in windowed_market_data)")
                    except Exception as e:
                        print(f"      🪙 {coin}: {amount:.6f} (price unavailable)")
            
            # Trading activity
            total_trades = len(current_trades)
            if total_trades > 0:
                recent_trades = current_trades[-5:]  # Last 5 trades
                total_fees = sum(trade.get('fee', 0) for trade in current_trades)
                
                # Calculate win rate for current phase
                winning_trades = sum(1 for trade in current_trades 
                                   if trade.get('action') == 'sell' and trade.get('proceeds', 0) > trade.get('cost', 0))
                sell_trades = sum(1 for trade in current_trades if trade.get('action') == 'sell')
                win_rate = (winning_trades / sell_trades * 100) if sell_trades > 0 else 0
                
                print(f"   💼 Trading Activity:")
                print(f"      📊 Total: {total_trades} trades, ${total_fees:.2f} fees")
                print(f"      🎯 Win Rate: {win_rate:.1f}% ({winning_trades}/{sell_trades} profitable sells)")
                
                if recent_trades:
                    print(f"      🔄 Recent Trades:")
                    for trade in recent_trades:
                        action = trade.get('action', 'unknown')
                        coin = trade.get('coin', 'unknown')
                        amount = trade.get('amount', 0)
                        price = trade.get('price', 0)
                        fee = trade.get('fee', 0)
                        timestamp = trade.get('timestamp', 'unknown')
                        print(f"        {action.upper()} {amount:.6f} {coin} @ ${price:.4f} (fee: ${fee:.4f}) [{timestamp}]")
            else:
                print(f"   💼 Trading Activity: No trades executed yet")
            
            # Recent actions (decisions)
            recent_actions = state['recent_actions']
            if recent_actions:
                print(f"   🎯 Last {len(recent_actions)} Strategy Actions:")
                for action_info in recent_actions[-5:]:  # Show last 5 actions
                    timestamp = action_info['timestamp']
                    phase = action_info['phase']
                    actions = action_info['actions']
                    portfolio_val = action_info['portfolio_value']
                    
                    if actions:
                        action_str = ", ".join([f"{coin}: {amount:+.6f}" for coin, amount in actions.items() if abs(amount) > 1e-8])
                        if action_str:
                            print(f"      [{timestamp}] {phase}: {action_str} (portfolio: ${portfolio_val:,.2f})")
                        else:
                            print(f"      [{timestamp}] {phase}: No significant actions (portfolio: ${portfolio_val:,.2f})")
                    else:
                        print(f"      [{timestamp}] {phase}: No actions (portfolio: ${portfolio_val:,.2f})")
            else:
                print(f"   🎯 Strategy Actions: No recent actions recorded")
            
            # Runtime stats
            runtime_stats = self.runtime_stats.get(strategy_name, {})
            avg_runtime = runtime_stats.get('avg_runtime', 0) * 1000  # Convert to milliseconds
            timeout_count = runtime_stats.get('timeout_count', 0)
            action_count = runtime_stats.get('action_count', 0)
            timeout_rate = (timeout_count / action_count * 100) if action_count > 0 else 0
            
            print(f"   ⏱️  Performance: {avg_runtime:.2f}ms avg, {timeout_count} timeouts ({timeout_rate:.1f}%)")
        
        # Global stats
        pending_orders = len(self.pending_orders)
        print(f"\n🌐 Global Stats:")
        print(f"   ⏳ Pending orders: {pending_orders}")
        print(f"   📅 Current phase: {'CALIBRATION' if current_timestep <= self.config.calibration_end_time else 'VALIDATION'}")
        print(f"   🕐 Network latency: {self.config.network_latency_delta:.3f}s")
        print(f"{'='*80}\n")
