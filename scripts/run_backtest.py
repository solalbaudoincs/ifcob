#!/usr/bin/env python3
"""
run_backtest.py
===============
Script to run backtests using strategies from the strategies folder.

Usage examples:
---------------
Run a single strategy:
    python run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --window-size 5

Run multiple strategies with data index:
    python run_backtest.py --strategy Mateo2StartStrategy --strategy RFPredAllSignedStratMateoCheating --data-index 2

Custom data and parameters:
    python run_backtest.py --strategy RFPredAllSignedStratMateo --data-index 1 --data-sources XBT:data/custom/XBT.parquet ETH:data/custom/ETH.parquet --initial-capital 500000

Save results:
    python run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --output-dir backtest_results --save-trades

Profile execution:
    python run_backtest.py --strategy Mateo2StartStrategy --data-index 2 --profile

Author: GitHub Copilot
Date: 2025-01-06
"""

import sys
import os
import argparse
import importlib.util
import json
import time
import inspect
import traceback
import cProfile
import builtins
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import backtesting components
from backtesting.types import MarketData, Action, FeesGraph
from backtesting.strategy import Strategy
from backtesting.portfolio import Portfolio
from backtesting.backtest import Backtester, BacktestConfig
from backtesting.dataloader import OrderBookDataFromDf


def create_fees_graph(fee_percentage: float = 0.1) -> FeesGraph:
    """
    Create a fees graph with configurable fee percentage.
    
    Args:
        fee_percentage: Trading fee as percentage (0.1 = 0.1%)
    
    Returns:
        FeesGraph with symmetric trading fees
    """
    fee_rate = fee_percentage / 100.0
    
    return {
        'EURC': [
            ('XBT', fee_rate),
            ('ETH', fee_rate),
        ],
        'XBT': [
            ('EURC', fee_rate),
            ('ETH', fee_rate),
        ],
        'ETH': [
            ('EURC', fee_rate),
            ('XBT', fee_rate),
        ]
    }


def discover_strategies():
    """
    Dynamically discover all available strategies in the strategies folder.
    
    Returns:
        Dict mapping strategy names to their classes
    """
    strategies_dir = os.path.join(PROJECT_ROOT, 'strategies')
    strategy_classes = {}
    
    # Import all strategy files and extract Strategy classes
    for py_file in Path(strategies_dir).glob('*.py'):
        if py_file.name.startswith('__'):
            continue
            
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all Strategy subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Strategy) and 
                        attr != Strategy):
                        strategy_classes[attr_name] = attr
                        
            except Exception as e:
                print(f"Warning: Could not load strategies from {py_file}: {e}")
    
    return strategy_classes


def requires_data_index(strategy_class):
    """
    Check if a strategy class requires a data_index parameter.
    
    Args:
        strategy_class: The strategy class to inspect
    
    Returns:
        bool: True if the strategy requires data_index parameter
    """
    try:
        sig = inspect.signature(strategy_class.__init__)
        return 'data_index' in sig.parameters
    except Exception:
        return False


def strategy_loads_precomputed_features(strategy_class):
    """
    Check if a strategy loads precomputed features by examining its source code.
    
    Args:
        strategy_class: The strategy class to inspect
    
    Returns:
        bool: True if the strategy appears to load precomputed features
    """
    try:
        source = inspect.getsource(strategy_class)
        # Look for patterns that indicate loading precomputed features
        indicators = [
            'read_parquet',
            'joblib.load',
            'DATA_',
            'features/',
            'predictors/',
            '.joblib'
        ]
        return any(indicator in source for indicator in indicators)
    except Exception:
        return False


def get_strategy_instance(strategy_name: str, data_index: int, **kwargs):
    """
    Create an instance of the specified strategy with required data_index parameter.
    
    Args:
        strategy_name: Name of the strategy class
        data_index: Data index for strategies that load precomputed features (required)
        **kwargs: Additional parameters to pass to strategy constructor
    
    Returns:
        Strategy instance
    """
    available_strategies = discover_strategies()
    
    if strategy_name not in available_strategies:
        raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {list(available_strategies.keys())}")
    
    strategy_class = available_strategies[strategy_name]
    
    # Always pass data_index if strategy accepts it
    if requires_data_index(strategy_class):
        kwargs['data_index'] = data_index
    elif strategy_loads_precomputed_features(strategy_class):
        # Strategy loads features but doesn't have data_index parameter - warn user
        print(f"Warning: Strategy '{strategy_name}' loads precomputed features but doesn't accept data_index parameter.")
        print(f"The strategy may be using hardcoded paths instead of DATA_{data_index}.")
    
    # Filter kwargs to only include parameters the strategy accepts
    sig = inspect.signature(strategy_class.__init__)
    valid_params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Try to create strategy instance and provide helpful error messages
    try:
        return strategy_class(**valid_params)
    except FileNotFoundError as e:
        if 'predictors/' in str(e) or '.joblib' in str(e):
            print(f"Error: Model file not found for strategy '{strategy_name}': {e}")
            print("This strategy requires a pre-trained model. Please ensure the model file exists or train the model first.")
        elif 'data/features/' in str(e) or '.parquet' in str(e):
            print(f"Error: Feature file not found for strategy '{strategy_name}': {e}")
            print("This strategy requires precomputed features. They will be generated automatically.")
        raise
    except Exception as e:
        print(f"Error creating strategy '{strategy_name}': {e}")
        raise


def parse_data_sources(data_source_args):
    """
    Parse data source arguments in the format SYMBOL:PATH.
    
    Args:
        data_source_args: List of strings like ['XBT:path/to/data.parquet', 'ETH:path/to/data.parquet']
    
    Returns:
        List of (symbol, path) tuples
    """
    data_sources = []
    for arg in data_source_args:
        if ':' not in arg:
            raise ValueError(f"Data source must be in format SYMBOL:PATH, got: {arg}")
        symbol, path = arg.split(':', 1)
        data_sources.append((symbol, path))
    return data_sources


def generate_default_data_sources(data_index: int):
    """
    Generate default data sources based on data index.
    
    Args:
        data_index: Data index to use for default paths (required)
    
    Returns:
        List of (symbol, path) tuples
    """
    return [
        ('XBT', f'data/features/DATA_{data_index}/XBT_EUR.parquet'),
        ('ETH', f'data/features/DATA_{data_index}/ETH_EUR.parquet')
    ]


def save_results(results, output_dir, timestamp_str, save_trades=False, save_portfolio=False):
    """
    Save backtest results to files.
    
    Args:
        results: Backtest results dictionary
        output_dir: Directory to save results
        timestamp_str: Timestamp string for file naming
        save_trades: Whether to save detailed trade information
        save_portfolio: Whether to save portfolio evolution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary results
    summary_file = os.path.join(output_dir, f'backtest_summary_{timestamp_str}.csv')
    summary_data = []
    
    for strategy_name, (cal_result, val_result) in results.items():
        summary_data.append({
            'strategy': strategy_name,
            'phase': 'calibration',
            'total_return': cal_result.total_return,
            'sharpe_ratio': cal_result.sharpe_ratio,
            'max_drawdown': getattr(cal_result, 'max_drawdown', None),
            'num_trades': len(cal_result.trades),
            'final_value': cal_result.final_portfolio_value
        })
        summary_data.append({
            'strategy': strategy_name,
            'phase': 'validation',
            'total_return': val_result.total_return,
            'sharpe_ratio': val_result.sharpe_ratio,
            'max_drawdown': getattr(val_result, 'max_drawdown', None),
            'num_trades': len(val_result.trades),
            'final_value': val_result.final_portfolio_value
        })
    
    import pandas as pd
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    # Save detailed results as JSON
    results_file = os.path.join(output_dir, f'backtest_results_{timestamp_str}.json')
    json_data = {}
    
    for strategy_name, (cal_result, val_result) in results.items():
        json_data[strategy_name] = {
            'calibration': {
                'total_return': cal_result.total_return,
                'sharpe_ratio': cal_result.sharpe_ratio,
                'final_portfolio_value': cal_result.final_portfolio_value,
                'num_trades': len(cal_result.trades)
            },
            'validation': {
                'total_return': val_result.total_return,
                'sharpe_ratio': val_result.sharpe_ratio,
                'final_portfolio_value': val_result.final_portfolio_value,
                'num_trades': len(val_result.trades)
            }
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    # Save trade details if requested
    if save_trades:
        for strategy_name, (cal_result, val_result) in results.items():
            # Calibration trades
            if cal_result.trades:
                cal_trades_file = os.path.join(output_dir, f'trades_details_{strategy_name}_cal_{timestamp_str}.csv')
                trades_df = pd.DataFrame([
                    {
                        'timestamp': trade.timestamp,
                        'symbol': trade.symbol,
                        'amount': trade.amount,
                        'price': trade.price,
                        'fees': trade.fees,
                        'trade_type': 'buy' if trade.amount > 0 else 'sell'
                    }
                    for trade in cal_result.trades
                ])
                trades_df.to_csv(cal_trades_file, index=False)
            
            # Validation trades
            if val_result.trades:
                val_trades_file = os.path.join(output_dir, f'trades_details_{strategy_name}_val_{timestamp_str}.csv')
                trades_df = pd.DataFrame([
                    {
                        'timestamp': trade.timestamp,
                        'symbol': trade.symbol,
                        'amount': trade.amount,
                        'price': trade.price,
                        'fees': trade.fees,
                        'trade_type': 'buy' if trade.amount > 0 else 'sell'
                    }
                    for trade in val_result.trades
                ])
                trades_df.to_csv(val_trades_file, index=False)
    
    # Save portfolio evolution if requested
    if save_portfolio:
        for strategy_name, (cal_result, val_result) in results.items():
            if hasattr(cal_result, 'portfolio_history') and cal_result.portfolio_history:
                cal_portfolio_file = os.path.join(output_dir, f'portfolio_evolution_{strategy_name}_cal_{timestamp_str}.csv')
                portfolio_df = pd.DataFrame(cal_result.portfolio_history)
                portfolio_df.to_csv(cal_portfolio_file, index=False)
            
            if hasattr(val_result, 'portfolio_history') and val_result.portfolio_history:
                val_portfolio_file = os.path.join(output_dir, f'portfolio_evolution_{strategy_name}_val_{timestamp_str}.csv')
                portfolio_df.to_csv(val_portfolio_file, index=False)


def check_precomputed_features_exist(data_index: int, symbols: list = None) -> bool:
    """
    Check if precomputed features exist for the specified data index.
    
    Args:
        data_index: Data index to check
        symbols: List of symbols to check (default: ['XBT', 'ETH'])
    
    Returns:
        bool: True if all required feature files exist
    """
    if symbols is None:
        symbols = ['XBT', 'ETH']
    
    features_dir = os.path.join(PROJECT_ROOT, 'data', 'features', f'DATA_{data_index}')
    
    for symbol in symbols:
        feature_file = os.path.join(features_dir, f'{symbol}_EUR.parquet')
        if not os.path.exists(feature_file):
            return False
    
    return True


def generate_precomputed_features(data_index: int, symbols: list = None, force_regenerate: bool = False):
    """
    Generate precomputed features for the specified data index and symbols.
    
    Args:
        data_index: Data index to generate features for
        symbols: List of symbols to generate features for (default: ['XBT', 'ETH'])
        force_regenerate: Whether to regenerate features even if they exist
    """
    if symbols is None:
        symbols = ['XBT', 'ETH']
    
    print(f"Generating precomputed features for DATA_{data_index}...")
    
    try:
        # Import the feature generation module
        feature_gen_script = os.path.join(PROJECT_ROOT, 'scripts', 'generate_features.py')
        
        if not os.path.exists(feature_gen_script):
            raise FileNotFoundError("Feature generation script not found. Please ensure scripts/generate_features.py exists.")
        
        # Import the feature generation functions
        sys.path.insert(0, os.path.dirname(feature_gen_script))
        spec = importlib.util.spec_from_file_location("generate_features", feature_gen_script)
        feature_gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_gen_module)
        
        # Generate features for each symbol
        for symbol in symbols:
            print(f"  Generating features for {symbol}...")
            
            success = feature_gen_module.generate_features_for_coin(
                coin=symbol,
                data_version=data_index,
                feature_names=None,  # Generate all features
                input_dir=os.path.join(PROJECT_ROOT, 'data', 'preprocessed'),
                output_dir=os.path.join(PROJECT_ROOT, 'data', 'features'),
                overwrite=force_regenerate,
                verbose=True
            )
            
            if not success:
                raise RuntimeError(f"Failed to generate features for {symbol}")
        
        print(f"✓ Successfully generated precomputed features for DATA_{data_index}")
        
    except Exception as e:
        print(f"✗ Error generating precomputed features: {e}")
        raise


def ensure_precomputed_features(data_index: int, strategies: list, force_check: bool = False):
    """
    Ensure that precomputed features exist for strategies that need them.
    
    Args:
        data_index: Data index to check/generate features for
        strategies: List of strategy instances to check
        force_check: Whether to force check even for strategies without data_index support
    """
    # Check if any strategy needs precomputed features
    needs_features = False
    strategy_symbols = set(['XBT', 'ETH'])  # Default symbols
    
    available_strategies = discover_strategies()
    
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        strategy_class = available_strategies.get(strategy_name)
        
        if strategy_class:
            if requires_data_index(strategy_class) or strategy_loads_precomputed_features(strategy_class):
                needs_features = True
                print(f"Strategy '{strategy_name}' requires precomputed features")
                break
    
    if not needs_features and not force_check:
        print("No strategies require precomputed features. Skipping feature check.")
        return
    
    # Check if features exist
    if check_precomputed_features_exist(data_index, list(strategy_symbols)):
        print(f"✓ Precomputed features for DATA_{data_index} are available")
        return
    
    # Features don't exist, generate them
    print(f"⚠ Precomputed features for DATA_{data_index} not found")
    
    response = input(f"Generate precomputed features for DATA_{data_index}? [Y/n]: ").strip().lower()
    if response in ['', 'y', 'yes']:
        try:
            generate_precomputed_features(data_index, list(strategy_symbols))
        except Exception as e:
            print(f"Failed to generate precomputed features: {e}")
            print("You may need to generate features manually using:")
            print(f"  python scripts/generate_features.py --coin {' '.join(strategy_symbols)} --data-version {data_index}")
            raise
    else:
        print("Skipping feature generation. Note: Strategies requiring precomputed features may fail.")


def run_backtest(args):
    """
    Run the backtest with specified parameters.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("BACKTESTING ENGINE")
    print("=" * 60)
    
    # Create timestamp for file naming
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Parse data sources
    if args.data_sources:
        data_sources = parse_data_sources(args.data_sources)
        print(f"Using custom data sources: {data_sources}")
    else:
        # Generate default data sources based on data_index (required)
        data_sources = generate_default_data_sources(args.data_index)
        print(f"Using default data sources for DATA_{args.data_index}: {data_sources}")
    
    print(f"Data index: {args.data_index}")
    
    # Create strategies
    strategies = []
    strategy_kwargs = {}
    
    # Add strategy-specific parameters
    if args.window_size is not None:
        strategy_kwargs['window_size'] = args.window_size
    if args.threshold is not None:
        strategy_kwargs['threshold'] = args.threshold
    
    for strategy_name in args.strategy:
        try:
            # Check if strategy loads precomputed features before trying to create instance
            available_strategies = discover_strategies()
            strategy_class = available_strategies.get(strategy_name)
            
            if strategy_class and strategy_loads_precomputed_features(strategy_class):
                if requires_data_index(strategy_class):
                    print(f"Strategy '{strategy_name}' uses precomputed features with data_index={args.data_index}")
                else:
                    print(f"Strategy '{strategy_name}' uses precomputed features (may use hardcoded paths)")
            
            strategy = get_strategy_instance(strategy_name, data_index=args.data_index, **strategy_kwargs)
            strategies.append(strategy)
            print(f"Loaded strategy: {strategy_name}")
                    
        except FileNotFoundError as e:
            if 'predictors/' in str(e) or '.joblib' in str(e):
                print(f"Skipping strategy {strategy_name}: Model file not found - {e}")
                print("Please ensure the required model files exist or train the models first.")
            elif 'data/features/' in str(e):
                print(f"Strategy {strategy_name} requires precomputed features that will be generated automatically.")
                # Continue to feature generation step
                continue
            else:
                print(f"Error loading strategy {strategy_name}: {e}")
                continue
        except Exception as e:
            print(f"Error loading strategy {strategy_name}: {e}")
            continue
    
    if not strategies:
        print("No valid strategies loaded.")
        print("This might be due to:")
        print("  1. Missing model files (*.joblib) - please train models first")
        print("  2. Missing precomputed features - will be generated automatically if needed")
        print("  3. Invalid strategy names - use --list-strategies to see available strategies")
        return
    
    # Check and ensure precomputed features are available
    print("\n" + "=" * 60)
    print("CHECKING PRECOMPUTED FEATURES")
    print("=" * 60)
    
    try:
        ensure_precomputed_features(args.data_index, strategies, args.force_feature_check)
    except Exception as e:
        print(f"Error ensuring precomputed features: {e}")
        if not args.ignore_feature_errors:
            print("Exiting due to feature generation failure. Use --ignore-feature-errors to continue anyway.")
            return
        else:
            print("Continuing despite feature generation failure...")
    
    # Create fees graph
    fees_graph = create_fees_graph(args.fee_percentage)
    print(f"\nTrading fees: {args.fee_percentage}%")
    
    # Initialize dataloader
    try:
        dataloader = OrderBookDataFromDf(data_sources)
        print("Dataloader initialized successfully")
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        return
    
    # Get timestamp range for validation split
    all_timesteps = dataloader.get_time_step_values()
    min_timestamp = min(min(timesteps) for timesteps in all_timesteps.values())
    max_timestamp = max(max(timesteps) for timesteps in all_timesteps.values())
    split_timestamp = min_timestamp + args.split_ratio * (max_timestamp - min_timestamp)
    
    print(f"Data timestamp range: {min_timestamp} to {max_timestamp}")
    print(f"Calibration/validation split at: {split_timestamp} ({args.split_ratio:.1%})")
    
    # Configure backtester
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        fees_graph=fees_graph,
        symbols=[symbol for symbol, _ in data_sources],
        window_size=args.window_size or 10,
        calibration_end_time=split_timestamp,
        validation_start_time=split_timestamp,
        max_action_runtime=args.max_action_runtime
    )
    
    backtester = Backtester(dataloader, config)
    print(f"Backtester configured with initial capital: ${args.initial_capital:,.0f}")
    
    # Run backtest
    print("\nStarting backtest...")
    start_time = time.time()
    
    try:
        results = backtester.backtest(strategies)
        elapsed_time = time.time() - start_time
        
        print(f"\nBacktest completed successfully in {elapsed_time:.2f} seconds!")
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        for strategy_name, (cal_result, val_result) in results.items():
            print(f"\nStrategy: {strategy_name}")
            print("-" * 40)
            print(f"Calibration:")
            print(f"  Total Return: {cal_result.total_return:.2%}")
            print(f"  Sharpe Ratio: {cal_result.sharpe_ratio:.3f}")
            print(f"  Final Value: ${cal_result.final_portfolio_value:,.2f}")
            print(f"  Total Trades: {len(cal_result.trades)}")
            
            print(f"Validation:")
            print(f"  Total Return: {val_result.total_return:.2%}")
            print(f"  Sharpe Ratio: {val_result.sharpe_ratio:.3f}")
            print(f"  Final Value: ${val_result.final_portfolio_value:,.2f}")
            print(f"  Total Trades: {len(val_result.trades)}")
        
        # Display runtime statistics
        if hasattr(backtester, 'runtime_stats'):
            print("\n" + "=" * 60)
            print("RUNTIME STATISTICS")
            print("=" * 60)
            for strategy_name, stats in backtester.runtime_stats.items():
                print(f"\n{strategy_name}:")
                print(f"  Average Runtime: {stats['avg_runtime']:.6f}s")
                print(f"  Total Actions: {stats['action_count']}")
                print(f"  Timeouts: {stats['timeout_count']}")
        
        # Save results if output directory specified
        if args.output_dir:
            save_results(results, args.output_dir, timestamp_str, 
                        args.save_trades, args.save_portfolio)
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run backtests using strategies from the strategies folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1
  python run_backtest.py --strategy Mateo2StartStrategy --data-index 2
  python run_backtest.py --strategy RFPredAllSignedStratMateoCheating --data-index 1 --window-size 5
  python run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1 --output-dir results --save-trades
        """
    )
    
    # Strategy selection
    parser.add_argument('--strategy', action='append',
                       help='Strategy class name to run (can be specified multiple times)')
    
    # Data configuration - data-index is now required (except for list-strategies)
    parser.add_argument('--data-index', type=int,
                       help='Data index for loading features (e.g., 1 for DATA_1, 2 for DATA_2) - REQUIRED')
    parser.add_argument('--data-sources', nargs='*',
                       help='Override default data sources in format SYMBOL:PATH (e.g., XBT:data/XBT.parquet)')
    parser.add_argument('--split-ratio', type=float, default=0.7,
                       help='Ratio for calibration/validation split (default: 0.7)')
    
    # Strategy parameters
    parser.add_argument('--window-size', type=int,
                       help='Window size parameter for strategies that support it')
    parser.add_argument('--threshold', type=float,
                       help='Threshold parameter for strategies that support it')
    
    # Backtesting configuration
    parser.add_argument('--initial-capital', type=float, default=1e6,
                       help='Initial capital for backtesting (default: 1,000,000)')
    parser.add_argument('--fee-percentage', type=float, default=0.1,
                       help='Trading fee percentage (default: 0.1%%)')
    parser.add_argument('--max-action-runtime', type=float, default=0.1,
                       help='Maximum runtime per strategy action in seconds (default: 0.1)')
    
    # Feature generation options
    parser.add_argument('--force-feature-check', action='store_true',
                       help='Force check for precomputed features even for strategies that may not need them')
    parser.add_argument('--ignore-feature-errors', action='store_true',
                       help='Continue with backtesting even if feature generation fails')
    parser.add_argument('--auto-generate-features', action='store_true',
                       help='Automatically generate features without prompting (non-interactive mode)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save backtest results')
    parser.add_argument('--save-trades', action='store_true',
                       help='Save detailed trade information')
    parser.add_argument('--save-portfolio', action='store_true',
                       help='Save portfolio evolution over time')
    
    # Debugging and profiling
    parser.add_argument('--profile', action='store_true',
                       help='Profile execution performance')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List all available strategies and exit')
    
    args = parser.parse_args()
    
    # List strategies if requested
    if args.list_strategies:
        strategies = discover_strategies()
        print("Available strategies:")
        print("=" * 50)
        for name, cls in strategies.items():
            description = cls.__doc__.strip().split('.')[0] if cls.__doc__ else 'No description'
            needs_data_index = requires_data_index(cls)
            loads_features = strategy_loads_precomputed_features(cls)
            
            print(f"  {name}:")
            print(f"    Description: {description}")
            if needs_data_index:
                print(f"    ✓ Supports --data-index parameter")
            elif loads_features:
                print(f"    ⚠ Loads precomputed features (may need manual update for data-index)")
            else:
                print(f"    ℹ No precomputed features detected")
            print()
        print("Note: --data-index is required for all backtests")
        print("      Precomputed features will be automatically checked and generated if needed")
        return
    
    # Check required arguments for actual backtesting
    if not args.strategy:
        parser.error("--strategy is required when not using --list-strategies")
    if args.data_index is None:
        parser.error("--data-index is required when not using --list-strategies")
    
    # Set auto-generation mode if non-interactive
    if args.auto_generate_features:
        # Monkey patch input function to always return 'yes'
        builtins.input = lambda prompt: 'yes'
    
    # Run backtest
    if args.profile:
        profile_output = f"profile_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        print(f"Profiling enabled. Output: {profile_output}")
        cProfile.run('run_backtest(args)', profile_output)
        print(f"Profiling complete. Use: snakeviz {profile_output}")
    else:
        run_backtest(args)


if __name__ == "__main__":
    main()