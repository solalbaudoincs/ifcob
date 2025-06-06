"""
Backtesting Architecture Example
This demonstrates the complete backtesting system as per the consulting-style deliverable requirements.
"""

from backtesting.types import MarketData, Action, FeesGraph
from backtesting.strategy import Strategy
from backtesting.portfolio import Portfolio
from backtesting.backtest import Backtester, BacktestConfig
from backtesting.dataloader import OrderBookDataFromDf
from strategies.rf_pred_all_signed_strat_mateo import RFPredAllSignedStratMateo, RFPredAllSignedStratMateoCheating
from strategies.mateo_2_start import Mateo2StartStrategy
from strategies.trend_following import TFCumulativeReturnStrategy, TFSharpeRatioStrategy


class SimpleExampleStrategy(Strategy):
    """
    Example strategy implementation - NOT FOR PRODUCTION USE
    This is just to demonstrate the architecture
    """
    
    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold
        
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """
        Simple example strategy that doesn't implement any real logic.
        Real strategies should be implemented separately.
        """
        # Return empty action (no trading) for architecture demonstration
        return {}


def create_example_fees_graph() -> FeesGraph:
    """
    Create an example fees graph for demonstration.
    In practice, this would be based on actual exchange fee structures.
    """
    return {
        'EURC': [
            ('XBT', 0.00),  # 0.1% fee to buy XBT with EURC
            ('ETH', 0.001),  # 0.1% fee to buy ETH with EURC
        ],
        'XBT': [
            ('EURC', 0.00),    # 0.15% fee to sell XBT for EURC (slightly higher)
            ('ETH', 0.00),  # 0.2% fee for XBT->ETH direct trade
        ],
        'ETH': [
            ('EURC', 0.001),    # 0.1% fee to sell ETH for EURC
            ('XBT', 0.00),  # 0.2% fee for ETH->XBT direct trade
        ]
    }


def demonstrate_backtesting_architecture():
    """
    Demonstrates the complete backtesting architecture addressing all slide requirements:
    
    1. High-frequency signal handling (actual timesteps from data)
    2. Asynchronous treatment 
    3. Cross-crypto lead-lag signal extraction
    4. Proper calibration/validation data separation
    5. Transaction cost modeling with FeesGraph
    6. Performance metrics calculation
    """
    
    # Create fees graph for realistic transaction cost modeling
    fees_graph = create_example_fees_graph()
    
    # Real data sources using parquet files
    data_sources = [
        ('XBT', 'data/features/DATA_1/XBT_EUR.parquet'),
        ('ETH', 'data/features/DATA_1/ETH_EUR.parquet')
    ]
    
    # Initialize dataloader first to get actual timestamps
    dataloader = OrderBookDataFromDf(data_sources)
    
    # Get actual timestamp values from the data
    all_timesteps = dataloader.get_time_step_values()
    min_timestamp = min(min(timesteps) for timesteps in all_timesteps.values())
    max_timestamp = max(max(timesteps) for timesteps in all_timesteps.values())
    print(f"Data timestamps range: {min_timestamp} to {max_timestamp}")
    # Split at 70% for calibration/validation
    split_timestamp = min_timestamp + 0.7 * (max_timestamp - min_timestamp)
    
    # Configuration for backtesting - uses actual timesteps from data
    config = BacktestConfig(
        initial_capital=1e6,
        fees_graph=fees_graph,
        symbols=['XBT', 'ETH'],
        window_size=10,  # Number of last rows for windowed market data
        
        # Proper validation split using actual timestamps
        calibration_end_time=split_timestamp,  # End of in-sample data
        validation_start_time=split_timestamp   # Start of out-of-sample data
    )
    
    backtester = Backtester(dataloader, config)
    
    print("Backtesting Architecture Initialized Successfully")
    strategies = [Mateo2StartStrategy(data_index=1,model_path="predictors/mateo/target-avg_10ms_of_mid_price_itincreases_after_200ms_with_threshold_5_depth-5_nest-100/model.joblib")]
    
    print(f"Data timestamp range: {min_timestamp} to {max_timestamp}")
    print(f"Calibration/validation split at: {split_timestamp}")

    # Run the actual backtest
    try:
        results = backtester.backtest(strategies)
        print("\nBacktest completed successfully!")
        
        for strategy_name, (cal_result, val_result) in results.items():
            print(f"\nStrategy {strategy_name}:")
            print(f"  Calibration Return: {cal_result.total_return:.2%}")
            print(f"  Validation Return: {val_result.total_return:.2%}")
            print(f"  Calibration Sharpe: {cal_result.sharpe_ratio:.3f}")
            print(f"  Validation Sharpe: {val_result.sharpe_ratio:.3f}")
            print(f"  Total Trades (Cal): {len(cal_result.trades)}")
            print(f"  Total Trades (Val): {len(val_result.trades)}")
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true',
                        help='Profile execution for use with snakeviz')
    args = parser.parse_args()

    print("\n" + "="*50)
    if args.profile:
        import cProfile
        profile_output = "profile_backtesting_example.prof"
        print(f"Profiling enabled. Output: {profile_output}")
        cProfile.run('demonstrate_backtesting_architecture()', profile_output)
        print(f"Profiling complete. Use: snakeviz {profile_output}")
    else:
        demonstrate_backtesting_architecture()
