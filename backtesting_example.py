"""
Backtesting Architecture Example
This demonstrates the complete backtesting system as per the consulting-style deliverable requirements.
"""

from backtesting.types import MarketData, Action, FeesGraph
from backtesting.strategy import Strategy
from backtesting.portolio import Portfolio
from backtesting.backtest import Backtester, BacktestConfig
from backtesting.dataloader import OrderBookDataFromDf


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
            ('XBT_EUR', 0.001),  # 0.1% fee to buy XBT with EURC
            ('ETH_EUR', 0.001),  # 0.1% fee to buy ETH with EURC
        ],
        'XBT_EUR': [
            ('EURC', 0.0015),    # 0.15% fee to sell XBT for EURC (slightly higher)
            ('ETH_EUR', 0.002),  # 0.2% fee for XBT->ETH direct trade
        ],
        'ETH_EUR': [
            ('EURC', 0.0015),    # 0.15% fee to sell ETH for EURC
            ('XBT_EUR', 0.002),  # 0.2% fee for ETH->XBT direct trade
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
        ('XBT_EUR', 'data/preprocessed/DATA_0/XBT_EUR.parquet'),
        ('ETH_EUR', 'data/preprocessed/DATA_0/ETH_EUR.parquet')
    ]
    
    # Initialize dataloader first to get actual timestamps
    dataloader = OrderBookDataFromDf(data_sources)
    
    # Get actual timestamp values from the data
    all_timesteps = dataloader.get_time_step_values()
    min_timestamp = min(min(timesteps) for timesteps in all_timesteps.values())
    max_timestamp = max(max(timesteps) for timesteps in all_timesteps.values())
    
    # Split at 70% for calibration/validation
    split_timestamp = min_timestamp + 0.7 * (max_timestamp - min_timestamp)
    
    # Configuration for backtesting - uses actual timesteps from data
    config = BacktestConfig(
        initial_capital=10000.0,
        fees_graph=fees_graph,
        symbols=['XBT_EUR', 'ETH_EUR'],
        window_size=10,  # Number of last rows for windowed market data
        
        # Proper validation split using actual timestamps
        calibration_end_time=split_timestamp,  # End of in-sample data
        validation_start_time=split_timestamp   # Start of out-of-sample data
    )
    
    backtester = Backtester(dataloader, config)
    
    print("Backtesting Architecture Initialized Successfully")
    strategies = [SimpleExampleStrategy()]
    
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
    print("=== CRYPTO BACKTESTING ARCHITECTURE ===")
    print("Consulting-Style Deliverable Implementation")
    print("Addresses all key research objective points from slides\n")
    
    #show_architecture_capabilities()
    print("\n" + "="*50)
    demonstrate_backtesting_architecture()