"""
Integration example showing how to use the preprocessing module
in a typical cryptocurrency data analysis workflow.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocessing import preprocess_data_folder, preprocess_crypto_data


def main():
    """Main integration example."""
    print("Cryptocurrency Data Preprocessing Integration Example")
    print("=" * 60)
    
    # Define data paths
    raw_data_path = "../data/raw/DATA_0"
    processed_data_path = "../data/integration_example"
    
    print(f"Raw data folder: {raw_data_path}")
    print(f"Output folder: {processed_data_path}")
    
    # Method 1: Process all files in a folder
    print("\n1. Processing all files in folder...")
    try:
        results = preprocess_data_folder(
            input_folder=raw_data_path,
            output_folder=processed_data_path,
            block_size=20
        )
        
        print(f"✓ Successfully processed {len(results)} files:")
        for coin, df in results.items():
            print(f"  {coin}: {df.shape[0]:,} rows, {df.shape[1]} features")
    
    except Exception as e:
        print(f"✗ Folder processing failed: {e}")
    
    # Method 2: Process specific coins
    print("\n2. Processing specific coins...")
    try:
        specific_results = preprocess_data_folder(
            input_folder=raw_data_path,
            output_folder=f"{processed_data_path}/specific",
            coins=['XBT'],  # Only process XBT
            block_size=25   # Custom block size
        )
        
        print(f"✓ Successfully processed specific coins:")
        for coin, df in specific_results.items():
            print(f"  {coin}: {df.shape[0]:,} rows, {df.shape[1]} features")
    
    except Exception as e:
        print(f"✗ Specific coin processing failed: {e}")
    
    # Method 3: Process single file with custom parameters
    print("\n3. Processing single file with custom parameters...")
    try:
        single_result = preprocess_crypto_data(
            input_file=f"{raw_data_path}/XBT_EUR.csv",
            output_file=f"{processed_data_path}/custom/XBT_EUR_custom.parquet",
            coin='XBT',
            block_size=15
        )
        
        print(f"✓ Single file processing completed:")
        print(f"  Shape: {single_result.shape[0]:,} rows, {single_result.shape[1]} features")
        print(f"  Sample columns: {list(single_result.columns[:5])}")
    
    except Exception as e:
        print(f"✗ Single file processing failed: {e}")
    
    # Show how to load and analyze the processed data
    print("\n4. Loading and analyzing processed data...")
    try:
        # Load processed data
        processed_df = pd.read_parquet(f"{processed_data_path}/XBT_EUR.parquet")
        
        print(f"✓ Loaded processed data:")
        print(f"  Shape: {processed_df.shape}")
        print(f"  Index levels: {processed_df.index.names}")
        print(f"  Date range: {processed_df.index.get_level_values('timestamp').min()} to {processed_df.index.get_level_values('timestamp').max()}")
        
        # Show sample of the data structure
        print(f"\n  Sample data structure:")
        print(processed_df.head(2))
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
    
    print("\n" + "=" * 60)
    print("Integration example completed!")
    print(f"Check the output folder: {processed_data_path}")


if __name__ == "__main__":
    main()
