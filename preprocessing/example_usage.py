"""
Example usage of the preprocessing module.

This script demonstrates how to use the preprocessing functions
both as imported functions and as a standalone module.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the preprocessing functions
from preprocessing import preprocess_crypto_data, preprocess_data_folder


def example_single_file():
    """Example of processing a single file."""
    print("=== Single File Processing Example ===")
    
    # Define paths (adjust these to match your actual data structure)
    input_file = "../data/raw/DATA_0/XBT_EUR.csv"
    output_file = "../data/preprocessed/DATA_0/XBT_EUR.parquet"
    
    try:
        # Process single file
        df_wide = preprocess_crypto_data(
            input_file=input_file,
            output_file=output_file,
            coin='XBT',
            block_size=20
        )
        
        print(f"Successfully processed single file!")
        print(f"Output shape: {df_wide.shape}")
        print(f"Columns: {list(df_wide.columns[:5])}...")  # Show first 5 columns
        
    except Exception as e:
        print(f"Error in single file processing: {e}")


def example_folder_processing():
    """Example of processing an entire folder."""
    print("\n=== Folder Processing Example ===")
    
    # Define paths
    input_folder = "../data/raw/DATA_0"
    output_folder = "../data/preprocessed/DATA_0"
    
    try:
        # Process entire folder
        results = preprocess_data_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            coins=['XBT', 'ETH'],  # Specify coins or leave as None for all
            block_size=20
        )
        
        print(f"Successfully processed folder!")
        print(f"Processed {len(results)} files:")
        for coin, df in results.items():
            print(f"  {coin}: {df.shape}")
            
    except Exception as e:
        print(f"Error in folder processing: {e}")


def example_custom_usage():
    """Example of using individual preprocessing functions."""
    print("\n=== Custom Usage Example ===")
    
    try:
        import pandas as pd
        from preprocessing import preprocessing_vectorized, clean_duplicates, pivot_to_wide_format
        
        # Load data manually
        df = pd.read_csv("../data/raw/DATA_0/XBT_EUR.csv")
        print(f"Loaded {len(df)} rows")
        
        # Apply preprocessing steps individually
        df_processed = preprocessing_vectorized(df, block_size=20)
        df_clean = clean_duplicates(df_processed)
        df_wide = pivot_to_wide_format(df_clean)
        
        print(f"Custom preprocessing completed: {df_wide.shape}")
        
    except Exception as e:
        print(f"Error in custom usage: {e}")


if __name__ == "__main__":
    print("Cryptocurrency Data Preprocessing Examples")
    print("=" * 50)
    
    # Run examples
    example_single_file()
    example_folder_processing()
    example_custom_usage()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output files in ../data/preprocessed/")
