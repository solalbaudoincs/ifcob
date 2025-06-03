"""
Test script to verify that the preprocessing module works correctly.
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocessing import (
    preprocess_crypto_data, 
    preprocess_data_folder,
    preprocessing_vectorized,
    clean_duplicates,
    pivot_to_wide_format
)
import pandas as pd


def test_individual_functions():
    """Test individual preprocessing functions."""
    print("=== Testing Individual Functions ===")
    
    try:
        # Load sample data
        df = pd.read_csv("../data/raw/DATA_0/XBT_EUR.csv")
        print(f"Loaded sample data: {df.shape}")
        
        # Test preprocessing_vectorized
        df_processed = preprocessing_vectorized(df.head(1000), block_size=10)
        print(f"After vectorized preprocessing: {df_processed.shape}")
        
        # Test clean_duplicates
        df_clean = clean_duplicates(df_processed)
        print(f"After cleaning duplicates: {df_clean.shape}")
        
        # Test pivot_to_wide_format
        df_wide = pivot_to_wide_format(df_clean)
        print(f"After pivot to wide format: {df_wide.shape}")
        print(f"Sample columns: {list(df_wide.columns[:5])}")
        
        print("✓ Individual functions test passed!")
        
    except Exception as e:
        print(f"✗ Individual functions test failed: {e}")


def test_single_file_processing():
    """Test single file processing."""
    print("\n=== Testing Single File Processing ===")
    
    try:
        df_wide = preprocess_crypto_data(
            input_file="../data/raw/DATA_0/XBT_EUR.csv",
            output_file="../data/test_preprocessed/single_test/XBT_EUR.parquet",
            coin='XBT',
            block_size=20
        )
        
        print(f"✓ Single file processing passed! Output shape: {df_wide.shape}")
        
    except Exception as e:
        print(f"✗ Single file processing failed: {e}")


def test_folder_processing():
    """Test folder processing."""
    print("\n=== Testing Folder Processing ===")
    
    try:
        results = preprocess_data_folder(
            input_folder="../data/raw/DATA_0",
            output_folder="../data/test_preprocessed/folder_test",
            coins=['XBT', 'ETH'],
            block_size=20
        )
        
        print(f"✓ Folder processing passed! Processed {len(results)} files:")
        for coin, df in results.items():
            print(f"  {coin}: {df.shape}")
            
    except Exception as e:
        print(f"✗ Folder processing failed: {e}")


def test_package_imports():
    """Test that all package imports work correctly."""
    print("\n=== Testing Package Imports ===")
    
    try:
        # Test importing from package
        from preprocessing import preprocess_crypto_data, preprocess_data_folder
        from preprocessing import preprocessing_vectorized, clean_duplicates, pivot_to_wide_format
        
        print("✓ All package imports successful!")
        
    except Exception as e:
        print(f"✗ Package imports failed: {e}")


if __name__ == "__main__":
    print("Running Preprocessing Module Tests")
    print("=" * 50)
    
    test_package_imports()
    test_individual_functions()
    test_single_file_processing()
    test_folder_processing()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("Check ../data/test_preprocessed/ for output files.")
