"""
Data preprocessing module for cryptocurrency data.

This module provides functions to preprocess cryptocurrency CSV data,
converting it to a wide format suitable for feature extraction and analysis.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union


def preprocessing_vectorized(df: pd.DataFrame, block_size: int = 20) -> pd.DataFrame:
    """
    Preprocess dataframe by creating blocks of data and aligning timestamps.
    
    Args:
        df: Input DataFrame with cryptocurrency data
        block_size: Size of each block for grouping rows (default: 20)
    
    Returns:
        Preprocessed DataFrame with row_id and aligned timestamps
    """
    df = df.copy()
    n = len(df)

    # Create block ID: 0,0,...,1,1,...,2,2,...
    df['row_id'] = np.repeat(np.arange((n + block_size - 1) // block_size), block_size)[:n]

    # Align timestamp by block (take max timestamp per block)
    df['timestamp'] = df.groupby('row_id')['timestamp'].transform('max')

    return df


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates based on timestamp, row_id, level, and side columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    # Check for duplicates
    dups = df.duplicated(subset=['timestamp', 'row_id', 'level', 'side'], keep=False)
    
    if dups.any():
        print(f"Found {dups.sum()} duplicate rows, removing them...")
    
    return df[~dups]


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to wide format suitable for feature extraction.
    
    Args:
        df: Input DataFrame with columns: timestamp, row_id, level, side, price, volume
    
    Returns:
        Wide format DataFrame with multi-level columns
    """
    # Pivot to wide format
    df_wide = df.pivot(
        index=['timestamp', 'row_id'], 
        columns=['level', 'side'], 
        values=['price', 'volume']
    )
    
    # Flatten the multi-index columns to desired format
    df_wide.columns = [
        f"level-{int(level)}-{side}-{var}"
        for var, level, side in df_wide.columns.tolist()
    ]
    
    return df_wide


def preprocess_crypto_data(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    coin: str = 'XBT',
    block_size: int = 20
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for cryptocurrency data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output parquet file
        coin: Cryptocurrency symbol (for logging purposes)
        block_size: Size of each block for grouping rows
    
    Returns:
        Preprocessed DataFrame in wide format
    """
    print(f"Processing {coin} data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply vectorized preprocessing
    df_processed = preprocessing_vectorized(df, block_size=block_size)
    print(f"Applied vectorized preprocessing with block_size={block_size}")
    
    # Clean duplicates
    df_clean = clean_duplicates(df_processed)
    print(f"After cleaning: {len(df_clean)} rows")
    
    # Convert to wide format
    df_wide = pivot_to_wide_format(df_clean)
    print(f"Converted to wide format: {df_wide.shape}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df_wide.to_parquet(output_file, index=True)
    print(f"Saved preprocessed data to {output_file}")
    
    return df_wide


def preprocess_data_folder(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    coins: Optional[list] = None,
    block_size: int = 20
) -> dict:
    """
    Preprocess all cryptocurrency data files in a folder.
    
    Args:
        input_folder: Path to input folder containing CSV files
        output_folder: Path to output folder for parquet files
        coins: List of coin symbols to process. If None, processes all CSV files
        block_size: Size of each block for grouping rows
    
    Returns:
        Dictionary with coin symbols as keys and processed DataFrames as values
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files to process
    if coins is None:
        csv_files = list(input_path.glob("*.csv"))
        coins = [f.stem.split('_')[0] for f in csv_files]  # Extract coin name from filename
    
    results = {}
    
    for coin in coins:
        try:
            input_file = input_path / f"{coin}_EUR.csv"
            output_file = output_path / f"{coin}_EUR.parquet"
            
            if input_file.exists():
                df_wide = preprocess_crypto_data(
                    input_file=input_file,
                    output_file=output_file,
                    coin=coin,
                    block_size=block_size
                )
                results[coin] = df_wide
            else:
                print(f"Warning: Input file {input_file} not found, skipping {coin}")
        
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(results)} files")
    return results
