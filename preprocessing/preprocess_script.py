#!/usr/bin/env python3
"""
Standalone script for preprocessing cryptocurrency data.

Usage:
    python preprocess_script.py --input_folder INPUT --output_folder OUTPUT [options]

Examples:
    # Process all CSV files in a data folder
    python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0

    # Process specific coins
    python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0 --coins XBT ETH

    # Use custom block size
    python preprocess_script.py --input_folder ../data/raw/DATA_0 --output_folder ../data/preprocessed/DATA_0 --block_size 30
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from data_preprocessor import preprocess_data_folder, preprocess_crypto_data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess cryptocurrency data from CSV to parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        required=True,
        help='Path to input folder containing CSV files'
    )
    
    parser.add_argument(
        '--output_folder', '-o',
        required=True,
        help='Path to output folder for parquet files'
    )
    
    parser.add_argument(
        '--coins',
        nargs='+',
        help='List of coin symbols to process (e.g., XBT ETH). If not specified, processes all CSV files'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=20,
        help='Block size for grouping rows (default: 20)'
    )
    
    parser.add_argument(
        '--single_file',
        help='Process a single file instead of a folder. Provide the coin symbol (e.g., XBT)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.single_file:
            # Process single file
            input_file = Path(args.input_folder) / f"{args.single_file}_EUR.csv"
            output_file = Path(args.output_folder) / f"{args.single_file}_EUR.parquet"
            
            df_wide = preprocess_crypto_data(
                input_file=input_file,
                output_file=output_file,
                coin=args.single_file,
                block_size=args.block_size
            )
            
            print(f"\nSingle file processing completed!")
            print(f"Output shape: {df_wide.shape}")
            
        else:
            # Process entire folder
            results = preprocess_data_folder(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                coins=args.coins,
                block_size=args.block_size
            )
            
            print(f"\nFolder processing completed!")
            print(f"Processed {len(results)} files:")
            for coin, df in results.items():
                print(f"  {coin}: {df.shape}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
