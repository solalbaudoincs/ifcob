#!/usr/bin/env python3
"""
Feature Generation Script

This script provides a command-line interface for generating features from
preprocessed order book data. It supports generating individual features,
specific sets of features, or all features at once.

Usage examples:
    # Generate all features for ETH data version 2
    python generate_features.py --coin ETH --data-version 2

    # Generate specific features
    python generate_features.py --coin XBT --data-version 1 --features spread bid-ask-imbalance-5-levels

    # List available features
    python generate_features.py --coin ETH --data-version 2 --list-features

    # Regenerate a single feature
    python generate_features.py --coin ETH --data-version 2 --features spread --overwrite

    # Generate features for multiple coins
    python generate_features.py --coin ETH XBT --data-version 2
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add the feature_extraction directory to the Python path
current_dir = Path(__file__).parent
feature_extraction_dir = current_dir.parent / "feature_extraction"
sys.path.insert(0, str(feature_extraction_dir))

from feature_generator import (
    FeatureGenerator, 
    load_and_generate_features, 
    save_features,
    preprocess_data,
    recompute_feature
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate features from preprocessed order book data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--coin', '-c',
        nargs='+',
        default=['ETH'],
        help='Coin symbol(s) to process (default: ETH). Can specify multiple coins.'
    )
    
    parser.add_argument(
        '--data-version', '-d',
        type=int,
        default=2,
        help='Data version number (default: 2)'
    )
    
    parser.add_argument(
        '--features', '-f',
        nargs='+',
        help='Specific features to generate (default: all features)'
    )
    
    parser.add_argument(
        '--list-features', '-l',
        action='store_true',
        help='List all available features and exit'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default=str(Path(__file__).parent.parent / 'data' / 'features'),
        help='Output directory for generated features (default: ../data/features)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        default=str(Path(__file__).parent.parent / 'data' / 'preprocessed'),
        help='Input directory for preprocessed data (default: ../data/preprocessed)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--recompute-feature',
        nargs='+',
        help='Recompute (in-place) one or more features for the specified coin(s) and data version, leaving other features untouched.'
    )
    
    return parser.parse_args()


def list_available_features():
    """List all available features."""
    generator = FeatureGenerator()
    
    print("Available features:")
    print("=" * 50)
    
    for feature_name in sorted(generator.list_features()):
        description = generator.get_feature_info(feature_name)
        print(f"{feature_name:25} : {description}")
    
    print(f"\nTotal: {len(generator.list_features())} features available")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(filepath)


def load_preprocessed_data(coin: str, data_version: int, input_dir: str) -> pd.DataFrame:
    """Load preprocessed data for a specific coin and data version."""
    filepath = f"{input_dir}/DATA_{data_version}/{coin}_EUR.parquet"
    
    if not check_file_exists(filepath):
        raise FileNotFoundError(f"Preprocessed data not found: {filepath}")
    
    return pd.read_parquet(filepath)


def generate_features_for_coin(
    coin: str, 
    data_version: int, 
    feature_names: Optional[List[str]], 
    input_dir: str, 
    output_dir: str, 
    overwrite: bool,
    verbose: bool
) -> bool:
    """
    Generate features for a single coin.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if output file already exists
        output_path = f"{output_dir}/DATA_{data_version}/{coin}_EUR.parquet"
        
        if check_file_exists(output_path) and not overwrite:
            if feature_names:
                # For specific features, we need to load existing and update
                if verbose:
                    print(f"Loading existing features for {coin} to update specific features...")
                existing_features = pd.read_parquet(output_path)
            else:
                print(f"Features file already exists for {coin} (DATA_{data_version}). Use --overwrite to regenerate.")
                return True
        
        # Load preprocessed data
        if verbose:
            print(f"Loading preprocessed data for {coin} (DATA_{data_version})...")
        
        df = load_preprocessed_data(coin, data_version, input_dir)
        
        # Preprocess data
        if verbose:
            print(f"Preprocessing data for {coin}...")
        
        df_cleaned = preprocess_data(df)
        
        # Generate features
        if verbose:
            print(f"Generating features for {coin}...")
            if feature_names:
                print(f"  Specific features: {', '.join(feature_names)}")
            else:
                print("  All features")
        
        generator = FeatureGenerator()
        new_features = generator.generate_features(df_cleaned, feature_names)
        
        # If we're updating specific features, merge with existing
        if feature_names and check_file_exists(output_path) and not overwrite:
            # Load existing features
            existing_features = pd.read_parquet(output_path)
            
            # Update specific features
            for feature_name in feature_names:
                if feature_name in new_features.columns:
                    existing_features[feature_name] = new_features[feature_name]
                    if verbose:
                        print(f"  Updated feature: {feature_name}")
            
            features = existing_features
        else:
            features = new_features
        
        # Save features
        save_features(features, coin, data_version, output_dir)
        
        if verbose:
            print(f"Successfully generated {len(features.columns)} features for {coin}")
            print(f"Generated features: {', '.join(features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Error generating features for {coin}: {str(e)}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # List features if requested
    if args.list_features:
        list_available_features()
        return
    
    # Validate feature names if provided
    if args.features:
        generator = FeatureGenerator()
        available_features = generator.list_features()
        invalid_features = [f for f in args.features if f not in available_features]
        
        if invalid_features:
            print(f"Error: Invalid feature names: {', '.join(invalid_features)}")
            print("Use --list-features to see available features")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle recompute-feature option
    if args.recompute_feature:
        for coin in args.coin:
            for feature_name in args.recompute_feature:
                print(f"Recomputing feature '{feature_name}' for {coin} (DATA_{args.data_version})...")
                try:
                    recompute_feature(
                        coin=coin,
                        data_version=args.data_version,
                        feature_name=feature_name,
                        output_dir=args.output_dir
                    )
                    print(f"✓ Successfully recomputed '{feature_name}' for {coin}")
                except Exception as e:
                    print(f"✗ Failed to recompute '{feature_name}' for {coin}: {e}")
        print("Done.")
        sys.exit(0)
    
    # Process each coin
    success_count = 0
    total_count = len(args.coin)
    
    for coin in args.coin:
        print(f"\nProcessing {coin}...")
        
        success = generate_features_for_coin(
            coin=coin,
            data_version=args.data_version,
            feature_names=args.features,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
            print(f"✓ Successfully processed {coin}")
        else:
            print(f"✗ Failed to process {coin}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary: {success_count}/{total_count} coins processed successfully")
    
    if success_count == total_count:
        print("All features generated successfully!")
        sys.exit(0)
    else:
        print("Some features failed to generate. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
