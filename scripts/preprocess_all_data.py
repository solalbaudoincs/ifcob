#!/usr/bin/env python3
"""
Batch preprocessing script for cryptocurrency data.

This script processes all XBT and ETH data files across multiple DATA folders
(DATA_0, DATA_1, DATA_2) and outputs them to the preprocessed data directory.

Usage:
    python scripts/preprocess_all_data.py [options]

The script will:
1. Scan data/raw/DATA_i folders for i=0,1,2
2. Process XBT_EUR.csv and ETH_EUR.csv files in each folder
3. Output preprocessed data to data/preprocessed/DATA_i/
4. Generate summary report of processing results

Author: Cryptocurrency Data Processing Pipeline
Date: June 2025
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocessing import preprocess_crypto_data, preprocess_data_folder


class DataPreprocessor:
    """Batch data preprocessor for cryptocurrency data."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            base_path: Base path to the project. If None, uses script location.
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.raw_data_path = self.base_path / "data" / "raw"
        self.preprocessed_path = self.base_path / "data" / "preprocessed"
        
        # Coins to process
        self.coins = ['XBT', 'ETH']
        
        # Data folders to process
        self.data_folders = ['DATA_0', 'DATA_1', 'DATA_2']
        
        # Processing statistics
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'errors': [],
            'processing_times': {},
            'file_sizes': {}
        }
    
    def check_input_files(self):
        """Check which input files exist and are accessible."""
        print("🔍 Scanning for input files...")
        print("=" * 60)
        
        files_found = []
        
        for data_folder in self.data_folders:
            folder_path = self.raw_data_path / data_folder
            
            if not folder_path.exists():
                print(f"⚠️  Folder {data_folder} does not exist")
                continue
            
            print(f"\n📁 Checking {data_folder}:")
            
            for coin in self.coins:
                file_path = folder_path / f"{coin}_EUR.csv"
                
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    
                    print(f"  ✅ {coin}_EUR.csv ({file_size_mb:.1f} MB)")
                    files_found.append({
                        'data_folder': data_folder,
                        'coin': coin,
                        'file_path': file_path,
                        'size_mb': file_size_mb
                    })
                    self.stats['file_sizes'][f"{data_folder}/{coin}"] = file_size_mb
                else:
                    print(f"  ❌ {coin}_EUR.csv (not found)")
        
        self.stats['total_files_found'] = len(files_found)
        
        print(f"\n📊 Summary: Found {len(files_found)} files to process")
        total_size = sum(f['size_mb'] for f in files_found)
        print(f"📊 Total data size: {total_size:.1f} MB")
        
        return files_found
    
    def process_single_file(self, data_folder: str, coin: str, block_size: int = 20):
        """
        Process a single cryptocurrency data file.
        
        Args:
            data_folder: Data folder name (e.g., 'DATA_0')
            coin: Coin symbol (e.g., 'XBT')
            block_size: Block size for preprocessing
        
        Returns:
            dict: Processing result with success status and metadata
        """
        input_file = self.raw_data_path / data_folder / f"{coin}_EUR.csv"
        output_file = self.preprocessed_path / data_folder / f"{coin}_EUR.parquet"
        
        if not input_file.exists():
            return {
                'success': False,
                'error': f"Input file not found: {input_file}",
                'processing_time': 0,
                'output_shape': None
            }
        
        try:
            start_time = time.time()
            
            # Process the file
            df_wide = preprocess_crypto_data(
                input_file=str(input_file),
                output_file=str(output_file),
                coin=coin,
                block_size=block_size
            )
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'error': None,
                'processing_time': processing_time,
                'output_shape': df_wide.shape
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {data_folder}/{coin}: {str(e)}"
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'output_shape': None
            }
    
    def process_all_data(self, block_size: int = 20, force: bool = False):
        """
        Process all cryptocurrency data files.
        
        Args:
            block_size: Block size for preprocessing
            force: If True, overwrite existing output files
        """
        print("🚀 Starting batch preprocessing...")
        print("=" * 60)
        
        # Check input files
        files_to_process = self.check_input_files()
        
        if not files_to_process:
            print("❌ No files found to process!")
            return
        
        # Confirm processing
        if not force:
            total_size = sum(f['size_mb'] for f in files_to_process)
            print(f"\n⚠️  About to process {len(files_to_process)} files ({total_size:.1f} MB total)")
            response = input("Continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Processing cancelled by user")
                return
        
        print(f"\n🔄 Processing {len(files_to_process)} files with block_size={block_size}...")
        start_time = time.time()
        
        # Process each file
        for i, file_info in enumerate(files_to_process, 1):
            data_folder = file_info['data_folder']
            coin = file_info['coin']
            
            print(f"\n[{i}/{len(files_to_process)}] Processing {data_folder}/{coin}...")
            
            # Check if output already exists
            output_file = self.preprocessed_path / data_folder / f"{coin}_EUR.parquet"
            if output_file.exists() and not force:
                print(f"  ⏭️  Output already exists, skipping (use --force to overwrite)")
                self.stats['files_skipped'] += 1
                continue
            
            # Process the file
            result = self.process_single_file(data_folder, coin, block_size)
            
            if result['success']:
                print(f"  ✅ Success! Output shape: {result['output_shape']}")
                print(f"  ⏱️  Processing time: {result['processing_time']:.1f} seconds")
                self.stats['files_processed'] += 1
                self.stats['processing_times'][f"{data_folder}/{coin}"] = result['processing_time']
            else:
                print(f"  ❌ Failed: {result['error']}")
                self.stats['errors'].append(result['error'])
        
        total_time = time.time() - start_time
        
        # Print final summary
        self.print_summary(total_time)
    
    def print_summary(self, total_time: float):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"📁 Files found: {self.stats['total_files_found']}")
        print(f"✅ Files processed: {self.stats['files_processed']}")
        print(f"⏭️  Files skipped: {self.stats['files_skipped']}")
        print(f"❌ Errors: {len(self.stats['errors'])}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times'].values()) / len(self.stats['processing_times'])
            print(f"⏱️  Average time per file: {avg_time:.1f} seconds")
        
        # Show processing times breakdown
        if self.stats['processing_times']:
            print(f"\n📈 Processing times by file:")
            for file_key, proc_time in self.stats['processing_times'].items():
                file_size = self.stats['file_sizes'].get(file_key, 0)
                rate = file_size / proc_time if proc_time > 0 else 0
                print(f"  {file_key}: {proc_time:.1f}s ({rate:.1f} MB/s)")
        
        # Show errors
        if self.stats['errors']:
            print(f"\n❌ Errors encountered:")
            for error in self.stats['errors']:
                print(f"  • {error}")
        
        print(f"\n📂 Output location: {self.preprocessed_path}")
        print("=" * 60)


def main():
    """Main function with command-line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch preprocess cryptocurrency data for all DATA folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess_all_data.py                    # Process with default settings
  python scripts/preprocess_all_data.py --block-size 30   # Use custom block size
  python scripts/preprocess_all_data.py --force           # Overwrite existing files
  python scripts/preprocess_all_data.py --check-only      # Only check what files exist
        """
    )
    
    parser.add_argument(
        '--block-size',
        type=int,
        default=20,
        help='Block size for preprocessing (default: 20)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check which files exist, do not process'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        help='Base path to the project (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(base_path=args.base_path)
    
    print("🔄 Cryptocurrency Data Batch Preprocessor")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project path: {preprocessor.base_path}")
    print(f"📂 Raw data path: {preprocessor.raw_data_path}")
    print(f"📂 Output path: {preprocessor.preprocessed_path}")
    
    if args.check_only:
        # Only check files, don't process
        files_found = preprocessor.check_input_files()
        print(f"\n✅ Check completed. Found {len(files_found)} files ready for processing.")
    else:
        # Process all data
        preprocessor.process_all_data(
            block_size=args.block_size,
            force=args.force
        )
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
