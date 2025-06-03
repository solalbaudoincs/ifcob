"""
Preprocessing module for cryptocurrency data.

This module provides functions and utilities for preprocessing cryptocurrency
market data from CSV format to a structured wide format suitable for
feature extraction and machine learning.

Main functions:
- preprocess_crypto_data: Process a single cryptocurrency data file
- preprocess_data_folder: Process all files in a folder
- preprocessing_vectorized: Apply vectorized preprocessing with block grouping
- clean_duplicates: Remove duplicate entries
- pivot_to_wide_format: Convert to wide format for analysis

Usage:
    from preprocessing import preprocess_crypto_data, preprocess_data_folder
    
    # Process single file
    df = preprocess_crypto_data('input.csv', 'output.parquet', coin='XBT')
    
    # Process entire folder
    results = preprocess_data_folder('input_folder/', 'output_folder/')
"""

from .data_preprocessor import (
    preprocessing_vectorized,
    clean_duplicates,
    pivot_to_wide_format,
    preprocess_crypto_data,
    preprocess_data_folder
)

__version__ = "1.0.0"
__author__ = "Cryptocurrency Data Preprocessing Module"

# Make main functions available at package level
__all__ = [
    'preprocessing_vectorized',
    'clean_duplicates', 
    'pivot_to_wide_format',
    'preprocess_crypto_data',
    'preprocess_data_folder'
]
