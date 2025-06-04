#!/usr/bin/env python3
"""
Feature Template Generator

This script helps developers create new feature classes quickly by generating
template code for new features.

Usage:
    python create_feature.py --name MyNewFeature --description "Description of the feature"
"""

import argparse
import os
from pathlib import Path


FEATURE_TEMPLATE = '''
class {class_name}(BaseFeature):
    """Generate {feature_name} feature."""
    
    def __init__(self, {init_params}):
        super().__init__(
            name="{feature_name_snake}",
            description="{description}"
        )
        {init_assignments}
    
    def generate(self, df_cleaned: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the {feature_name} feature.
        
        Args:
            df_cleaned: Preprocessed DataFrame with order book data
            **kwargs: Additional keyword arguments
        
        Returns:
            pd.Series: The generated feature values
        """
        # TODO: Implement feature generation logic here
        # Example:
        # mid_price = (df_cleaned["level-1-bid-price"] + df_cleaned["level-1-ask-price"]) / 2
        # return mid_price.rolling(window=self.window).std()
        
        raise NotImplementedError("Feature generation logic not implemented yet")
'''


def snake_case(text):
    """Convert text to snake_case."""
    import re
    
    # Handle camelCase and PascalCase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', text)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1)
    
    # Replace spaces and other separators with underscores
    s3 = re.sub(r'[\\s-]+', '_', s2)
    
    return s3.lower()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate template code for new features"
    )
    
    parser.add_argument(
        '--name', '-n',
        required=True,
        help='Name of the new feature class (e.g., MyNewFeature)'
    )
    
    parser.add_argument(
        '--description', '-d',
        default="Custom feature",
        help='Description of the feature'
    )
    
    parser.add_argument(
        '--parameters', '-p',
        nargs='+',
        default=['window: int = 20'],
        help='Initialization parameters (e.g., "window: int = 20" "side: str")'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: print to stdout)'
    )
    
    return parser.parse_args()


def generate_feature_code(class_name, description, parameters):
    """Generate the feature class code."""
    
    # Parse parameters
    init_params = ', '.join(parameters)
    
    # Generate assignments
    init_assignments = []
    for param in parameters:
        param_name = param.split(':')[0].strip()
        init_assignments.append(f"        self.{param_name} = {param_name}")
    
    init_assignments_str = '\\n'.join(init_assignments)
    
    # Generate feature name in snake_case
    feature_name_snake = snake_case(class_name.replace('Feature', ''))
    
    # Generate the code
    code = FEATURE_TEMPLATE.format(
        class_name=class_name,
        feature_name=class_name.replace('Feature', ''),
        feature_name_snake=feature_name_snake,
        description=description,
        init_params=init_params,
        init_assignments=init_assignments_str
    )
    
    return code


def main():
    """Main function."""
    args = parse_arguments()
    
    # Ensure class name ends with 'Feature'
    class_name = args.name
    if not class_name.endswith('Feature'):
        class_name += 'Feature'
    
    # Generate the code
    code = generate_feature_code(class_name, args.description, args.parameters)
    
    # Output the code
    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Feature template written to {args.output}")
    else:
        print("# Add this code to your feature_generator.py file:")
        print("# " + "="*60)
        print(code)
        print("# " + "="*60)
        print()
        print("# Don't forget to register the feature in _register_default_features():")
        print(f"# self.register_feature({class_name}())")


if __name__ == "__main__":
    main()
