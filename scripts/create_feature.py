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
import shutil


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
        init_assignments.append(f"self.{param_name} = {param_name}")
    
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


def create_feature_file(class_name, code):
    """Create a new feature file in feature_extraction/features."""
    features_dir = Path(__file__).parent.parent / 'feature_extraction' / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    file_name = snake_case(class_name.replace('Feature', '')) + '.py'
    file_path = features_dir / file_name
    if file_path.exists():
        print(f"File {file_path} already exists. Aborting to avoid overwrite.")
        return None
    with open(file_path, 'w') as f:
        f.write(f"from ..base import BaseFeature\nimport pandas as pd\n\n{code}")
    print(f"Feature file created: {file_path}")
    return file_name, class_name


def add_import_to_init(feature_file, class_name):
    """Add import to __init__.py in features directory, properly formatted."""
    init_path = Path(__file__).parent.parent / 'feature_extraction' / 'features' / '__init__.py'
    import_line = f"from .{feature_file[:-3]} import {class_name}\n"
    # Read all lines, remove any duplicate import for this class, and ensure newline at end
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Remove any previous import for this class
        lines = [l for l in lines if f"import {class_name}" not in l]
        # Ensure file ends with a newline
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        # Add the new import at the end
        lines.append(import_line)
        with open(init_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    else:
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(import_line)
    print(f"Added import to {init_path}")


def add_import_to_feature_generator(class_name):
    """Add import to feature_generator.py if not present."""
    fg_path = Path(__file__).parent.parent / 'feature_extraction' / 'feature_generator.py'
    with open(fg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    import_line = f"        {class_name},\n"
    # Insert into the try import block if not present
    for i, line in enumerate(lines):
        if 'from feature_extraction.features import (' in line:
            # Find the next closing parenthesis
            for j in range(i+1, len(lines)):
                if ')' in lines[j]:
                    if import_line not in lines[i+1:j]:
                        lines.insert(j, import_line)
                    break
            break
    with open(fg_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Added import to {fg_path}")


def remove_feature_code(class_name):
    """Remove feature file, import from __init__.py, and import from feature_generator.py."""
    features_dir = Path(__file__).parent.parent / 'feature_extraction' / 'features'
    file_name = snake_case(class_name.replace('Feature', '')) + '.py'
    file_path = features_dir / file_name
    # Remove feature file
    if file_path.exists():
        file_path.unlink()
        print(f"Removed feature file: {file_path}")
    # Remove import from __init__.py
    init_path = features_dir / '__init__.py'
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [l for l in lines if f"import {class_name}" not in l]
        with open(init_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Removed import from {init_path}")
    # Remove import from feature_generator.py
    fg_path = Path(__file__).parent.parent / 'feature_extraction' / 'feature_generator.py'
    if fg_path.exists():
        with open(fg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [l for l in lines if f"{class_name}," not in l]
        with open(fg_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Removed import from {fg_path}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Ensure class name ends with 'Feature'
    class_name = args.name
    if not class_name.endswith('Feature'):
        class_name += 'Feature'
    
    # Generate the code
    code = generate_feature_code(class_name, args.description, args.parameters)
    
    # Create feature file
    result = create_feature_file(class_name, code)
    if result:
        feature_file, class_name = result
        add_import_to_init(feature_file, class_name)
        add_import_to_feature_generator(class_name)
    # Output code as before
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
    import sys
    if '--remove' in sys.argv:
        parser = argparse.ArgumentParser(description="Remove a feature class and its imports")
        parser.add_argument('--remove', required=True, help='Name of the feature class to remove (e.g., MyFeature)')
        args = parser.parse_args()
        remove_feature_code(args.remove)
    else:
        main()
