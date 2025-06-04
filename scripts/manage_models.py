"""
manage_models.py
================
Script to train, test, tune, and compare models using the ModelManager.

Usage examples:
---------------
Train a model:
    python manage_models.py train --model random_forest_mateo --features <features_path> --target <target_path>

Test a model:
    python manage_models.py test --model random_forest_mateo --features <features_path> --target <target_path> --load <model_path>

Compare hyperparameters:
    python manage_models.py compare --model random_forest_mateo --features <features_path> --target <target_path> --param_grid '{"n_estimators": [50, 100], "max_depth": [3, 5]}'

Author: Your Name
Date: 2025-06-04
"""

import sys
import os
# Add the project root and script folder to sys.path for module resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from prediction_model.model_manager import ModelManager


def train(args):
    # Use notebook defaults if not specified
    if not args.features or not args.target:
        from prediction_model.model_manager import RandomForestMateoModel
        features_path, target_path, model_path = RandomForestMateoModel.get_default_paths()
        if not args.features:
            args.features = features_path
        if not args.target:
            args.target = target_path
        if not args.save:
            args.save = model_path
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size)
    model = ModelManager.get_model(args.model)
    # Use notebook-style reporting for random_forest_mateo
    if args.model == 'random_forest_mateo':
        model.train_and_report(X_train, X_test, y_train, y_test, save_path=args.save)
    else:
        model.train(X_train, y_train)
        model.save(args.save)
        print(f"Model trained and saved to {args.save}")
        results = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {results['accuracy']:.4f}")


def test(args):
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size)
    model = ModelManager.get_model(args.model)
    model.load(args.load)
    results = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    print(json.dumps(results['report'], indent=2))


def compare(args):
    from sklearn.model_selection import ParameterGrid
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size)
    param_grid = json.loads(args.param_grid)
    best_acc = -1
    best_params = None
    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        model = ModelManager.get_model(args.model, **params)
        model.train(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        acc = results['accuracy']
        print(f"Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = params
    print(f"Best accuracy: {best_acc:.4f} with params: {best_params}")


def main():
    parser = argparse.ArgumentParser(description="Manage ML models.")
    subparsers = parser.add_subparsers(dest='command')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model', required=True)
    parser_train.add_argument('--features', required=False, default=None)
    parser_train.add_argument('--target', required=False, default=None)
    parser_train.add_argument('--save', required=False, default=None)
    parser_train.add_argument('--test_size', type=float, default=0.2)
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--model', required=True)
    parser_test.add_argument('--features', required=True)
    parser_test.add_argument('--target', required=True)
    parser_test.add_argument('--load', required=True)
    parser_test.add_argument('--test_size', type=float, default=0.2)
    parser_test.set_defaults(func=test)

    # Compare
    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument('--model', required=True)
    parser_compare.add_argument('--features', required=True)
    parser_compare.add_argument('--target', required=True)
    parser_compare.add_argument('--param_grid', required=True, help='JSON string, e.g. {"n_estimators": [50, 100], "max_depth": [3, 5]}')
    parser_compare.add_argument('--test_size', type=float, default=0.2)
    parser_compare.set_defaults(func=compare)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
