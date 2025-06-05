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
        args.features, args.target, test_size=args.test_size, n_samples=args.n_samples)
    model = ModelManager.get_model(args.model)
    # Génération d'un nom de fichier reconnaissable basé sur quelques caractéristiques du modèle
    def get_model_id(model, args):
        import re
        # Récupération des principaux hyperparamètres
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            params = model.model.get_params()
        else:
            params = getattr(model, 'hyperparams', {})
        # Champs à inclure dans le nom (modèle, n_estimators, max_depth, learning_rate, target, nb_features)
        model_name = args.model
        n_estimators = params.get('n_estimators', None)
        max_depth = params.get('max_depth', None)
        learning_rate = params.get('learning_rate', None)
        target = getattr(model, 'target_column', None)
        features = getattr(model, 'feature_columns', None)
        nb_features = len(features) if features else None
        # Nettoyage
        def clean(val):
            if val is None:
                return 'NA'
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(val))
        # Construction du nom
        parts = [
            f"model-{clean(model_name)}",
            f"target-{clean(target)}" if target else None,
            f"nfeat-{nb_features}" if nb_features else None,
            f"nest-{n_estimators}" if n_estimators is not None else None,
            f"depth-{max_depth}" if max_depth is not None else None,
            f"lr-{learning_rate}" if learning_rate is not None else None
        ]
        # Filtrer les None
        parts = [p for p in parts if p]
        # Nom de fichier final
        model_id = "_".join(parts)
        return model_id
    # Si args.save n'est pas précisé, on génère un nom reconnaissable et dédié
    if not args.save:
        model_id = get_model_id(model, args)
        save_dir = os.path.join('predictors', args.model, model_id)
        os.makedirs(save_dir, exist_ok=True)
        args.save = os.path.join(save_dir, f"{model_id}.joblib")
    # Use notebook-style reporting for random_forest_mateo, xgboost, adaboost
    if hasattr(model, 'train_and_report'):
        model.train_and_report(X_train, X_test, y_train, y_test, save_path=args.save)
    else:
        model.train(X_train, y_train)
        model.save(args.save)
        print(f"Model trained and saved to {args.save}")
        results = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {results['accuracy']:.4f}")
    # Save hyperparameters and performance
    results = model.evaluate(X_test, y_test)
    import numpy as np
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_convert(v) for v in d]
        else:
            return to_serializable(d)
    # Correction: hyperparams must be enrichis et explicites
    # On récupère les vrais hyperparams du modèle sklearn/xgboost si possible
    def get_explicit_hyperparams(model):
        params = dict(getattr(model, 'hyperparams', {}))
        # Ajout des params du modèle sklearn/xgboost si possible
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            params.update(model.model.get_params())
        # Ajout des features et target si présents
        if hasattr(model, 'feature_columns'):
            params['feature_columns'] = model.feature_columns
        if hasattr(model, 'target_column'):
            params['target_column'] = model.target_column
        # Supprimer les champs à valeur None ou NaN
        import math
        def is_valid(v):
            if v is None:
                return False
            if isinstance(v, float) and (math.isnan(v) or v is None):
                return False
            return True
        params = {k: v for k, v in params.items() if is_valid(v)}
        return params
    hyperparams = get_explicit_hyperparams(model)
    hyperparams = recursive_convert(hyperparams)
    perf_path = args.save + '.perf.json'
    perf_data = {
        'model': args.model,
        'hyperparameters': hyperparams,
        'performance': recursive_convert(results)
    }
    with open(perf_path, 'w') as f:
        json.dump(perf_data, f, indent=2)
    print(f"Performance saved to {perf_path}")


def test(args):
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size, n_samples=args.n_samples)
    model = ModelManager.get_model(args.model)
    model.load(args.load)
    results = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    print(json.dumps(results['report'], indent=2))
    # Save performance
    perf_path = args.load + '.perf.json'
    perf_data = {
        'model': args.model,
        'hyperparameters': getattr(model, 'hyperparams', {}),
        'performance': results
    }
    with open(perf_path, 'w') as f:
        json.dump(perf_data, f, indent=2)
    print(f"Performance saved to {perf_path}")


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
    parser_train.add_argument('--n_samples', type=int, default=None, help='Nombre de data points à utiliser (train+test)')
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--model', required=True)
    parser_test.add_argument('--features', required=True)
    parser_test.add_argument('--target', required=True)
    parser_test.add_argument('--load', required=True)
    parser_test.add_argument('--test_size', type=float, default=0.2)
    parser_test.add_argument('--n_samples', type=int, default=None, help='Nombre de data points à utiliser (train+test)')
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
