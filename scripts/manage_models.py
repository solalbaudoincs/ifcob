"""
manage_models.py
================
Script to train, test, tune, and compare models using the ModelManager.

Usage examples:
---------------
Train a model:
    python manage_models.py train --model random_forest_mateo --features <features_path> --target <target_path> [--n_estimators 5] [--max_depth 3] [--learning_rate 0.1]

Test a model:
    python manage_models.py test --model random_forest_mateo --features <features_path> --target <target_path> --load <model_path>

Compare hyperparameters:
    python manage_models.py compare --model random_forest_mateo --features <features_path> --target <target_path>
    # (param_grid is no longer supported; set hyperparameters directly with --n_estimators, --max_depth, --learning_rate, etc.)

Select best hyperparameters:
    python manage_models.py select --model random_forest_mateo

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

def train(args):
    # Use notebook defaults if not specified
    import glob
    # If --select-best is set, try to load best hyperparameters from previous runs
    if hasattr(args, 'select_best') and args.select_best:
        search_dir = os.path.join('predictors', args.model)
        perf_files = glob.glob(os.path.join(search_dir, '**', '*.perf.json'), recursive=True)
        best_acc = -1
        best_params = None
        for pf in perf_files:
            with open(pf, 'r') as f:
                data = json.load(f)
            perf = data.get('performance', {})
            acc = perf.get('accuracy')
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_params = data.get('hyperparameters', {})
        if best_params:
            print(f"Using best hyperparameters from previous runs: {best_params}")
            args.__dict__.update(best_params)
        else:
            print("No previous hyperparameters found. Proceeding with defaults or provided values.")
    if not args.features or not args.target:
        features_path, target_path, model_path = ModelManager.MODELS[args.model].get_default_paths()
        if not args.features:
            args.features = features_path
        if not args.target:
            args.target = target_path
        if not args.save:
            args.save = model_path
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size, n_samples=args.n_samples, model_name=args.model)
    # Collect supported hyperparameters
    model_kwargs = {}
    if getattr(args, 'n_estimators', None) is not None:
        model_kwargs['n_estimators'] = args.n_estimators
    if getattr(args, 'max_depth', None) is not None:
        model_kwargs['max_depth'] = args.max_depth
    if getattr(args, 'learning_rate', None) is not None:
        model_kwargs['learning_rate'] = args.learning_rate
    model = ModelManager.get_model(args.model, **model_kwargs)
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
        if 'estimator' in params.keys():
            # Si c'est un modèle sklearn, on enlève l'estimateur pour éviter la récursion infinie
            params.pop('estimator', None)
        params = {k: v for k, v in params.items() if is_valid(v)}
        return params
    
    hyperparams = get_explicit_hyperparams(model)
    hyperparams = recursive_convert(hyperparams)
    perf_path = args.save + '.perf.json'
    perf_data = {
        'model': str(args.model),
        'hyperparameters': hyperparams,
        'performance': recursive_convert(results)
    }
    with open(perf_path, 'w') as f:
        json.dump(perf_data, f, indent=2)
    print(f"Performance saved to {perf_path}")

def test(args):
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size, n_samples=args.n_samples, model_name=args.model)
    model = ModelManager.get_model(args.model)
    model.load(args.load)
    results = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    print(json.dumps(results['report'], indent=2))
    # Save performance
    perf_path = args.load + 'test.perf.json'
    perf_data = {
        'model': str(args.model),
        'hyperparameters': getattr(model, 'hyperparams', {}),
        'performance': recursive_convert(results)
    }
    print(perf_data)
    with open(perf_path, 'w') as f:
        json.dump(perf_data, f, indent=2)
    print(f"Performance saved to {perf_path}")


def compare(args):
    from sklearn.model_selection import ParameterGrid
    X_train, X_test, y_train, y_test = ModelManager.prepare_data(
        args.features, args.target, test_size=args.test_size, model_name=args.model)
    param_grid = json.loads(args.param_grid)
    best_acc = -1
    best_params = None
    # for params in ParameterGrid(param_grid):
    #     print(f"Testing params: {params}")
    #     model = ModelManager.get_model(args.model, **params)
    #     model.train(X_train, y_train)
    #     results = model.evaluate(X_test, y_test)
    #     acc = results['accuracy']
    #     print(f"Accuracy: {acc:.4f}")
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_params = params
    print(f"Best accuracy: {best_acc:.4f} with params: {best_params}")


def select(args):
    """Select and print the best hyperparameters from previous compare runs."""
    import glob
    import json
    import os
    # Find all .perf.json files in predictors/<model>/
    search_dir = os.path.join('predictors', args.model)
    perf_files = glob.glob(os.path.join(search_dir, '**', '*.perf.json'), recursive=True)
    best_acc = -1
    best_file = None
    best_perf = None
    for pf in perf_files:
        with open(pf, 'r') as f:
            data = json.load(f)
        perf = data.get('performance', {})
        acc = perf.get('accuracy')
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_file = pf
            best_perf = data
    if best_file:
        print(f"Best model: {best_file}")
        print(json.dumps(best_perf, indent=2))
    else:
        print("No performance files found or no accuracy available.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage ML models.\n\n"
        "Commands:\n"
        "  train   Train a model and save it with performance report.\n"
        "  test    Test a model and print/save performance report.\n"
        "  compare Grid search over hyperparameters and report best.\n"
        "  select  Select and display the best hyperparameters from previous runs.\n\n"
        "Examples:\n"
        "  python manage_models.py train --model random_forest_mateo --features <features_path> --target <target_path>\n"
        "  python manage_models.py test --model random_forest_mateo --features <features_path> --target <target_path> --load <model_path>\n"
        "  python manage_models.py compare --model random_forest_mateo --features <features_path> --target <target_path>\n"
        "  python manage_models.py select --model random_forest_mateo\n"
    )
    subparsers = parser.add_subparsers(dest='command')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model', required=True)
    parser_train.add_argument('--features', required=False, default=None)
    parser_train.add_argument('--target', required=False, default=None)
    parser_train.add_argument('--save', required=False, default=None)
    parser_train.add_argument('--test_size', type=float, default=0.5)
    parser_train.add_argument('--n_samples', type=int, default=None, help='Nombre de data points à utiliser (train+test)')
    parser_train.add_argument('--select-best', action='store_true', help='Use best hyperparameters from previous runs if available')
    parser_train.add_argument('--n_estimators', type=int, default=None, help='Number of estimators for the model (if supported)')
    parser_train.add_argument('--max_depth', type=int, default=None, help='Maximum depth for the model (if supported)')
    parser_train.add_argument('--learning_rate', type=float, default=None, help='Learning rate for the model (if supported)')
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--model', required=True)
    parser_test.add_argument('--features', required=True)
    parser_test.add_argument('--target', required=True)
    parser_test.add_argument('--load', required=True)
    parser_test.add_argument('--test_size', type=float, default=0.5)
    parser_test.add_argument('--n_samples', type=int, default=None, help='Nombre de data points à utiliser (train+test)')
    parser_test.set_defaults(func=test)

    # Compare
    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument('--model', required=True)
    parser_compare.add_argument('--features', required=True)
    parser_compare.add_argument('--target', required=True)
    parser_compare.add_argument('--test_size', type=float, default=0.5)
    parser_compare.set_defaults(func=compare)

    # Select
    parser_select = subparsers.add_parser('select')
    parser_select.add_argument('--model', required=True)
    parser_select.set_defaults(func=select)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
