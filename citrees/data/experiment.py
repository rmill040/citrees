from __future__ import division, print_function

import json
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import sys
import time

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from citrees import CITreeClassifier, CIForestClassifier

DATA_SETS = ['wine', 'orlraws10P', 'glass', 'warpPIE10P', 'warpAR10P', 'pixraw10P', 
             'ALLAML', 'CLL_SUB_111', 'ORL', 'TOX_171', 'Yale', 'musk',
             'vowel-context', 'gamma', 'isolet', 'letter', 'madelon',
             'page-blocks', 'pendigits', 'spam']


def load_data(name):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    if name in ['ALLAML', 'CLL_SUB_111', 'ORL', 'orlraws10P',
                'pixraw10P', 'TOX_171', 'warpAR10P', 'warpPIE10P',
                'Yale']:

        # Load data
        df   = loadmat(name + '.mat')
        X, y = df.values()[1], df.values()[0]

    elif name in ['glass', 'wine', 'vowel-context', 'gamma', 'isolet', 'letter',
                  'madelon', 'musk', 'page-blocks', 'pendigits', 'spam']:
        
        # Load data
        df = pd.read_table(name + '.data', delimiter=',', header=None)

        if name == 'vowel-context':
            X, y = df.iloc[:, 3:-1].values, df.iloc[:, -1].values
        
        elif name == 'wine':
            X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
        
        elif name == 'gamma':
            mapper = {'g': 0, 'h': 1}
            X, y   = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
            y      = np.array([mapper[value] for value in y])
        
        elif name == 'isolet':
            X, y = df.iloc[1:, 1:-1].values, df.iloc[1:, -1].values
        
        elif name == 'letter':
            from string import ascii_uppercase
            mapper = {ch: i for i, ch in enumerate(ascii_uppercase)} 
            X, y   = df.iloc[:, 1:].values, df.iloc[:, 0].values
        
        elif name == 'musk':
            X, y = df.iloc[1:, 2:-1].values, df.iloc[1:, -1].values
        
        else:
            X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    else:
        raise ValueError("%s not a recognized data set name" % name)

    # Fix labels for y so that minimum is 0
    classes     = np.unique(y)
    new_classes = dict(zip(classes, np.arange(len(classes), dtype=int))) 
    return X, np.array([new_classes[_] for _ in y.ravel()])


def binarize(y_true, y_hat):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    lb = LabelBinarizer().fit(y_true)
    return lb.transform(y_true), lb.transform(y_hat)


def cross_validate(X, y, model, params, k, cv, cols=None):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Subset columns in specified
    if cols: X = X[:, cols]

    n_classes, scores, fold = len(set(y)), np.zeros(k), 0
    for train_idx, test_idx in cv.split(X, y):

        # Split into train and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model and make predictions
        clf = model(**params).fit(X_train, y_train)

        if n_classes == 2:
            y_probs = clf.predict_proba(X_test)
            y_hat   = clf.predict(X_test)
            if min(y_test) != 0:
                y_test -= min(y_test)
                y_hat  -= min(y_test)

            # Calculate metric
            try:
                scores[fold] = roc_auc_score(y_test, y_probs[:, -1])
            except:
                print("[ERROR] Exception raised computing AUC score")
                pass
        
        else:
            # Binarize labels for auc score
            y_hat         = clf.predict(X_test)
            y_test, y_hat = binarize(y_test, y_hat)

            # Calculate metric
            try:
                scores[fold] = roc_auc_score(y_test, y_hat)
            except:
                print("[ERROR] Exception raised computing AUC score")
                pass

        # Next fold
        print("[CV] Fold %d: AUC = %.4f" % (fold+1, scores[fold]))
        fold += 1

    print("[CV] Overall: AUC = %.4f +/- %.4f\n" % (scores.mean(), scores.std()))
    return scores


def metrics_other(model, params, model_name):
    """Calculate metrics for other tree models
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Define cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1718)

    # Iterate over each data set
    results, start = [], time.time()
    for name in DATA_SETS:

        # Load data
        X, y = load_data(name)
        print("[DATA] Name: %s" % name)
        print("[DATA] Shape: %s" % (X.shape,))
        print("[DATA] Labels: %s\n" % np.unique(y))

        n, p = X.shape

        try:
            scores = cross_validate(X, 
                                    y, 
                                    model=model, 
                                    params=params, 
                                    k=5,
                                    cv=skf)
            
            # Summary metrics, update results, and continue
            mean_score, std_score, min_score, max_score = \
                scores.mean(), scores.std(), scores.min(), scores.max()
            
            iteration               = params.copy()
            iteration['name']       = name
            iteration['mean_score'] = mean_score
            iteration['std_score']  = std_score
            iteration['min_score']  = min_score
            iteration['max_score']  = max_score
            results.append(iteration)
        except Exception as e:
            print(e)
            iteration               = params.copy()
            iteration['name']       = name
            iteration['mean_score'] = 0.0
            iteration['std_score']  = 0.0
            iteration['min_score']  = 0.0
            iteration['max_score']  = 0.0
            continue

    # To pandas dataframe and write to disk
    results = pd.DataFrame(results)
    results.to_csv("metrics_%s.csv" % model_name, index=False)

    overall_time = (time.time() - start)/60.
    print("\nScript finished in %.2f minutes" % overall_time)


def metrics_citrees():
    """Calculate metrics for conditional inference tree models"""

    # Create hyperparameter grid
    grid = {
        'alpha': [.01, .05, .25, .50, .75, 1.0],
        'selector': ['pearson', 'distance', 'hybrid'],
        'n_estimators': [200],
        'early_stopping': [True, False],
        'bootstrap': [True],
        'bayes': [True, False],
        'class_weight': ['balanced'],
        'n_jobs': [-1],
        'random_state': [1718]
    }

    grid = list(ParameterGrid(grid))
    print("[CV] Testing %d hyperparameter combinations\n" % len(grid))

    # Define cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1718)

    # Iterate over each data set
    results, start = [], time.time()
    for name in DATA_SETS:

        # Load data
        X, y = load_data(name)
        print("[DATA] Name: %s" % name)
        print("[DATA] Shape: %s" % (X.shape,))
        print("[DATA] Labels: %s\n" % np.unique(y))

        n, p = X.shape

        # Test each hyperparameter grid using cross-validation
        for params in grid:

            print("[CV] Hyperparameters:\n%s" % params)
            try:
                scores = cross_validate(X, 
                                        y, 
                                        model=CIForestClassifier, 
                                        params=params, 
                                        k=5,
                                        cv=skf)
                
                # Summary metrics, update results, and continue
                mean_score, std_score, min_score, max_score = \
                    scores.mean(), scores.std(), scores.min(), scores.max()
                
                iteration               = params.copy()
                iteration['name']       = name
                iteration['mean_score'] = mean_score
                iteration['std_score']  = std_score
                iteration['min_score']  = min_score
                iteration['max_score']  = max_score
                results.append(iteration)
            except Exception as e:
                print(e)
                iteration               = params.copy()
                iteration['name']       = name
                iteration['mean_score'] = 0.0
                iteration['std_score']  = 0.0
                iteration['min_score']  = 0.0
                iteration['max_score']  = 0.0
                continue

    # To pandas dataframe and write to disk
    results = pd.DataFrame(results)
    results.to_csv("forest_experiment_results.csv", index=False)

    overall_time = (time.time() - start)/60.
    print("\nScript finished in %.2f minutes" % overall_time)


def calculate_fi():
    """ADD DESCRIPTION"""

    # Create hyperparameter grid
    grid = {
        'alpha': [.01, .05, .50, .75, 1.0],
        'selector': ['distance', 'pearson', 'hybrid'],
        'early_stopping': [True, False],
        'n_estimators': [200],
        'bootstrap': [True],
        'bayes': [True, False],
        'n_jobs': [-1],
        'verbose': [1],
        'random_state': [1718]
    }
    
    grid = list(ParameterGrid(grid))
    print("[FI] Testing %d hyperparameter combinations\n" % len(grid))

    # Iterate over each data set
    results, start = [], time.time()
    for name in DATA_SETS[::-1]:

        # Load data
        X, y = load_data(name)
        print("[DATA] Name: %s" % name)
        print("[DATA] Shape: %s" % (X.shape,))
        print("[DATA] Labels: %s\n" % np.unique(y))

        n, p = X.shape

        # Test each hyperparameter grid using cross-validation
        for params in grid:

            print("[FI] Hyperparameters:\n%s" % params)
            try:
                # Train model to calculate feature importances
                clf = CIForestClassifier(**params).fit(X, y)

                # Save results
                iteration         = params.copy()
                iteration['name'] = name
                iteration['fi']   = clf.feature_importances_.tolist()
                results.append(iteration)
            except Exception as e:
                print("Error calculating fi for %s because %s" % (name, str(e)))
                iteration         = params.copy()
                iteration['name'] = name
                iteration['fi']   = np.zeros(X.shape[1]).tolist()
                continue

    # To json and write to disk
    with open('forest_experiment_fi_results.json', 'w') as f:
        json.dump(results, f)
    
    overall_time = (time.time() - start)/60.
    print("\nScript finished in %.2f minutes" % overall_time)



if __name__ == "__main__":
    #calculate_fi()
    # examine_hp()
    # metrics_other(model=RandomForestClassifier, 
    #               params={'n_estimators': 200, 'n_jobs': -1,
    #                       'class_weight': 'balanced', 'random_state': 1718},
    #               model_name='rf')
    # metrics_other(model=ExtraTreesClassifier, 
    #               params={'n_estimators': 200, 'n_jobs': -1,
    #                       'class_weight': 'balanced', 'random_state': 1718},
    #               model_name='et')
    metrics_citrees()