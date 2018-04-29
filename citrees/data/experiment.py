from __future__ import division, print_function

import numpy as np
from os.path import abspath, dirname
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import sys
import time

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from citrees import CITreeClassifier, CIForestClassifier


DATA_SETS = ['ALLAML', 'CLL_SUB_111', 'ORL', 'orlraws10P',
             'pixraw10P', 'TOX_171', 'warpAR10P', 'warpPIE10P',
             'Yale', 'glass', 'wine', 'vowel-context']

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
        df   = loadmat(name + '.mat')
        X, y = df.values()[1], df.values()[0]

    elif name in ['glass', 'wine', 'vowel-context']:
        df = pd.read_table(name + '.data', delimiter=',', header=None)
        if name == 'vowel-context':
            X, y = df.iloc[:, 3:-1].values, df.iloc[:, -1].values
        elif name == 'wine':
            X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
        else:
            X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values

    else:
        raise ValueError("%s not a recognized data set name" % name)

    # Fix labels for y so that minimum is 0
    classes     = np.unique(y)
    new_classes = dict(zip(classes, np.arange(len(classes), dtype=int))) 
    return X, np.array([new_classes[_] for _ in y.ravel()])


def calculate_fi_ranks():
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    pass


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


def run():
    """ADD DESCRIPTION"""

    # Create hyperparameter grid
    grid = {
        'alpha': [.01, .05, .25, .50, 1.0],
        'n_permutations': [50, 100, 250, 500],
        'selector': ['pearson', 'distance', 'hybrid'],
        'early_stopping': [True, False],
    }
    grid = list(ParameterGrid(grid))
    print("[CV] Testing %d hyperparameter combinations\n" % len(grid))

    # Define cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1718)

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

            # Skip computationally intense conditions and infeasible conditons
            if p > 250 and \
               params['early_stopping'] == False and \
               params['selector'] in ['hybrid', 'distance']: 
               continue

            if params['alpha'] == .01 and params['n_permutations'] == 50:
                continue

            print("[CV] Hyperparameters:\n%s" % params)
            try:
                scores = cross_validate(X, 
                                        y, 
                                        model=CITreeClassifier, 
                                        params=params, 
                                        k=5,
                                        cv=skf)
                
                # Summary metrics, update results, and continue
                mean_score, std_score, min_score, max_score = \
                    scores.mean(), scores.std(), scores.min(), scores.max()
                
                iteration = params.copy()
                iteration['name']       = name
                iteration['mean_score'] = mean_score
                iteration['std_score']  = std_score
                iteration['min_score']  = min_score
                iteration['max_score']  = max_score
                results.append(iteration)
            except:
                iteration = params.copy()
                iteration['name']       = name
                iteration['mean_score'] = 0.0
                iteration['std_score']  = 0.0
                iteration['min_score']  = 0.0
                iteration['max_score']  = 0.0
                continue

    # To pandas dataframe and write to disk
    results = pd.DataFrame(results)
    results.to_csv("experiment_results.csv", index=False)

    overall_time = (time.time() - start)/60.
    print("\nScript finished in %.2f minutes" % overall_time)

if __name__ == "__main__":
    run()