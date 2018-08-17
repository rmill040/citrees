from __future__ import division, print_function

import json
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import time

# Add path to avoid relative imports
PATH = abspath(__file__).split('experiments')[0]
if PATH not in sys.path: sys.path.append(PATH)

from citrees import CITreeRegressor, CIForestRegressor

DATA_DIR  = abspath(__file__).split('scripts')[0]
DATA_DIR  = join(DATA_DIR, 'data')
DATA_SETS = ['coepra2', 'parkinsons', 'coepra3', 'residential', 'skill_craft', 
             'comm_violence', 'community_crime', 'facebook', 'imports-85', 
             'coepra1']


def load_data(name):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    df = pd.read_csv(join(DATA_DIR, name + '.data')).replace('?', 0)

    if name in ['comm_violence', 'community_crime', 'facebook', 'imports-85']:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    elif name in ['coepra1', 'coepra2', 'coepra3']:
        X, y = df.iloc[:, df.columns != 'Prop_001'], df['Prop_001']
    
    elif name in ['parkinsons']:
        X, y = df.loc[:, df.columns != 'total_UPDRS'], df['total_UPDRS']
    
    elif name in ['residential']:
        X, y = df.drop(['V-9', 'V-10'], axis=1), df['V-9']
    
    elif name in ['skill_craft']:
        df   = df.replace('?', 0).astype(float)
        X, y = df.loc[:, df.columns != 'APM'], df['APM']

    else:
        raise ValueError("%s not a valid data set" % name)

    # Convert object dtypes to float
    if 'object' in X.dtypes.values:
        for col in X.select_dtypes(['object']).columns:
            try:
                X[col] = X[col].astype(float)
            except:
                X[col] = LabelEncoder().fit_transform(X[col])

    return X.astype(float).values, y.values


def cross_validate(X, y, model, params, k, cv):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes, scores, fold = len(set(y)), np.zeros(k), 0
    for train_idx, test_idx in cv.split(X):

        # Split into train and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler          = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        # Train model and make predictions
        reg          = model(**params).fit(X_train, y_train)
        y_hat        = reg.predict(X_test) 
        r2           = r2_score(y_test, y_hat)

        # Only update if greater than 0
        if r2 > 0: scores[fold] = r2

        # Next fold
        print("[CV] Fold %d: R2 = %.4f" % (fold+1, scores[fold]))
        fold += 1

    print("[CV] Overall: R2 = %.4f +/- %.4f\n" % (scores.mean(), scores.std()))
    return scores



def main():
    """ADD DESCRIPTION"""

    kf = KFold(n_splits=5, shuffle=True, random_state=1718)

    for name in DATA_SETS:
        print("[NAME] %s" % name)
        X, y = load_data(name)
        cross_validate(X, y, RandomForestRegressor, {}, 5, kf)


if __name__ == '__main__':
    main()
