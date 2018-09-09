from __future__ import division, print_function

import json
import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath, exists
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
import time

# Custom imports
from regression_experiment import load_data

# Add path to avoid relative imports
PATH = abspath(__file__).split('experiments')[0]
if PATH not in sys.path: sys.path.append(PATH)
from scorers import pcor

# Constants
RANDOM_STATE = 1718


def create_params_key(method, params):
    """Creates unique key based on hyperparameters.

    Parameters
    ----------
    method : str
        Name of method

    params : dict
        Dictionary of parameters

    Returns
    -------
    key : str
        Unique key
    """
    if method in ['lasso', 'ridge']:
        key = '_alpha_' + str(params['alpha'])
    else:
        key = '_es_' + str(int(params['early_stopping'])) + \
              '_vm_' + str(int(params['muting'])) + \
              '_alpha_' + str(params['alpha']) + \
              '_select_' + str(params['selector'])
    return key


def full_cv(X, y, model, k, cv, ranks):
    """Runs cross-validation for different number of features and evaluates
    performance using MSE and R^2.

    Parameters
    ----------
    X : 2d array-like
        Features

    y : 1d array-like
        Labels

    model : object with fit() and predict() methods
        Regression model

    k : int
        Number of folds for cross-validation

    cv : object that returns indices for splits
        Cross-validation object to generate indices

    ranks : 1d array-like
        Sorted array containing indices of most important features in X

    Returns
    -------
    ADD HERE

    ADD HERE
    """
    # Determine number of columns to use
    if X.shape[1] >= 100:
        cols_to_keep = np.arange(5, 105, 5)
    else:
        cols_to_keep = np.arange(0, X.shape[1])+1

    # CV scores
    mse, r2 = np.zeros(len(cols_to_keep)), np.zeros(len(cols_to_keep))
    for i, col in enumerate(cols_to_keep):

        # Subset X
        X_ = X[:, ranks[:col]].reshape(-1, col)
        assert X_.shape[1] == col, "Number of subsetted features is incorrect"

        print("[CV] Running CV with top %d features" % X_.shape[1])
        scores_mse, scores_r2 = single_cv(X_, y, model, k, cv)

        print("\tMSE = %.4f" % scores_mse.mean())
        print("\tR^2 = %.4f" % scores_r2.mean())

        # Append results
        mse[i] = scores_mse.mean()
        r2[i]  = scores_r2.mean()

    return mse, r2, cols_to_keep


def single_cv(X, y, model, k, cv, verbose=False):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    scores_mse, scores_r2, fold = np.zeros(k), np.zeros(k), 0
    for train_idx, test_idx in cv.split(X):

        # Split into train and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize data based on training
        scaler          = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        # Train model and make predictions
        clf   = clone(model).fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        # Calculate MSE
        try:
            scores_mse[fold] = np.mean((y_test-y_hat)*(y_test-y_hat))
        except Exception as e:
            print("[ERROR] Exception raised computing MSE because %s" % e)
            pass

        # Calculate R^2
        try:
            scores_r2[fold] = pcor(y_test, y_hat)**2
        except Exception as e:
            print("[ERROR] Exception raised computing R^2 because %s" % e)
            pass

        # Next fold
        if verbose:
            print("[CV] Fold %d" % (fold+1))
            print("\tMSE = %.4f" % scores_mse[fold])
            print("\tR^2 = %.4f" % scores_r2[fold])
        fold += 1

    # Overall metrics
    if verbose:
        print("[CV] Overall")
        print("\tMSE = %.4f +/- %.4f" % (scores_mse.mean(), scores_mse.std()))
        print("\tR^2 = %.4f +/- %.4f" % (scores_r2.mean(), scores_r2.std()))

    return scores_mse, scores_r2


def split(string):
    string = string.replace('\n', '').replace('\r', '').replace('\\', '')[1:]
    string = [s.replace('r', '').strip().replace('(','').replace(')', '').replace('[', '').replace(']', '') for s in string.split('array') if s]
    mse = map(float, [s.strip() for s in string[0].strip().split(',') if s])
    r2 = map(float, [s.strip() for s in string[1].strip().split(',') if s])
    try:
        cols = map(int, [s.strip() for s in string[2].strip().split(',') if s])
    except:
        string[2] = string[2].replace('dtype=int64', '')
        cols      = map(int, [s.strip() for s in string[2].strip().split(',') if s])
    return mse, r2, cols


def calculate_metrics():
    """Runs SVM on baseline data sets to compare feature importance results
    across methods

    Parameters
    ----------
    None

    Returns
    -------
    df : pandas DataFrame
        Dataframe with metrics on baseline datasets
    """

    # Cross-validation and model objects
    k     = 5
    cv    = KFold(n_splits=k, random_state=RANDOM_STATE)
    model = SVR()

    # Prepare data structures to hold cv data
    cv_results, start = [], time.time()

    # Load feature importance results
    results = pd.DataFrame(
        [json.loads(line) for line in open('regression.json', 'r')]
        )

    # Loop over each data set
    for name in results['data'].unique():

        # Load data and split into features and labels
        X, y = load_data(name)
        print("\n\n-- Data set %s with %d samples and %d features --" % \
              (name, X.shape[0], X.shape[1]))

        # Grab results for current data set in iteration
        df = results[results['data'] == name]['results']

        # Iterate over individual methods
        for method in df.apply(lambda x: x['method']).unique():

            sub = df[df.apply(lambda x: x['method'])==method]

            # Iterate over each set of parameters for current method
            if sub.shape[0] > 1:
                all_params = sub.apply(lambda x: x['params'])
                all_ranks  = sub.apply(lambda x: x['ranks']).values
                for idx, params in enumerate(all_params):
                    ranks = all_ranks[idx]
                    key   = create_params_key(method, params)
                    print("\n* Method = %s *" % (method+key))
                    mse, r2, cols = full_cv(X, y, model, k, cv, ranks)
                    for k in range(mse.shape[0]):
                        cv_results.append([name, method+key, mse[k], r2[k], cols[k]])
            else:
                ranks = sub.apply(lambda x: x['ranks']).values[0]
                print("\n* Method = %s *" % method)
                mse, r2, cols = full_cv(X, y, model, k, cv, ranks)
                for k in range(mse.shape[0]):
                    cv_results.append([name, method, mse[k], r2[k], cols[k]])

    # Convert to dataframe and save to disk
    df = pd.DataFrame(cv_results, columns=['Data', 'Method', 'MSE', 'R2', 'N_Feats'])
    df.to_csv('regression_metrics.csv', index=False)

    return df


def main():
    """Calculates metrics on data sets and analyzes results"""

    # Load or calculate metrics
    if exists("regression_metrics.csv"):
        df = pd.read_csv("regression_metrics.csv")
    else:
        df = calculate_metrics()

    # Calculate summary statistics
    for name in df['Data'].unique():
        sub  = df[df['Data'] == name]
        for method in sub['Method'].unique():
            if 'rf' in method or 'cf' in method:
                pattern = '-o' if 'cf' in method else '--'
                lw = 1.0 if 'ct' in method else 5.0
                plt.plot(sub[sub['Method']==method]['Feats'],
                         sub[sub['Method']==method]['R2'], pattern,
                         linewidth=lw,
                         label=method)
        plt.title("Data = %s" % name)
        # plt.legend()
        plt.show()

        # tmp  = sub[sub['Method'].apply(lambda x: 'cf' in x or 'ct' in x)]
        # tmp['Method'] = tmp['Method'].str.split('_')
        # tmp['selector'] = tmp['Method'].apply(lambda x: x[-1]). \
        #                     map({'pearson': 0, 'distance': 1, 'hybrid': 2})
        # tmp['model']    = tmp['Method'].apply(lambda x: x[0]). \
        #                     map({'ct': 0, 'cf': 1})
        # tmp['es']       = tmp['Method'].apply(lambda x: int(x[2]))
        # tmp['vm']       = tmp['Method'].apply(lambda x: int(x[4]))
        # tmp['alpha']    = tmp['Method'].apply(lambda x: float(x[6])). \
        #                     map({.01: 0, .05: 1, .95: 2})
        # from statsmodels.stats.anova import anova_lm
        # from statsmodels.formula.api import ols
        # from statsmodels.graphics.api import interaction_plot
        # lm = ols("R2 ~ C(alpha) + C(model) + C(es) + C(vm) + Feats", tmp).fit()
        # print(ols("R2 ~ C(selector) + C(alpha) + C(selector)*C(alpha)", tmp).fit().summary())
        # import pdb; pdb.set_trace()
        #
        # interaction_plot(tmp['selector'].values, tmp['vm'].values, tmp['R2'].fillna(0).values)
        # print(ols("R2 ~ C(selector) + C(alpha) + C(model) + C(es) + C(vm) + Feats", tmp).fit().summary())

        # interaction_plot(tmp['selector'], tmp['vm'], tmp['R2'])
        # plt.show()

if __name__ == "__main__":
    main()
