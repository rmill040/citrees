"""ADD DESCRIPTION"""

import json
import numpy as np
import os
from os.path import exists, join
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

# Custom imports
from regression_experiment import load_data

# Constants
DATA_DIR   = '../data'
N_SPLITS   = 5
ITERATIONS = 10


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
        key = '_es_'     + str(int(params['early_stopping'])) + \
              '_vm_'     + str(int(params['muting'])) + \
              '_alpha_'  + str(params['alpha']) + \
              '_select_' + str(params['selector'])
    return key


def cv_scores(X, y):
    """Evaluate performance of feature subsets using multiple iterations
    of cross-validation.
    
    Parameters
    ----------
    X : 2d array-like
        Features

    y : 1d array-like
        Labels
    
    Returns
    -------
    mses : float
        Average mean squared error score
    
    r2s : float
        Average r squared score
    """
    mses = np.zeros(ITERATIONS)
    r2s  = np.zeros(ITERATIONS)

    # Run CV with different seeds for specified number of iterations
    for i, random_state in enumerate(list(range(1, ITERATIONS+1))):
    
        # SGD hyperparameters
        sgd_params = {
            'penalty'      : None,
            'random_state' : random_state,
            'max_iter'     : int(np.ceil(10**6 / float(X.shape[0]))),
            'tol'          : 1e-3
            }

        # Variables for CV
        tmp_mses = np.zeros(N_SPLITS)
        tmp_r2s  = np.zeros(N_SPLITS)
        cv       = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        
        # Run CV
        fold = 0
        for train_idx, test_idx in cv.split(X):
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Define pipeline
            pipeline = Pipeline([
                ('ss', StandardScaler()),
                ('clf', SGDRegressor(**sgd_params))
            ])

            # Fit pipeline, predict classes, calculate metrics
            pipeline.fit(X_train, y_train)
            y_hat          = pipeline.predict(X_test)
            tmp_mses[fold] = mean_squared_error(y_test, y_hat)
            tmp_r2s[fold]  = np.corrcoef(y_test, y_hat, rowvar=0)[0, 1]**2

            # Handle case with constant prediction, force to 0 correlation
            if np.isnan(tmp_r2s[fold]): tmp_r2s[fold] = 0.0
                
            # Increase counter
            fold += 1
        
        # Update results
        mses[i] = tmp_mses.mean()
        r2s[i]  = tmp_r2s.mean()
    
    # Return averaged results
    return mses.mean(), r2s.mean()
        

def reg_metrics():
    """Calculate regression metrics.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    results : pandas DataFrame
        Regression metrics by method and number of features
    """

    # Load data
    with open(join(DATA_DIR, 'regression.json')) as f:
        data = [json.loads(line) for line in f]

    # Calculate metrics for each data set and method based on feature ranks
    results   = []
    dfs_cache = {}
    for row in data:
        
        # Parse info
        name       = row['data']
        feat_ranks = row['results']['ranks']
        method     = row['results']['method']

        # Skip these for now
        if method in ['lasso', 'ridge']: continue

        # If methods tested with different hyperparameters, make unique key
        if method in ['ct', 'cf']:
            method += create_params_key(method, row['results']['params'])
        
        # Current data
        if name in dfs_cache.keys():
            X, y = dfs_cache[name]
        else:
            X, y            = load_data(name)
            dfs_cache[name] = (X, y)
        
        # Determine number of columns to use
        n, p = X.shape
        if p >= 100:
            cols_to_keep = np.arange(5, 105, 5)
        else:
            cols_to_keep = np.arange(0, p) + 1

        # Run CV for subsets of features
        print("[info] Running CV: DATA = %s, METHOD = %s" % (name, method))
        for i, col in enumerate(cols_to_keep):

            # Subset X
            X_ = X[:, feat_ranks[:col]].reshape(-1, col)
            assert X_.shape[1] == col, "Number of subsetted features is incorrect"

            # Calculate CV scores
            mse, r2 = cv_scores(X_, y)
            
            # Append results
            results.append(
                [name, method, col, mse, r2]
            )

    # Convert to dataframe and save to disk
    columns = ['data', 'method', 'n_feats', 'mse', 'r2']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv(join(DATA_DIR, 'regression_cv_metrics.csv'), index=False)

    return results


def aggregate_metrics(metrics, metric_name):
    """Aggregate specific metrics by methods.
    
    Parameters
    ----------
    metrics : pandas DataFrame
        Input data
    
    metric_name : str
        Name of metric
    
    Returns
    -------
    None
    """
    # Drop lasso and ridge
    mask    = (~metrics['method'].str.contains('lasso')) & \
              (~metrics['method'].str.contains('ridge'))
    metrics = metrics[mask].reset_index(drop=True)

    # Iterate over each data set
    table = {}
    for data in metrics['data'].unique():
            
        summary = {}
        lower_is_better = True if metric_name == 'mse' else False
        
        # Pivot table
        df = metrics[metrics['data'] == data]\
                .pivot_table(values=metric_name, index='method', columns='n_feats')
        
        # Raw metric summary statistics
        summary['mean']   = df.mean(axis=1)
        summary['sd']     = df.std(axis=1)
        if lower_is_better:
            summary['best'] = df.min(axis=1)
        else:
            summary['best'] = df.max(axis=1)
        fn                  = lambda x: np.argmin(x) if lower_is_better else np.argmax(x)
        summary['loc_best'] = df.apply(fn, axis=1)/df.columns.max()

        # Make dataframe and find best metrics per method
        summary = pd.DataFrame(summary)
        ct      = summary[summary.index.str.startswith('ct')]
        cf      = summary[summary.index.str.startswith('cf')]
        r_cf    = summary[summary.index.str.startswith('r_cf')]
        dt      = summary[summary.index.str.startswith('dt')]
        rf      = summary[summary.index.str.startswith('rf')]
        et      = summary[summary.index.str.startswith('et')]
        hy      = summary[summary.index.str.startswith('hybrid')]
        dist    = summary[summary.index.str.startswith('distance')]
        pearson = summary[summary.index.str.startswith('pearson')]

        # Populate table with mean score
        table[data + '_mean'] = {
            'ct'      : ct['mean'].min() if lower_is_better else ct['mean'].max(),
            'cf'      : cf['mean'].min() if lower_is_better else cf['mean'].max(),
            'r_cf'    : r_cf['mean'].min() if lower_is_better else r_cf['mean'].max(),
            'dt'      : dt['mean'][0],
            'et'      : et['mean'][0],
            'rf'      : rf['mean'][0],
            'pearson' : pearson['mean'][0],
            'dist'    : dist['mean'][0],
            'hy'      : hy['mean'][0],
        }

        # Given multiple params tested, find ct and cf hyperparameter 
        # configuration that yielded best mean score
        ct_config   = np.argmin(ct['mean']) if lower_is_better else np.argmax(ct['mean'])
        cf_config   = np.argmin(cf['mean']) if lower_is_better else np.argmax(cf['mean'])
        r_cf_config = np.argmin(r_cf['mean']) if lower_is_better else np.argmax(r_cf['mean'])

        # Populate table with sd of row with mean scores
        table[data + '_sd'] = {
            'ct'      : ct[ct.index==ct_config]['sd'][0],
            'cf'      : cf[cf.index==cf_config]['sd'][0],
            'r_cf'    : r_cf[r_cf.index==r_cf_config]['sd'][0],
            'dt'      : dt['sd'][0],
            'et'      : et['sd'][0],
            'rf'      : rf['sd'][0],
            'pearson' : pearson['sd'][0],
            'dist'    : dist['sd'][0],
            'hy'      : hy['sd'][0],
        }

        # Populate table with best score
        table[data + '_best'] = {
            'ct'      : ct['best'].min() if lower_is_better else ct['best'].max(),
            'cf'      : cf['best'].min() if lower_is_better else cf['best'].max(),
            'r_cf'    : r_cf['best'].min() if lower_is_better else r_cf['best'].max(),
            'dt'      : dt['best'][0],
            'et'      : et['best'][0],
            'rf'      : rf['best'][0],
            'pearson' : pearson['best'][0],
            'dist'    : dist['best'][0],
            'hy'      : hy['best'][0],
        }

        # Given multiple params tested, find ct and cf hyperparameter 
        # configuration that yielded best overall score
        ct_config   = np.argmin(ct['best']) if lower_is_better else np.argmax(ct['best'])
        cf_config   = np.argmin(cf['best']) if lower_is_better else np.argmax(cf['best'])
        r_cf_config = np.argmin(r_cf['best']) if lower_is_better else np.argmax(r_cf['best'])

        # Populate table with % location of best score
        table[data + '_loc_best'] = {
            'ct'      : ct[ct.index==ct_config]['loc_best'][0],
            'cf'      : cf[cf.index==cf_config]['loc_best'][0],
            'r_cf'    : r_cf[r_cf.index==r_cf_config]['loc_best'][0],
            'dt'      : dt['loc_best'][0],
            'et'      : et['loc_best'][0],
            'rf'      : rf['loc_best'][0],
            'pearson' : pearson['loc_best'][0],
            'dist'    : dist['loc_best'][0],
            'hy'      : hy['loc_best'][0],
        }
    
    # Write results to disk
    pd.DataFrame(table).T.to_csv(
            join(DATA_DIR, 'table_' + metric_name + '.csv'),
            index=True
        )


def main():
    """Main function to run regression analysis"""

    start = time.time()

    # Calculate raw metrics
    path = join(DATA_DIR, 'regression_cv_metrics.csv')
    if not exists(path):
        metrics = reg_metrics()
    else:
        metrics = pd.read_csv(path)

    # Combine metrics from R
    try:
        r_metrics = pd.read_csv(join(DATA_DIR, 'r_regression_cv_metrics.csv'))
    except:
        os.system('python run_r_regression_models.py')
        r_metrics = pd.read_csv(join(DATA_DIR, 'r_regression_cv_metrics.csv'))

    # Aggregate metrics and calculate different statistics for tables
    metrics = pd.concat([metrics, r_metrics], axis=0)
    aggregate_metrics(metrics, 'r2')
    aggregate_metrics(metrics, 'mse')

    # Script finished
    minutes = (time.time() - start)/60
    print("\n[info] Script finished in %.2f minutes" % minutes)

if __name__ == "__main__":
    main()