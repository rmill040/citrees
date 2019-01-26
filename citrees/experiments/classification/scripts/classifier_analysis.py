import json
import numpy as np
import os
from os.path import exists, join
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
import time

# Custom imports
from classifier_experiment import load_data

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
    acc : float
        Average accuracy score
    
    aucs : float
        Average AUC score
    
    f1 : float
        Average f1 score
    """
    accs = np.zeros(ITERATIONS)
    aucs = np.zeros(ITERATIONS)
    f1s  = np.zeros(ITERATIONS)

    # Number of classes
    classes   = list(set(y))
    n_classes = len(classes)

    # Run CV with different seeds for specified number of iterations
    for i, random_state in enumerate(list(range(1, ITERATIONS+1))):
    
        # SGD hyperparameters
        sgd_params = {
            'penalty'      : None,
            'random_state' : random_state,
            'class_weight' : 'balanced',
            'max_iter'     : int(np.ceil(10**6 / float(X.shape[0]))),
            'tol'          : 1e-3
            }

        # Variables for CV
        tmp_accs = np.zeros(N_SPLITS)
        tmp_aucs = np.zeros(N_SPLITS)
        tmp_f1s  = np.zeros(N_SPLITS)
        cv       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, 
                                   random_state=random_state)
        
        # Run CV
        fold = 0
        for train_idx, test_idx in cv.split(X, y):
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Define pipeline
            pipeline = Pipeline([
                ('ss', StandardScaler()),
                ('clf', SGDClassifier(**sgd_params))
            ])

            # Fit pipeline, predict classes, calculate accuracy, calculate f1 score
            pipeline.fit(X_train, y_train)
            y_hat          = pipeline.predict(X_test)
            tmp_accs[fold] = np.mean(y_test == y_hat)
            tmp_f1s[fold]  = f1_score(y_test, y_hat, average='micro')

            # Calculate AUC
            if n_classes > 2: y_test = label_binarize(y_test, classes)
            y_df           = pipeline.decision_function(X_test)
            tmp_aucs[fold] = roc_auc_score(y_test, y_df)

            # Increase counter
            fold += 1
        
        # Update results
        accs[i] = tmp_accs.mean()
        aucs[i] = tmp_aucs.mean()
        f1s[i]  = tmp_f1s.mean()
    
    # Return averaged results
    return accs.mean(), aucs.mean(), f1s.mean()
        

def clf_metrics():
    """Calculate classifier metrics.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    results : pandas DataFrame
        Classifier metrics by method and number of features
    """
    # Load data
    with open(join(DATA_DIR, 'classifier.json')) as f:
        data = [json.loads(line) for line in f]

    # Calculate metrics for each data set and method based on feature ranks
    results   = []
    dfs_cache = {}
    for row in data:
        
        # Parse info
        name       = row['data']
        feat_ranks = row['results']['ranks']
        method     = row['results']['method']

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
            acc, auc, f1 = cv_scores(X_, y)
            
            # Append results
            results.append(
                [name, method, col, acc, auc, f1]
            )

    # Convert to dataframe and save to disk
    columns = ['data', 'method', 'n_feats', 'acc', 'auc', 'f1']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv(join(DATA_DIR, 'classifier_cv_metrics.csv'), index=False)

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
    # Iterate over each data set
    table = {}
    for data in metrics['data'].unique():
            
        summary = {}
        
        # Pivot table
        df = metrics[metrics['data'] == data]\
                .pivot_table(values=metric_name, index='method', columns='n_feats')
        
        # Raw metric summary statistics
        summary['mean']     = df.mean(axis=1)
        summary['sd']       = df.std(axis=1)
        summary['best']     = df.max(axis=1)
        summary['loc_best'] = df.apply(lambda x: np.argmax(x), axis=1)/df.columns.max()

        # Make dataframe and find best metrics per method
        summary = pd.DataFrame(summary)
        ct      = summary[summary.index.str.startswith('ct')]
        cf      = summary[summary.index.str.startswith('cf')]
        r_cf    = summary[summary.index.str.startswith('r_cf')]
        dt      = summary[summary.index.str.startswith('dt')]
        rf      = summary[summary.index.str.startswith('rf')]
        et      = summary[summary.index.str.startswith('et')]
        mc      = summary[summary.index.str.startswith('mc')]
        mi      = summary[summary.index.str.startswith('mi')]
        hy      = summary[summary.index.str.startswith('hybrid')]

        # Populate table with mean score
        table[data + '_mean'] = {
            'ct'   : ct['mean'].max(),
            'cf'   : cf['mean'].max(),
            'r_cf' : r_cf['mean'].max(),
            'dt'   : dt['mean'][0],
            'et'   : et['mean'][0],
            'rf'   : rf['mean'][0],
            'mc'   : mc['mean'][0],
            'mi'   : mi['mean'][0],
            'hy'   : hy['mean'][0],
        }

        # Given multiple params tested, find ct and cf hyperparameter 
        # configuration that yielded best mean score
        ct_config   = np.argmax(ct['mean'])
        cf_config   = np.argmax(cf['mean'])
        r_cf_config = np.argmax(r_cf['mean'])

        # Populate table with sd of row with mean scores
        table[data + '_sd'] = {
            'ct'   : ct[ct.index==ct_config]['sd'][0],
            'cf'   : cf[cf.index==cf_config]['sd'][0],
            'r_cf' : r_cf[r_cf.index==r_cf_config]['sd'][0],
            'dt'   : dt['sd'][0],
            'et'   : et['sd'][0],
            'rf'   : rf['sd'][0],
            'mc'   : mc['sd'][0],
            'mi'   : mi['sd'][0],
            'hy'   : hy['sd'][0],
        }

        # Populate table with best score
        table[data + '_best'] = {
            'ct'   : ct['best'].max(),
            'cf'   : cf['best'].max(),
            'r_cf' : r_cf['best'].max(),
            'dt'   : dt['best'][0],
            'et'   : et['best'][0],
            'rf'   : rf['best'][0],
            'mc'   : mc['best'][0],
            'mi'   : mi['best'][0],
            'hy'   : hy['best'][0],
        }

        # Given multiple params tested, find ct and cf hyperparameter 
        # configuration that yielded best overall score
        ct_config   = np.argmax(ct['best'])
        cf_config   = np.argmax(cf['best'])
        r_cf_config = np.argmax(r_cf['best'])

        # Populate table with % location of best score
        table[data + '_loc_best'] = {
            'ct'   : ct[ct.index==ct_config]['loc_best'][0],
            'cf'   : cf[cf.index==cf_config]['loc_best'][0],
            'r_cf' : r_cf[r_cf.index==r_cf_config]['loc_best'][0],
            'dt'   : dt['loc_best'][0],
            'et'   : et['loc_best'][0],
            'rf'   : rf['loc_best'][0],
            'mc'   : mc['loc_best'][0],
            'mi'   : mi['loc_best'][0],
            'hy'   : hy['loc_best'][0]
        }

    # Write results to disk
    pd.DataFrame(table).T.to_csv(
            join(DATA_DIR, 'table_' + metric_name + '.csv'),
            index=True
        )


def main():
    """Main function to run classifier analysis"""

    start = time.time()

    # Calculate raw metrics
    path = join(DATA_DIR, 'classifier_cv_metrics.csv')
    if not exists(path):
        metrics = clf_metrics()
    else:
        metrics = pd.read_csv(path)

    # Combine metrics from R
    try:
        r_metrics = pd.read_csv(join(DATA_DIR, 'r_classifier_cv_metrics.csv'))
    except:
        os.system('python run_r_classifier_models.py')
        r_metrics = pd.read_csv(join(DATA_DIR, 'r_classifier_cv_metrics.csv'))

    # Aggregate metrics and calculate different statistics for tables
    metrics = pd.concat([metrics, r_metrics], axis=0)
    aggregate_metrics(metrics, 'acc')
    aggregate_metrics(metrics, 'auc')
    aggregate_metrics(metrics, 'f1')

    # Script finished
    minutes = (time.time() - start)/60
    print("\n[info] Script finished in %.2f minutes" % minutes)

if __name__ == "__main__":
    main()