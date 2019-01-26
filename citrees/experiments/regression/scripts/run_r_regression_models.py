import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shutil
from six import iteritems
import time

from regression_analysis import cv_scores
from regression_experiment import load_data

# Constants
DATA_SETS = ['coepra1', 'coepra2', 'coepra3', 'residential', 'comm_violence',
             'community_crime', 'facebook', 'imports-85']


def generate_csv_files():
    """Generate temporary csv files"""

    print("[info] Generating data")
    
    # Make temporary directory
    if not os.path.exists('r_data'): os.mkdir('r_data')

    # Begin experiments
    for name in DATA_SETS:

        X, y = load_data(name)

        # For consistency, standardize features
        X_ = StandardScaler().fit_transform(X) 
        
        # Write to disk
        columns  = ['x' + str(j) for j in range(1, X_.shape[1]+1)] 
        columns += ['y']
        pd.DataFrame(np.column_stack([X_, y]), columns=columns)\
            .to_csv('r_data/%s.csv' % name, index=False)
        

def main():
    """Main function to generate regression metrics from R results"""

    start = time.time()

    # Launch feature importance script in R
    if not os.path.exists('../data/r_regression.json'):

        # Generate temporary csv files for easier read in R
        generate_csv_files()

        # Run script and then clean up temporary data
        system_call = 'Rscript --vanilla r_feature_importance.R %s' % os.getcwd()
        os.system(system_call)
        shutil.rmtree('r_data')

    # Run cv to get metrics
    if not os.path.exists('../data/r_regression_cv_metrics.csv'):
        
        # Load data
        fi = eval(json.load(open('../data/r_regression.json', 'r'))[0])

        # Calculate metrics for each data set and method based on feature ranks
        results   = []
        dfs_cache = {}
        for name, data in iteritems(fi):
            for method, feat_ranks in iteritems(data):

                # Update method name                    
                method = 'r_cf_alpha_' + method
                
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
        results.to_csv('../data/r_regression_cv_metrics.csv', index=False)

    # Script finished
    minutes = (time.time() - start) / 60.0
    print("[FINISHED] Script finished in %.2f minutes" % minutes)


if __name__ == "__main__":
    main()