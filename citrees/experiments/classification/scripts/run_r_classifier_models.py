import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
import shutil
from six import iteritems
import time

# Custom imports
from classifier_analysis import cv_scores
from classifier_experiment import load_data

# Constants
DATA_SETS  = ['wine', 'orlraws10P', 'glass', 'warpPIE10P', 'warpAR10P', 'pixraw10P',
              'ALLAML', 'CLL_SUB_111', 'ORL', 'TOX_171', 'Yale', 'musk',
              'vowel-context', 'gamma', 'isolet', 'letter', 'madelon',
              'page-blocks', 'pendigits', 'spam']
DATA_DIR   = '../data'
N_SPLITS   = 5
ITERATIONS = 10


def generate_csv_files():
    """Generate temporary csv files"""

    print("[info] Generating data")
    
    # Make temporary directory
    if not os.path.exists('r_data'): os.mkdir('r_data')

    # Begin experiments
    for name in DATA_SETS:

        X, y = load_data(name)
        if X.shape[0] > 2000: continue

        # For consistency, standardize features
        X_ = StandardScaler().fit_transform(X) 
        
        # Write to disk
        columns  = ['x' + str(j) for j in range(1, X_.shape[1]+1)] 
        columns += ['y']
        pd.DataFrame(np.column_stack([X_, y]), columns=columns)\
            .to_csv('r_data/%s.csv' % name, index=False)


def main():
    """Main function to generate classifier metrics from R results"""

    start = time.time()

    # Launch feature importance script in R
    if not os.path.exists('../data/r_classifier.json'):

        # Generate temporary csv files for easier read in R
        generate_csv_files()

        # Run script and then clean up temporary data
        system_call = 'Rscript --vanilla r_feature_importance.R %s' % os.getcwd()
        os.system(system_call)
        shutil.rmtree('r_data')

    # Run cv to get metrics
    if not os.path.exists('../data/r_classifier_cv_metrics.csv'):
        
        # Load data
        fi = eval(json.load(open('../data/r_classifier.json', 'r'))[0])

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
                    acc, auc, f1 = cv_scores(X_, y)
                    
                    # Append results
                    results.append(
                        [name, method, col, acc, auc, f1]
                    )

        # Convert to dataframe and save to disk
        columns = ['data', 'method', 'n_feats', 'acc', 'auc', 'f1']
        results = pd.DataFrame(results, columns=columns)
        results.to_csv('../data/r_classifier_cv_metrics.csv', index=False)

    # Script finished
    minutes = (time.time() - start) / 60.0
    print("[FINISHED] Script finished in %.2f minutes" % minutes)


if __name__ == "__main__":
    main()