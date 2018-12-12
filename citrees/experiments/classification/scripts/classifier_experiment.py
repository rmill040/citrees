from __future__ import division, print_function

import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
import pymongo
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import sys
import time

# Add path to avoid relative imports
PATH = abspath(__file__).split('experiments')[0]
if PATH not in sys.path: sys.path.append(PATH)

from citrees import CITreeClassifier, CIForestClassifier
from externals.six.moves import range
from scorers import mc_fast, mi

# Constants
DATA_DIR     = abspath(__file__).split('scripts')[0]
DATA_DIR     = join(DATA_DIR, 'data')
DATA_SETS    = ['wine', 'orlraws10P', 'glass', 'warpPIE10P', 'warpAR10P', 'pixraw10P',
                'ALLAML', 'CLL_SUB_111', 'ORL', 'TOX_171', 'Yale', 'musk',
                'vowel-context', 'gamma', 'isolet', 'letter', 'madelon',
                'page-blocks', 'pendigits', 'spam']
RANDOM_STATE = 1718


def load_data(name):
    """Load baseline data

    Parameters
    ----------
    name : str
        Name of data set

    Returns
    -------
    X : 2d array-like
        Features

    y : 1d array-like
        Labels
    """
    if name in ['ALLAML', 'CLL_SUB_111', 'ORL', 'orlraws10P',
                'pixraw10P', 'TOX_171', 'warpAR10P', 'warpPIE10P',
                'Yale']:

        # Load data
        df   = loadmat(join(DATA_DIR, name + '.mat'))
        X, y = df.values()[1], df.values()[0]

    elif name in ['glass', 'wine', 'vowel-context', 'gamma', 'isolet', 'letter',
                  'madelon', 'musk', 'page-blocks', 'pendigits', 'spam']:

        # Load data
        df = pd.read_table(join(DATA_DIR, name + '.data'), delimiter=',', header=None)

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


# Keep track of errors during experiment
n_errors = 0

def calculate_fi(X, y, name, collection=None):
    """Calculate feature importances for different methods"""

    # Access global errors variable
    global n_errors

    # 1. Multiple correlation
    print("[FI] Multiple correlation")
    try:
        n_classes = len(set(y))
        fi        = [mc_fast(X[:, j], y, n_classes) for j in range(X.shape[1])]
        ranks     = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'mc',
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Multiple correlation failed because %s" % e)
        pass

    # 2. Mutual information
    print("[FI] Mutual information")
    try:
        fi    = [mi(X[:, j], y) for j in range(X.shape[1])]
        ranks = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'mi',
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Mutual information failed because %s" % e)
        pass

    # 3. Hybrid correlation
    print("[FI] Hybrid correlation")
    try:
        fi, n_classes = [], len(set(y))
        for j in range(X.shape[1]):
            multiple = mc_fast(X[:, j], y, n_classes)
            mutual   = mi(X[:, j], y)
            if multiple > mutual:
                fi.append(multiple)
            else:
                fi.append(mutual)
        ranks = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'hybrid',
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Hybrid correlation failed because %s" % e)
        pass

    # 4. Random Forest
    print("[FI] Random forest")
    try:
        fi    = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1) \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'rf',
                'params': {'n_estimators': 200, 'max_features': 'sqrt'},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Random forest failed because %s" % e)
        pass

    # 5. Extra Trees
    print("[FI] Extra trees")
    try:
        fi    = ExtraTreesClassifier(n_estimators=200,  max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1) \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'et',
                'params': {'n_estimators': 200, 'max_features': 'sqrt'},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Extra trees failed because %s" % e)
        pass

    # 6. Lasso regression (different regularization parameters)
    print("[FI] LASSO 1/3")
    try:
        fi    = LogisticRegression(penalty='l1', C=.25, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'C': .25},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] LASSO 1/3 failed because %s" % e)
        pass

    print("[FI] LASSO 2/3")
    try:
        fi    = LogisticRegression(penalty='l1', C=.15, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'C': .15},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] LASSO 2/3 failed because %s" % e)
        pass

    print("[FI] LASSO 3/3")
    try:
        fi    = LogisticRegression(penalty='l1', C=.05, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'C': .05},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] LASSO 3/3 failed because %s" % e)
        pass

    # 7. Ridge regression
    print("[FI] Ridge 1/3")
    try:
        fi    = LogisticRegression(penalty='l2', C=.25, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'C': .25},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Ridge 1/3 failed because %s" % e)
        pass

    print("[FI] Ridge 2/3")
    try:
        fi    = LogisticRegression(penalty='l2', C=.15, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'C': .15},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Ridge 2/3 failed because %s" % e)
        pass

    print("[FI] Ridge 3/3")
    try:
        fi    = LogisticRegression(penalty='l2', C=.05, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'C': .05},
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Ridge 3/3 failed because %s" % e)
        pass

    # 8. Decision tree
    print("[FI] Decision tree")
    try:
        fi    = DecisionTreeClassifier(random_state=RANDOM_STATE, max_features='sqrt') \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'dt',
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Decision tree failed because %s" % e)
        pass

    # Grid search for conditional tree models
    n_combos, counter = 3*2*2*3, 1
    for s in ['mc', 'mi', 'hybrid']:
        for e in [True, False]:
            for m in [True, False]:
                for a in [.01, .05, .95]:

                    # Define hyperparameters
                    params = {
                        'n_permutations': 150,
                        'selector': s,
                        'max_feats': 'sqrt',
                        'early_stopping': e,
                        'muting': m,
                        'alpha': a,
                        'n_jobs': -1,
                        'verbose': 1,
                        'random_state': RANDOM_STATE
                    }

                    # 9. Conditional inference tree
                    print("[FI] Conditional tree %d/%d" % (counter, n_combos))
                    try:
                        fi    = CITreeClassifier(**params) \
                                  .fit(X, y) \
                                  .feature_importances_
                        ranks = np.argsort(fi)[::-1]
                        collection.insert_one({
                            "data": name,
                            "results": {
                                'method': 'ct',
                                'params': params,
                                'ranks': ranks.tolist()
                            }
                        })
                    except Exception as e:
                        n_errors += 1
                        print("[ERROR] Conditional tree %d/%d failed because %s" % \
                                    (counter, n_combos, e))
                        pass

                    # Add in number of trees for the forest
                    params['n_estimators'] = 200

                    # 10. Conditional inference forest
                    print("[FI] Conditional forest %d/%d" % (counter, n_combos))
                    try:
                        fi    = CIForestClassifier(**params) \
                                  .fit(X, y) \
                                  .feature_importances_
                        ranks = np.argsort(fi)[::-1]
                        collection.insert_one({
                            "data": name,
                            "results": {
                                'method': 'cf',
                                'params': params,
                                'ranks': ranks.tolist()
                            }
                        })
                    except Exception as e:
                        n_errors += 1
                        print("[ERROR] Conditional forest %d/%d failed because %s" % \
                                    (counter, n_combos, e))
                        pass

                    # Increase counter
                    counter += 1


def main():
    """Run experiment and save results"""

    start = time.time()

    # Connect to mongodb
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
    except:
        raise IOError("Mongo client not detected. Start MongoDB on localhost " \
                      "using port 27017 and try again")

    # Define database and collection
    db = client["fi"]
    if "classifier" in db.collection_names(): db.regression.drop()
    collection = db["classifier"]

    # Begin experiments
    for name in DATA_SETS:
        X, y = load_data(name)
        if X.shape[0] > 2000: continue
        print("\n\n[DATA] Name = %s, Samples = %s, Features = %s" % \
              (name, X.shape[0], X.shape[1]))

        # Calculate feature importances for each method
        X_ = StandardScaler().fit_transform(X) # Standardize features first
        calculate_fi(X_, y, name, collection=collection)

    # Script finished
    global n_errors
    minutes = (time.time()-start)/60.0
    print("[FINISHED] Script finished in %.2f minutes with %d errors" % \
                (minutes, n_errors))

if __name__ == '__main__':
    main()
