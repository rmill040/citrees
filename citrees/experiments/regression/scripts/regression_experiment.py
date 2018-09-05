from __future__ import division, print_function

import json
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
import pymongo
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import sys
import time

# Add path to avoid relative imports
PATH = abspath(__file__).split('experiments')[0]
if PATH not in sys.path: sys.path.append(PATH)

from citrees import CIForestRegressor, CITreeRegressor
from externals.six.moves import range
from scorers import pcor, py_dcor

# Constants
DATA_DIR     = abspath(__file__).split('scripts')[0]
DATA_DIR     = join(DATA_DIR, 'data')
DATA_SETS    = ['coepra1', 'coepra2', 'coepra3', 'residential', 'comm_violence',
                'community_crime', 'facebook', 'imports-85']
RANDOM_STATE = 1718


def load_data(name):
    """Loads data, preprocesses, and splits into features and labels

    Parameters
    ----------
    name : str
        Name of data set

    Returns
    -------
    X : 2d array-like
        Array of features

    y : 1d array-like
        Array of labels
    """
    df = pd.read_csv(join(DATA_DIR, name + '.data')).replace('?', 0)

    if name in ['comm_violence', 'community_crime', 'facebook', 'imports-85']:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]

    elif name in ['coepra1', 'coepra2', 'coepra3']:
        X, y = df.iloc[:, df.columns != 'Prop_001'], df['Prop_001']

    elif name in ['residential']:
        X, y = df.drop(['V-9', 'V-10'], axis=1), df['V-9']

    else:
        raise ValueError("%s not a valid data set" % name)

    # Convert object dtypes to float
    if 'object' in X.dtypes.values:
        for col in X.select_dtypes(['object']).columns:
            try:
                X[col] = X[col].astype(float)
            except:
                X[col] = LabelEncoder().fit_transform(X[col])

    # Cast to float before returning
    return X.astype(float).values, y.astype(float).values


# Keep track of errors during experiment
n_errors = 0

def calculate_fi(X, y, name, collection=None):
    """Calculate feature importances for different methods"""

    # Access global errors variable
    global n_errors

    # 1. Pearson correlationv
    print("[FI] Pearson correlation")
    try:
        fi    = np.fabs([pcor(X[:, j], y) for j in range(X.shape[1])])
        ranks = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'pearson',
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Pearson correlation failed because %s" % e)
        pass

    # 2. Distance correlation
    print("[FI] Distance correlation")
    try:
        fi    = [py_dcor(X[:, j], y) for j in range(X.shape[1])]
        ranks = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'distance',
                'fi': fi,
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Distance correlation failed because %s" % e)
        pass

    # 3. Hybrid correlation
    print("[FI] Hybrid correlation")
    try:
        fi = []
        for j in range(X.shape[1]):
            pearson  = np.fabs(pcor(X[:, j], y))
            distance = py_dcor(X[:, j], y)
            if pearson > distance:
                fi.append(pearson)
            else:
                fi.append(distance)
        ranks = np.argsort(np.nan_to_num(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'hybrid',
                'fi': fi,
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
        fi    = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1) \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'rf',
                'params': {'n_estimators': 200, 'max_features': 'sqrt'},
                'fi': fi.tolist(),
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
        fi    = ExtraTreesRegressor(n_estimators=200,  max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1) \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'et',
                'params': {'n_estimators': 200, 'max_features': 'sqrt'},
                'fi': fi.tolist(),
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
        fi    = Lasso(alpha=.25, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'alpha': .25},
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] LASSO 1/3 failed because %s" % e)
        pass

    print("[FI] LASSO 2/3")
    try:
        fi    = Lasso(alpha=.15, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'alpha': .15},
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] LASSO 2/3 failed because %s" % e)
        pass

    print("[FI] LASSO 3/3")
    try:
        fi    = Lasso(alpha=.05, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'lasso',
                'params': {'alpha': .05},
                'fi': fi.tolist(),
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
        fi    = Ridge(alpha=.25, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'alpha': .25},
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Ridge 1/3 failed because %s" % e)
        pass

    print("[FI] Ridge 2/3")
    try:
        fi    = Ridge(alpha=.15, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'alpha': .15},
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Ridge 2/3 failed because %s" % e)
        pass

    print("[FI] Ridge 3/3")
    try:
        fi    = Ridge(alpha=.05, random_state=RANDOM_STATE).fit(X, y).coef_
        ranks = np.argsort(np.fabs(fi))[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'ridge',
                'params': {'alpha': .05},
                'fi': fi.tolist(),
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
        fi    = DecisionTreeRegressor(random_state=RANDOM_STATE, max_features='sqrt') \
                  .fit(X, y) \
                  .feature_importances_
        ranks = np.argsort(fi)[::-1]
        collection.insert_one({
            "data": name,
            "results": {
                'method': 'dt',
                'fi': fi.tolist(),
                'ranks': ranks.tolist()
            }
        })
    except Exception as e:
        n_errors += 1
        print("[ERROR] Decision tree failed because %s" % e)
        pass

    # Grid search for conditional tree models
    n_combos, counter = 3*2*2*3, 1
    for s in ['pearson', 'distance', 'hybrid']:
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
                        fi    = CITreeRegressor(**params) \
                                  .fit(X, y) \
                                  .feature_importances_
                        ranks = np.argsort(fi)[::-1]
                        collection.insert_one({
                            "data": name,
                            "results": {
                                'method': 'ct',
                                'params': params,
                                'fi': fi.tolist(),
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
                        fi    = CIForestRegressor(**params) \
                                  .fit(X, y) \
                                  .feature_importances_
                        ranks = np.argsort(fi)[::-1]
                        collection.insert_one({
                            "data": name,
                            "results": {
                                'method': 'cf',
                                'params': params,
                                'fi': fi.tolist(),
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
    """ADD DESCRIPTION"""

    start = time.time()

    # Connect to mongodb
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
    except:
        raise IOError("Mongo client not detected. Start MongoDB on localhost " \
                      "using port 27017 and try again")

    # Define database and collection
    db = client["fi"]
    if "regression" in db.collection_names(): db.regression.drop()
    collection = db["regression"]

    # Begin experiments
    for name in DATA_SETS:
        X, y = load_data(name)
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
