from __future__ import division, print_function

import json
import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC
import sys
import time

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from scorers import pcor, py_dcor

DATA_SETS = ['wine', 'TOX_171', 'Yale', 'orlraws10P', 'warpPIE10P', 'warpAR10P', 
             'pixraw10P', 'ALLAML', 'CLL_SUB_111', 'ORL', 'glass', 
             'vowel-context']

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


def binarize(y_true, y_hat):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    lb = LabelBinarizer().fit(y_true)
    return lb.transform(y_true), lb.transform(y_hat)


def rank_filter(X, y, filter_method='pearson'):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    fi = np.zeros(X.shape[1])
    
    if filter_method == 'pearson':
        for j in range(X.shape[1]): 
            fi[j] = np.fabs(pcor(X[:, j], y))
    
    elif filter_method == 'distance': 
        for j in range(X.shape[1]): 
            fi[j] = py_dcor(X[:, j], y, X.shape[0])
    
    elif filter_method == 'hybrid': 
        for j in xrange(X.shape[1]): 
            fi[j] = max(np.fabs(pcor(X[:, j], y)), py_dcor(X[:, j], y, X.shape[0]))

    else:
        for j in range(X.shape[1]):
            fi[j] = mutual_info_classif(X[:, j].reshape(-1, 1), y, random_state=1718)

    return fi, np.argsort(fi)[::-1]


def rank_tree(X, y, tree_model='rf'):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    params = {'class_weight': 'balanced', 'n_jobs': -1, 'n_estimators': 100}
    model  = RandomForestClassifier(**params) if tree_model == 'rf' else \
             ExtraTreesClassifier(**params)
    fi     = model.fit(X, y).feature_importances_
    
    return fi, np.argsort(fi)[::-1]


def all_cross_validate_svm(X, y, model, k, cv, fi_ranks):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Determine number of columns to use
    if X.shape[1] > 100:
        cols_to_keep = np.arange(5, 205, 5)
    else:
        cols_to_keep = np.arange(0, X.shape[1])+1

    # CV scores
    aucs, accs = np.zeros(len(cols_to_keep)), np.zeros(len(cols_to_keep))
    for i, col in enumerate(cols_to_keep):

        # Subset X
        X_ = X[:, fi_ranks[:col]].reshape(-1, col)

        print("[CV] Running CV with top %d features" % X_.shape[1])
        scores_auc, scores_acc = single_cross_validate_svm(X_, y, model, k, cv)

        print("\tAUC = %.4f" % scores_auc.mean())
        print("\tAcc = %.4f" % scores_acc.mean())

        # Append results
        aucs[i] = scores_auc.mean()
        accs[i] = scores_acc.mean()

    return aucs, accs, cols_to_keep


def single_cross_validate_svm(X, y, model, k, cv, verbose=False):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes, scores_auc, scores_acc, fold = len(set(y)), np.zeros(k), np.zeros(k), 0
    for train_idx, test_idx in cv.split(X, y):

        # Split into train and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize data based on training
        scaler          = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        # Train model and make predictions
        clf = clone(model).fit(X_train, y_train)

        if n_classes == 2:
            y_probs = clf.predict_proba(X_test)
            y_hat   = clf.predict(X_test)
            if min(y_test) != 0:
                y_test -= min(y_test)
                y_hat  -= min(y_test)

            # Calculate AUC
            try:
                scores_auc[fold] = roc_auc_score(y_test, y_probs[:, -1])
            except Exception as e:
                print("[ERROR] Exception raised computing AUC score because %s" % e)
                pass

            # Calculate accuracy
            try:
                scores_acc[fold] = accuracy_score(y_test, y_hat)
            except Exception as e:
                print("[ERROR] Exception raised computing accuracy score because %s" % e)
                pass

        else:
            # Make predictions
            y_hat = clf.predict(X_test)
            
            # Calculate accuracy
            try:
                scores_acc[fold] = accuracy_score(y_test, y_hat)
            except Exception as e:
                print("[ERROR] Exception raised computing accuracy score because %s" % e)
                pass

            # Calculate AUC
            try:
                # Binarize predictions for AUC score
                y_test, y_hat = binarize(y_test, y_hat)
                scores_auc[fold]  = roc_auc_score(y_test, y_hat)
            except Exception as e:
                print("[ERROR] Exception raised computing AUC score because %s" % e)
                pass

        # Next fold
        if verbose:
            print("[CV] Fold %d" % (fold+1))
            print("\tAUC = %.4f" % scores_auc[fold])
            print("\tAcc = %.4f" % scores_acc[fold])
        fold += 1

    # Overall metrics
    if verbose:
        print("[CV] Overall")
        print("\tAUC = %.4f +/- %.4f" % (scores_auc.mean(), scores_auc.std()))
        print("\tAcc = %.4f +/- %.4f" % (scores_acc.mean(), scores_acc.std()))

    return scores_auc, scores_acc


def prepare_ciforest_results():
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    df = json.load(open('forest_experiment_fi_results.json', 'r'))
    fi = {name:{} for name in DATA_SETS}
    for data in df:
        # Grab relevant data
        alpha    = data['alpha']
        et       = data['early_stopping']
        selector = data['selector'] 
        bayes    = data['bayes']

        # Update dictionary
        key = 'a_' + str(alpha) + '_es_' + str(int(et)) + '_' + selector + \
              '_bayes_' + str(int(bayes))
        fi[data['name']][key] = data['fi']

    return fi


def compare_methods():
    """ADD HERE"""

    # Load results
    ciforest_fi_all = prepare_ciforest_results()

    # Iterate over each data set
    results, start = {}, time.time()
    for name in DATA_SETS:

        # Load data
        X, y = load_data(name)
        print("[DATA] Name: %s" % name)
        print("[DATA] Shape: %s" % (X.shape,))
        print("[DATA] Labels: %s\n" % np.unique(y))

        # Base results
        results[name] = {
            'pcor': None,
            'dcor': None,
            'hybrid': None,
            'mi': None,
            'rf': None,
            'et': None,
        }

        # Grab results for conditional inference forests and update base results
        ciforest_fi = ciforest_fi_all[name]
        for key in ciforest_fi.iterkeys(): results[name].update({key: None})

        # Calculate ranks for filter methods
        print("[FI] Calculating ranks for filter methods")
        pcor_fi, pcor_ranks     = rank_filter(X, y, filter_method='pearson')
        dcor_fi, dcor_ranks     = rank_filter(X, y, filter_method='distance')
        hybrid_fi, hybrid_ranks = rank_filter(X, y, filter_method='hybrid')
        mi_fi, mi_ranks         = rank_filter(X, y, filter_method='mutual')

        # Calculate ranks for tree models
        print("[FI] Calculating ranks for embedded methods")
        rf_fi, rf_ranks = rank_tree(X, y, tree_model='rf')
        et_fi, et_ranks = rank_tree(X, y, tree_model='et')

        # Base SVM model and CV generator
        model = SVC(random_state=1718, probability=True)
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=1718)

        # Calculate scores for pearson correlation
        print("\n----- PEARSON CORRELATION -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=pcor_ranks)
        results[name]['pcor'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

        # Calculate scores for distance correlation
        print("\n----- DISTANCE CORRELATION -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=dcor_ranks)
        results[name]['dcor'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

        # Calculate scores for hybrid correlation
        print("\n----- HYBRID CORRELATION -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=hybrid_ranks)
        results[name]['hybrid'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

        # Calculate scores for mutual information
        print("\n----- MUTUAL INFORMATION -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=mi_ranks)
        results[name]['mi'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]
    
        # Calculate scores for random forest
        print("\n----- RANDOM FOREST -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=rf_ranks)
        results[name]['rf'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

        # Calculate scores for extra trees
        print("\n----- EXTRA TREES -----\n")
        aucs, accs, cols_to_keep = \
            all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=et_ranks)
        results[name]['et'] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

        # Iterate over all conditional inference forests
        for key, value in ciforest_fi.iteritems():

            # Rank features
            fi_ranks = np.argsort(value)[::-1]

            # Calculate scores for conditional inference forests
            print("\n----- %s -----\n" % key.upper())
            aucs, accs, cols_to_keep = \
                all_cross_validate_svm(X, y, model, k=5, cv=skf, fi_ranks=fi_ranks)
            results[name][key] = [aucs.tolist(), accs.tolist(), cols_to_keep.tolist()]

    # To json and write to disk
    with open('svm_fi_results.json', 'w') as f:
        json.dump(results, f)
    
    overall_time = (time.time() - start)/60.
    print("\nScript finished in %.2f minutes" % overall_time)


def rank_results():
    """ADD HERE"""
    
    from scipy.stats import rankdata

    results = json.load(open('svm_fi_results.json', 'r'))

    # Iterate over data sets
    for name, data in results.iteritems():
        tmp   = [value[0] for value in data.itervalues()]
        tmp   = np.column_stack(tmp)
        tmp   = np.row_stack([rankdata(-1*tmp[i,:], 'dense') for i in xrange(tmp.shape[0])])
        avg   = np.mean(tmp, axis=0)
        feats = np.array(data.keys())
        idx   = np.argsort(avg) 
        tmp   = zip(feats[idx], avg[idx])

        print("\n\n---- DATA: %s ----\n" % name.upper())
        for t in tmp: print(t) 


def plot_results():
    """ADD HERE"""

    results = json.load(open('svm_fi_results.json', 'r'))

    # Iterate over data sets
    for name, data in results.iteritems():
        
        f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
        for key, value in data.iteritems():
            axarr[0].plot(value[2], value[0], '-', label=key, alpha=.7)
            axarr[1].plot(value[2], value[1], '-', label=key, alpha=.7)
        axarr[0].set_title("AUC")
        axarr[1].set_title("Acc")
        plt.legend()
        plt.suptitle("Data: %s" % name)
        plt.show()


if __name__ == '__main__':
    #compare_methods()
    #plot_results()
    #rank_results()