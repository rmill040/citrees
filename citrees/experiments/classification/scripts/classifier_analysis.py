from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC
import sys
import time

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from classifier_experiment import load_data
from scorers import pcor, py_dcor


def binarize(y_true, y_hat):
    """Binarize labels for multi-class AUC

    Parameters
    ----------
    y_true : 1d array-like
        True labels

    y_hat : 1d array-like
        Predicted labels

    Returns
    -------
    y_true_lb : 1d array-like
        Label binarized true labels

    y_hat_lb : 1d array-like
        Label binarized predicted labels
    """
    lb = LabelBinarizer().fit(y_true)
    return lb.transform(y_true), lb.transform(y_hat)


def recursive_search(tree, splits):
    """Recursively search tree structure for split points

    Parameters
    ----------
    tree : conditional inference model
        Trained conditional inference tree model

    splits : list
        Structure to append splits

    Returns
    -------
    None
    """
    if tree.value is None:
        splits.append(tree.col)
        recursive_search(tree.left_child, splits)
        recursive_search(tree.right_child, splits)


def collect_splits(tree, sklearn=False):
    """Finds all splits for tree models

    Parameters
    ----------
    tree : conditional inference model
        Trained conditional inference tree model

    sklearn : bool
        ADD HERE

    Returns
    -------
    splits : 1d array-like
        ADD HERE
    """
    splits = []
    for estimator in tree.estimators_:
        if sklearn:
            tmp = estimator.tree_.feature
            splits.append(tmp[tmp != -2])
        else:
            tmp = []
            recursive_search(estimator.root, tmp)
            splits.append(tmp)

    return np.concatenate(splits)


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
    pass
