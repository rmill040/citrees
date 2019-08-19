from __future__ import absolute_import, division, print_function

from joblib import delayed, Parallel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import multiprocessing
import threading
import warnings
warnings.simplefilter('ignore')

# Package imports
from externals.six.moves import range
from feature_selectors import (permutation_test_mc, permutation_test_mi,
                               permutation_test_dcor, permutation_test_pcor,
                               permutation_test_rdc)
from feature_selectors import mc_fast, mi, pcor, py_dcor
from scorers import gini_index, mse
from utils import bayes_boot_probs, logger


###################
"""SINGLE MODELS"""
###################


class Node(object):
    """Decision node in tree

    Parameters
    ----------
    col : int
        Integer indexing the location of feature or column

    col_pval : float
        Probability value from permutation test for feature selection

    threshold : float
        Best split found in feature

    impurity : float
        Impurity measuring quality of split

    value : 1d array-like or float
        For classification trees, estimate of each class probability
        For regression trees, central tendency estimate

    left_child : tuple
        For left child node, two element tuple with first element a 2d array of
        features and second element a 1d array of labels

    right_child : tuple
        For right child node, two element tuple with first element a 2d array of
        features and second element a 1d array of labels
    """
    def __init__(self, col=None, col_pval=None, threshold=None, impurity=None,
                 value=None, left_child=None, right_child=None):
        self.col         = col
        self.col_pval    = col_pval
        self.threshold   = threshold
        self.impurity    = impurity
        self.value       = value
        self.left_child  = left_child
        self.right_child = right_child


class CITreeBase(object):
    """Base class for conditional inference tree

    Parameters
    ----------
    min_samples_split : int
        Minimum samples required for a split

    alpha : float
        Threshold value for selecting feature with permutation tests. Smaller
        values correspond to shallower trees

    max_depth : int
        Maximum depth to grow tree

    max_feats : str or int
        Maximum feats to select at each split. String arguments include 'sqrt',
        'log', and 'all'

    n_permutations : int
        Number of permutations during feature selection

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    muting : bool
        Whether to perform variable muting

    verbose : bool or int
        Controls verbosity of training and testing

    n_jobs : int
        Number of jobs for permutation testing

    random_state : int
        Sets seed for random number generator
    """
    def __init__(self, min_samples_split=2, alpha=.05, max_depth=-1,
                 max_feats=-1, n_permutations=100, early_stopping=False,
                 muting=True, verbose=0, n_jobs=-1, random_state=None):

        # Error checking
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be in (0, 1]" % alpha)
        if n_permutations < 0:
            raise ValueError("n_permutations (%d) should be > 0" % \
                             n_permutations)
        if not isinstance(max_feats, int) and max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))

        # Define attributes
        self.alpha             = float(alpha)
        self.min_samples_split = max(1, int(min_samples_split))
        self.n_permutations    = int(n_permutations)
        self.max_feats         = max_feats
        self.early_stopping    = early_stopping
        self.muting            = muting
        self.verbose           = verbose
        self.n_jobs            = n_jobs
        self.root              = None
        self.splitter_counter_ = 0

        if max_depth == -1:
            self.max_depth = np.inf
        else:
            self.max_depth = int(max(1, max_depth))

        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)


    def _mute_feature(self, col_to_mute):
        """Removes variable from being selected

        Parameters
        ----------
        col_to_mute : int
            Integer index of column to remove
        """
        # Remove feature from protected features array
        idx = np.where(self.available_features_ == col_to_mute)[0]

        # Mute feature if not in protected set
        if idx in self.protected_features_:
            return
        else:
            self.available_features_ = np.delete(self.available_features_, idx)

            # Recalculate actual number for max_feats before fitting
            p = self.available_features_.shape[0]
            if self.max_feats == 'sqrt':
                self.max_feats = int(np.sqrt(p))
            elif self.max_feats == 'log':
                self.max_feats = int(np.log(p+1))
            elif self.max_feats in ['all', -1]:
                self.max_feats = p
            else:
                self.max_feats = int(self.max_feats)

            # Check to make sure max_feats is not larger than the number of remaining
            # features
            if self.max_feats > len(self.available_features_):
                self.max_feats = len(self.available_features_)


    def _selector(self, X, y, col_idx):
        """Find feature most correlated with label"""
        raise NotImplementedError("_splitter method not callable from base class")


    def _splitter(self, *args, **kwargs):
        """Finds best split for feature"""
        raise NotImplementedError("_splitter method not callable from base class")


    def _build_tree(self, X, y, depth=0):
        """Recursively builds tree

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        depth : int
            Depth of current recursive call

        Returns
        -------
        Node : object
            Child node or terminal node in recursive splitting
        """
        n, p = X.shape

        # Check for stopping criteria
        if n > self.min_samples_split and \
           depth < self.max_depth and \
           not np.all(y == y[0]):

            # Controls randomness of column sampling
            self.splitter_counter_ += 1
            np.random.seed(self.random_state*self.splitter_counter_)

            # Find column with strongest association with outcome
            try:
                col_idx = np.random.choice(self.available_features_,
                                           size=self.max_feats, replace=False)
            except:
                col_idx = np.random.choice(self.available_features_,
                                           size=len(self.available_features_),
                                           replace=False)
            col, col_pval = self._selector(X, y, col_idx)

            # Add selected feature to protected features
            if col not in self.protected_features_:
                self.protected_features_.append(col)
                if self.verbose > 1:
                    logger("tree", "Added feature %d to protected set, size "
                           "= %d" % (col, len(self.protected_features_)))

            if col_pval <= self.alpha:

                # Find best split among selected variable
                impurity, threshold, left, right = self._splitter(X, y, n, col)
                if left and right and len(left[0]) > 0 and len(right[0]) > 0:

                    # Build subtrees for the right and left branches
                    if self.verbose:
                        logger("tree", "Building left subtree with "
                                       "%d samples at depth %d" % \
                                       (len(left[0]), depth+1))
                    left_child = self._build_tree(*left, depth=depth+1)

                    if self.verbose:
                        logger("tree", "Building right subtree with "
                                       "%d samples at depth %d" % \
                                        (len(right[0]), depth+1))
                    right_child = self._build_tree(*right, depth=depth+1)

                    # Return all arguments to constructor except value
                    return Node(col=col, col_pval=col_pval, threshold=threshold,
                                left_child=left_child, right_child=right_child,
                                impurity=impurity)

        # Calculate terminal node value
        if self.verbose: logger("tree", "Root node reached at depth %d" % depth)
        value = self.node_estimate(y)

        # Terminal node, no other values to pass to constructor
        return Node(value=value)


    def fit(self, X, y=None):
        """Trains model

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        Returns
        -------
        self : CITreeBase
            Instance of CITreeBase class
        """
        if self.verbose:
            logger("tree", "Building root node with %d samples" % X.shape[0])

        # Calculate actual number for max_feats before fitting
        p = X.shape[1]
        if self.max_feats == 'sqrt':
            self.max_feats = int(np.sqrt(p))
        elif self.max_feats == 'log':
            self.max_feats = int(np.log(p+1))
        elif self.max_feats in ['all', -1]:
            self.max_feats = p
        else:
            self.max_feats = int(self.max_feats)

        # Begin recursive build
        self.protected_features_  = []
        self.available_features_  = np.arange(p, dtype=int)
        self.feature_importances_ = np.zeros(p)
        self.root                 = self._build_tree(X, y)
        sum_fi                    = np.sum(self.feature_importances_)
        if sum_fi > 0: self.feature_importances_ /= sum_fi

        return self


    def predict_label(self, X, tree=None):
        """Predicts label

        Parameters
        ----------
        X : 2d array-like
            Array of features for single sample

        tree : CITreeBase
            Trained tree

        Returns
        -------
        label : int or float
            Predicted label
        """
        # If we have a value => return value as the prediction
        if tree is None: tree = self.root
        if tree.value is not None: return tree.value

        # Determine if we will follow left or right branch
        feature_value = X[tree.col]
        branch        = tree.left_child if feature_value <= tree.threshold \
                                        else tree.right_child

        # Test subtree
        return self.predict_label(X, branch)


    def predict(self, *args, **kwargs):
        """Predicts labels on test data"""
        raise NotImplementedError("predict method not callable from base class")


    def print_tree(self, tree=None, indent=" ", child=None):
        """Prints tree structure

        Parameters
        ----------
        tree : CITreeBase
            Trained tree model

        indent : str
            Indent spacing

        child : Node
            Left or right child node
        """
        # If we're at leaf => print the label
        if not tree: tree = self.root
        if tree.value is not None: print("label:", tree.value)

        # Go deeper down the tree
        else:
            # Print splitting rule
            print("X[:,%s] %s %s " % (tree.col,
                                      '<=' if child in [None, 'left'] else '>',
                                      tree.threshold))

            # Print the left child
            print("%sL: " % (indent), end="")
            self.print_tree(tree.left_child, indent + indent, 'left')

            # Print the right
            print("%sR: " % (indent), end="")
            self.print_tree(tree.right_child, indent + indent, 'right')


class CITreeClassifier(CITreeBase, BaseEstimator, ClassifierMixin):
    """Conditional inference tree classifier

    Parameters
    ----------
    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    Derived from CITreeBase class; see constructor for parameter definitions

    """
    def __init__(self,
                 min_samples_split=2,
                 alpha=.05,
                 selector='mc',
                 max_depth=-1,
                 max_feats=-1,
                 n_permutations=100,
                 early_stopping=False,
                 muting=True,
                 verbose=0,
                 n_jobs=-1,
                 random_state=None):

        # Define node estimate
        self.node_estimate = self._estimate_proba

        # Define selector
        if selector not in ['mc', 'mi', 'hybrid']:
            raise ValueError("%s not a valid selector, valid selectors are " \
                             "mc, mi, and hybrid")
        self.selector = selector

        if self.selector != 'hybrid':
            # Wrapper correlation selector
            self._selector = self._cor_selector

            # Permutation test based on correlation measure
            if self.selector == 'mc':
                self._perm_test = permutation_test_mc
            else:
                self._perm_test = permutation_test_mi

        else:
            self._perm_test = None
            self._selector  = self._hybrid_selector

        super(CITreeClassifier, self).__init__(
                    min_samples_split=min_samples_split,
                    alpha=alpha,
                    max_depth=max_depth,
                    max_feats=max_feats,
                    n_permutations=n_permutations,
                    early_stopping=early_stopping,
                    muting=muting,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    random_state=random_state)


    def _hybrid_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a hybrid of multiple correlation and mutual information measures

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        # Iterate over columns
        for col in col_idx:
            if mc_fast(X[:, col], y, self.n_classes_) >= mi(X[:, col], y):
                pval = permutation_test_mc(x=X[:, col],
                                           y=y,
                                           n_classes=self.n_classes_,
                                           B=self.n_permutations,
                                           random_state=self.random_state)
            else:
                pval = permutation_test_mi(x=X[:, col],
                                           y=y,
                                           B=self.n_permutations,
                                           random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _splitter(self, X, y, n, col):
        """Splits data set into two child nodes based on optimized weighted
        gini index

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        n : int
            Number of samples

        col : list
            Column of X to search for best split

        Returns
        -------
        best_impurity : float
            Gini index associated with best split

        best_threshold : float
            X value associated with splitting of data set into two child nodes

        left : tuple
            Left child node data consisting of two elements: (features, labels)

        right : tuple
            Right child node data consisting of two elements: (features labels)
        """
        if self.verbose > 1:
            logger("splitter", "Testing splits on feature %d" % col)

        # Initialize variables for splitting
        impurity, threshold = 0.0, None
        left, right         = None, None

        # Call sklearn's optimized implementation of decision tree classifiers
        # to make split using Gini index
        base = DecisionTreeClassifier(
                max_depth=1, min_samples_split=self.min_samples_split
            ).fit(X[:, col].reshape(-1, 1), y).tree_

        # Make split based on best threshold
        threshold        = base.threshold[0]
        idx              = np.where(X[:, col] <= threshold, 1, 0)
        X_left, y_left   = X[idx==1], y[idx==1]
        X_right, y_right = X[idx==0], y[idx==0]
        n_left, n_right  = X_left.shape[0], X_right.shape[0]

        # Skip small splits
        if n_left < self.min_samples_split or n_right < self.min_samples_split:
            return impurity, threshold, left, right

        # Calculate parent and weighted children impurities
        if len(base.impurity) == 3:
            node_impurity  = base.impurity[0]
            left_impurity  = base.impurity[1]*(n_left/float(n))
            right_impurity = base.impurity[2]*(n_right/float(n))
        else:
            node_impurity  = gini_index(y, self.labels_)
            left_impurity  = gini_index(y_left, self.labels_)*(n_left/float(n))
            right_impurity = gini_index(y_right, self.labels_)*(n_right/float(n))

        # Define groups and calculate impurity decrease
        left, right = (X_left, y_left), (X_right, y_right)
        impurity    = node_impurity - (left_impurity + right_impurity)

        # Update feature importance (mean decrease impurity)
        self.feature_importances_[col] += impurity

        return impurity, threshold, left, right


    def _cor_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a correlation measure

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        # Iterate over columns
        for col in col_idx:

            # Mute feature and continue since constant
            if np.all(X[:, col] == X[0, col]) and len(self.available_features_) > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "Constant values, muting feature %d" \
                                        % col)
                continue

            pval = self._perm_test(x=X[:, col],
                                   y=y,
                                   n_classes=self.n_classes_,
                                   B=self.n_permutations,
                                   random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _estimate_proba(self, y):
        """Estimates class distribution in node

        Parameters
        ----------
        y : 1d array-like
            Array of labels

        Returns
        -------
        class_probs : 1d array-like
            Array of class probabilities
        """
        return np.array([np.mean(y == label) for label in self.labels_])


    def fit(self, X, y, labels=None):
        """Trains conditional inference tree classifier

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        labels : 1d array-like
            Array of unique class labels

        Returns
        -------
        self : CITreeClassifier
            Instance of CITreeClassifier class
        """
        self.labels_    = labels if labels is not None else np.unique(y)
        self.n_classes_ = len(self.labels_)
        super(CITreeClassifier, self).fit(X, y)
        return self


    def predict_proba(self, X):
        """Predicts class probabilities for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        class_probs : 2d array-like
            Array of predicted class probabilities
        """
        if self.verbose:
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        return np.array([self.predict_label(sample) for sample in X])


    def predict(self, X):
        """Predicts class labels for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        y : 1d array-like
            Array of predicted classes
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)


class CITreeRegressor(CITreeBase, BaseEstimator, RegressorMixin):
    """Conditional inference tree regressor

    Parameters
    ----------
    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    Derived from CITreeBase class; see constructor for rest of parameter definitions

    """
    def __init__(self,
                 min_samples_split=2,
                 alpha=.05,
                 selector='pearson',
                 max_depth=-1,
                 max_feats=-1,
                 n_permutations=100,
                 early_stopping=False,
                 muting=True,
                 verbose=0,
                 n_jobs=-1,
                 random_state=None):

        # Define node estimate
        self.node_estimate = self._estimate_mean

        # Define selector
        if selector not in ['pearson', 'distance', 'rdc', 'hybrid']:
            raise ValueError("%s not a valid selector, valid selectors are " \
                             "pearson, distance, rdc, and hybrid")
        self.selector = selector

        if self.selector != 'hybrid':
            # Wrapper correlation selector
            self._selector = self._cor_selector

            # Permutation test based on correlation measure
            if self.selector == 'pearson':
                self._perm_test = permutation_test_pcor
            elif self.selector == 'distance':
                self._perm_test = permutation_test_dcor
            else:
                self._perm_test = permutation_test_rdc

        else:
            self._perm_test = None
            self._selector  = self._hybrid_selector

        super(CITreeRegressor, self).__init__(
                    min_samples_split=min_samples_split,
                    alpha=alpha,
                    max_depth=max_depth,
                    max_feats=max_feats,
                    n_permutations=n_permutations,
                    early_stopping=early_stopping,
                    muting=muting,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    random_state=random_state)


    def _hybrid_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a hybrid of pearson and distance correlation measures

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        # Iterate over columns
        for col in col_idx:

            if abs(pcor(X[:, col], y)) >= abs(py_dcor(X[:, col], y)):
                pval = permutation_test_pcor(x=X[:, col],
                                             y=y,
                                             B=self.n_permutations,
                                             random_state=self.random_state)
            else:
                pval = permutation_test_dcor(x=X[:, col],
                                             y=y,
                                             B=self.n_permutations,
                                             random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _cor_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a correlation measure

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        # Iterate over columns
        for col in col_idx:

            # Mute feature and continue since constant
            if np.all(X[:, col] == X[0, col]) and len(self.available_features_) > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "Constant values, muting feature %d" \
                                        % col)
                continue

            pval = self._perm_test(x=X[:, col],
                                   y=y,
                                   B=self.n_permutations,
                                   random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _splitter(self, X, y, n, col):
        """Splits data set into two child nodes based on optimized weighted
        mean squared error

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        n : int
            Number of samples

        col : list
            Column of X to search for best split

        Returns
        -------
        best_impurity : float
            Mean squared error associated with best split

        best_threshold : float
            X value associated with splitting of data set into two child nodes

        left : tuple
            Left child node data consisting of two elements: (features, labels)

        right : tuple
            Right child node data consisting of two elements: (features labels)
        """
        if self.verbose > 1:
            logger("splitter", "Testing splits on feature %d" % col)

        # Initialize variables for splitting
        impurity, threshold = 0.0, None
        left, right         = None, None

        # Call sklearn's optimized implementation of decision tree regressors
        # to make split using mean squared error
        base = DecisionTreeRegressor(
                max_depth=1, min_samples_split=self.min_samples_split
            ).fit(X[:, col].reshape(-1, 1), y).tree_

        # Make split based on best threshold
        threshold        = base.threshold[0]
        idx              = np.where(X[:, col] <= threshold, 1, 0)
        X_left, y_left   = X[idx==1], y[idx==1]
        X_right, y_right = X[idx==0], y[idx==0]
        n_left, n_right  = X_left.shape[0], X_right.shape[0]

        # Skip small splits
        if n_left < self.min_samples_split or n_right < self.min_samples_split:
            return impurity, threshold, left, right

        # Calculate parent and weighted children impurities
        if len(base.impurity) == 3:
            node_impurity  = base.impurity[0]
            left_impurity  = base.impurity[1]*(n_left/float(n))
            right_impurity = base.impurity[2]*(n_right/float(n))
        else:
            node_impurity  = mse(y)
            left_impurity  = mse(y_left)*(n_left/float(n))
            right_impurity = mse(y_right)*(n_right/float(n))


        # Define groups and calculate impurity decrease
        left, right = (X_left, y_left), (X_right, y_right)
        impurity    = node_impurity - (left_impurity + right_impurity)

        # Update feature importance (mean decrease impurity)
        self.feature_importances_[col] += impurity

        return impurity, threshold, left, right


    def _estimate_mean(self, y):
        """Estimates mean in node

        Parameters
        ----------
        y : 1d array-like
            Array of labels

        Returns
        -------
        mu : float
            Node mean estimate
        """
        return np.mean(y)


    def fit(self, X, y):
        """Trains conditional inference tree regressor

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        Returns
        -------
        self : CITreeRegressor
            Instance of CITreeRegressor class
        """
        super(CITreeRegressor, self).fit(X, y)
        return self


    def predict(self, X):
        """Predicts labels for feature vectors in X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        y_hat : 1d array-like
            Array of predicted labels
        """
        if self.verbose:
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        return np.array([self.predict_label(sample) for sample in X])


#####################
"""ENSEMBLE MODELS"""
#####################


def stratify_sampled_idx(random_state, y, bayes):
    """Indices for stratified bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Stratified sampled indices for each class
    """
    np.random.seed(random_state)
    idx = []
    for label in np.unique(y):

        # Grab indices for class
        tmp = np.where(y==label)[0]

        # Bayesian bootstrapping if specified
        p = bayes_boot_probs(n=len(tmp)) if bayes else None

        idx.append(np.random.choice(tmp, size=len(tmp), replace=True, p=p))

    return idx


def stratify_unsampled_idx(random_state, y, bayes):
    """Unsampled indices for stratified bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Stratified unsampled indices for each class
    """
    np.random.seed(random_state)
    sampled = stratify_sampled_idx(random_state, y, bayes)
    idx     = []
    for i, label in enumerate(np.unique(y)):
        idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
    return idx


def balanced_sampled_idx(random_state, y, bayes, min_class_p):
    """Indices for balanced bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    idx : list
        Balanced sampled indices for each class
    """
    np.random.seed(random_state)
    idx, n = [], int(np.floor(min_class_p*len(y)))
    for i, label in enumerate(np.unique(y)):

        # Grab indices for class
        tmp = np.where(y==label)[0]

        # Bayesian bootstrapping if specified
        p = bayes_boot_probs(n=len(tmp)) if bayes else None

        idx.append(np.random.choice(tmp, size=n, replace=True, p=p))

    return idx


def balanced_unsampled_idx(random_state, y, bayes, min_class_p):
    """Unsampled indices for balanced bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    idx : list
        Balanced unsampled indices for each class
    """
    np.random.seed(random_state)
    sampled = balanced_sampled_idx(random_state, y, bayes, min_class_p)
    idx     = []
    for i, label in enumerate(np.unique(y)):
        idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
    return idx


def normal_sampled_idx(random_state, n, bayes):
    """Indices for bootstrap sampling

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    n : int
        Sample size

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Sampled indices
    """
    np.random.seed(random_state)

    # Bayesian bootstrapping if specified
    p = bayes_boot_probs(n=n) if bayes else None

    return np.random.choice(np.arange(n, dtype=int), size=n, replace=True, p=p)


def normal_unsampled_idx(random_state, n, bayes):
    """Unsampled indices for bootstrap sampling

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    n : int
        Sample size

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Unsampled indices
    """
    sampled = normal_sampled_idx(random_state, n, bayes)
    counts  = np.bincount(sampled, minlength=n)
    return np.arange(n, dtype=int)[counts==0]


def _parallel_fit_classifier(tree, X, y, n, tree_idx, n_estimators, bootstrap,
                             bayes, verbose, random_state, class_weight=None,
                             min_dist_p=None):
    """Utility function for building trees in parallel

    Note: This function can't go locally in a class, because joblib complains
          that it cannot pickle it when placed there

    Parameters
    ----------
    tree : CITreeClassifier
        Instantiated conditional inference tree

    X : 2d array-like
        Array of features

    y : 1d array-like
        Array of labels

    n : int
        Number of samples

    tree_idx : int
        Index of tree in forest

    n_estimators : int
        Number of total estimators

    bootstrap : bool
        Whether to perform bootstrap sampling

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    verbose : bool or int
        Controls verbosity of training process

    random_state : int
        Sets seed for random number generator

    class_weight : str
        Type of sampling during bootstrap, None for regular bootstrapping,
        'balanced' for balanced bootstrap sampling, and 'stratify' for
        stratified bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    tree : CITreeClassifier
        Fitted conditional inference tree
    """
    # Print status if conditions met
    if verbose and n_estimators >= 10:
        denom = n_estimators if verbose > 1 else 10
        if (tree_idx+1) % int(n_estimators/denom) == 0:
            logger("tree", "Building tree %d/%d" % (tree_idx+1, n_estimators))

    # Bootstrap sample if specified
    if bootstrap:
        random_state = random_state*(tree_idx+1)
        if class_weight == 'balanced':
            idx = np.concatenate(
                balanced_sampled_idx(random_state, y, bayes, min_dist_p)
                )
        elif class_weight == 'stratify':
            idx = np.concatenate(
                stratify_sampled_idx(random_state, y, bayes)
                )
        else:
            idx = normal_sampled_idx(random_state, n, bayes)

        # Note: We need to pass the classes in the case of the bootstrap
        # because not all classes may be sampled and when it comes to prediction,
        # the tree models learns a different number of classes across different
        # bootstrap samples
        tree.fit(X[idx], y[idx], np.unique(y))
    else:
        tree.fit(X, y)
    
    return tree


def _parallel_fit_regressor(tree, X, y, n, tree_idx, n_estimators, bootstrap,
                            bayes, verbose, random_state):
    """Utility function for building trees in parallel

    Note: This function can't go locally in a class, because joblib complains
          that it cannot pickle it when placed there

    Parameters
    ----------
    tree : CITreeRegressor
        Instantiated conditional inference tree

    X : 2d array-like
        Array of features

    y : 1d array-like
        Array of labels

    n : int
        Number of samples

    tree_idx : int
        Index of tree in forest

    n_estimators : int
        Number of total estimators

    bootstrap : bool
        Whether to perform bootstrap sampling

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    verbose : bool or int
        Controls verbosity of training process

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    tree : CITreeRegressor
        Fitted conditional inference tree
    """
    # Print status if conditions met
    if verbose and n_estimators >= 10:
        denom = n_estimators if verbose > 1 else 10
        if (tree_idx+1) % int(n_estimators/denom) == 0:
            logger("tree", "Building tree %d/%d" % (tree_idx+1, n_estimators))

    # Bootstrap sample if specified
    if bootstrap:
        random_state = random_state*(tree_idx+1)
        idx          = normal_sampled_idx(random_state, n, bayes)

        # Train
        tree.fit(X[idx], y[idx])
    else:
        tree.fit(X, y)
    
    return tree


def _accumulate_prediction(predict, X, out, lock):
    """Utility function to aggregate predictions in parallel

    Parameters
    ----------
    predict : function handle
        Alias to prediction method of class

    X : 2d array-like
        Array of features

    out : 1d or 2d array-like
        Array of labels

    lock : threading lock
        A lock that controls worker access to data structures for aggregating
        predictions

    Returns
    -------
    None
    """
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)): out[i] += prediction[i]


class CIForestClassifier(BaseEstimator, ClassifierMixin):
    """Conditional forest classifier

    Parameters
    ----------
    min_samples_split : int
        Minimum samples required for a split

    alpha : float
        Threshold value for selecting feature with permutation tests. Smaller
        values correspond to shallower trees

    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    max_depth : int
        Maximum depth to grow tree

    max_feats : str or int
        Maximum feats to select at each split. String arguments include 'sqrt',
        'log', and 'all'

    n_permutations : int
        Number of permutations during feature selection

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    muting : bool
        Whether to perform variable muting

    verbose : bool or int
        Controls verbosity of training and testing

    bootstrap : bool
        Whether to perform bootstrap sampling for each tree

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    class_weight : str
        Type of sampling during bootstrap, None for regular bootstrapping,
        'balanced' for balanced bootstrap sampling, and 'stratify' for
        stratified bootstrap sampling

    n_jobs : int
        Number of jobs for permutation testing

    random_state : int
        Sets seed for random number generator
    """
    def __init__(self, min_samples_split=2, alpha=.05, selector='mc', max_depth=-1,
                 n_estimators=100, max_feats='sqrt', n_permutations=100,
                 early_stopping=True, muting=True, verbose=0, bootstrap=True,
                 bayes=True, class_weight='balanced', n_jobs=-1, random_state=None):

        # Error checking
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be in (0, 1]" % alpha)
        if selector not in ['mc', 'mi', 'hybrid']:
            raise ValueError("%s not a valid selector, valid selectors are " \
                             "mc, mi, and hybrid")
        if n_permutations < 0:
            raise ValueError("n_permutations (%s) should be > 0" % \
                             str(n_permutations))
        if not isinstance(max_feats, int) and max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))
        if n_estimators < 0:
            raise ValueError("n_estimators (%s) must be > 0" % \
                             str(n_estimators))

        # Only for classifier model
        if class_weight not in [None, 'balanced', 'stratify']:
            raise ValueError("%s not a valid argument for class_weight" % \
                             str(class_weight))

        # Placeholder variable for regression model (not applicable)
        if class_weight is None: self.min_class_p = None

        # Define attributes
        self.alpha             = float(alpha)
        self.selector          = selector
        self.min_samples_split = max(1, min_samples_split)
        self.n_permutations    = int(n_permutations)
        if max_depth == -1:
            self.max_depth = max_depth
        else:
            self.max_depth = int(max(1, max_depth))

        self.n_estimators   = int(max(1, n_estimators))
        self.max_feats      = max_feats
        self.bootstrap      = bootstrap
        self.early_stopping = early_stopping
        self.muting         = muting
        self.n_jobs         = n_jobs
        self.verbose        = verbose
        self.class_weight   = class_weight
        self.bayes          = bayes

        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)

        # Package params for calling CITreeClassifier
        self.params = {
            'alpha'             : self.alpha,
            'selector'          : self.selector,
            'min_samples_split' : self.min_samples_split,
            'n_permutations'    : self.n_permutations,
            'max_feats'         : self.max_feats,
            'early_stopping'    : self.early_stopping,
            'muting'            : self.muting,
            'verbose'           : 0,
            'n_jobs'            : 1,
            'random_state'      : None,
            }


    def fit(self, X, y):
        """Fit conditional forest classifier

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        Returns
        -------
        self : CIForestClassifier
            Instance of CIForestClassifier
        """
        self.labels_    = np.unique(y)
        self.n_classes_ = len(self.labels_)

        if self.verbose:
            logger("tree", "Training ensemble with %d trees on %d samples" % \
                    (self.n_estimators, X.shape[0]))

        # Instantiate base tree models
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.params['random_state'] = self.random_state*(i+1)
            self.estimators_.append(CITreeClassifier(**self.params))

        # Define class distribution
        self.class_dist_p = np.array([
                np.mean(y==label) for label in np.unique(y)
            ])

        # Train models
        n = X.shape[0]
        self.estimators_ = \
            Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_parallel_fit_classifier)(
                self.estimators_[i], X, y, n, i, self.n_estimators,
                self.bootstrap, self.bayes, self.verbose, self.random_state,
                self.class_weight, np.min(self.class_dist_p)
                )
            for i in range(self.n_estimators)
            )

        # Accumulate feature importances (mean decrease impurity)
        self.feature_importances_ = np.sum([
                tree.feature_importances_ for tree in self.estimators_],
                axis=0
            )
        sum_fi = np.sum(self.feature_importances_)
        if sum_fi > 0: self.feature_importances_ /= sum_fi

        return self


    def predict_proba(self, X):
        """Predicts class probabilities for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        class_probs : 2d array-like
            Array of predicted class probabilities
        """
        if self.verbose:
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        # Parallel prediction
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        lock      = threading.Lock()
        Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        # Normalize probabilities
        all_proba /= len(self.estimators_)
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


    def predict(self, X):
        """Predicts class labels for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        y : 1d array-like
            Array of predicted classes
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)


class CIForestRegressor(BaseEstimator, RegressorMixin):
    """Conditional forest regressor

    Parameters
    ----------
    min_samples_split : int
        Minimum samples required for a split

    alpha : float
        Threshold value for selecting feature with permutation tests. Smaller
        values correspond to shallower trees

    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    max_depth : int
        Maximum depth to grow tree

    max_feats : str or int
        Maximum feats to select at each split. String arguments include 'sqrt',
        'log', and 'all'

    n_permutations : int
        Number of permutations during feature selection

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    muting : bool
        Whether to perform variable muting

    verbose : bool or int
        Controls verbosity of training and testing

    bootstrap : bool
        Whether to perform bootstrap sampling for each tree

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    n_jobs : int
        Number of jobs for permutation testing

    random_state : int
        Sets seed for random number generator
    """
    def __init__(self, min_samples_split=2, alpha=.01, selector='pearson', max_depth=-1,
                 n_estimators=100, max_feats='sqrt', n_permutations=100,
                 early_stopping=True, muting=True, verbose=0, bootstrap=True,
                 bayes=True, n_jobs=-1, random_state=None):

        # Error checking
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be in (0, 1]" % alpha)

        if selector not in ['pearson', 'distance', 'rdc', 'hybrid']:
            raise ValueError("%s not a valid selector, valid selectors are " \
                             "pearson, distance, rdc, hybrid")

        if n_permutations < 0:
            raise ValueError("n_permutations (%s) should be > 0" % \
                             str(n_permutations))
        if not isinstance(max_feats, int) and max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))
        if n_estimators < 0:
            raise ValueError("n_estimators (%s) must be > 0" % \
                             str(n_estimators))

        # Define attributes
        self.alpha             = float(alpha)
        self.selector          = selector
        self.min_samples_split = max(1, min_samples_split)
        self.n_permutations    = int(n_permutations)
        if max_depth == -1:
            self.max_depth = max_depth
        else:
            self.max_depth = int(max(1, max_depth))
        self.n_estimators   = int(max(1, n_estimators))
        self.max_feats      = max_feats
        self.bootstrap      = bootstrap
        self.early_stopping = early_stopping
        self.muting         = muting
        self.n_jobs         = n_jobs
        self.verbose        = verbose
        self.bayes          = bayes

        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)

        # Package params for calling CITreeRegressor
        self.params = {
            'alpha'             : self.alpha,
            'selector'          : self.selector,
            'min_samples_split' : self.min_samples_split,
            'n_permutations'    : self.n_permutations,
            'max_feats'         : self.max_feats,
            'early_stopping'    : self.early_stopping,
            'muting'            : muting,
            'verbose'           : 0,
            'n_jobs'            : 1,
            'random_state'      : None,
            }


    def fit(self, X, y):
        """Fit conditional forest regressor

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        Returns
        -------
        self : CIForestRegressor
            Instance of CIForestRegressor
        """
        if self.verbose:
            logger("tree", "Training ensemble with %d trees on %d samples" % \
                    (self.n_estimators, X.shape[0]))

        # Instantiate base tree models
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.params['random_state'] = self.random_state*(i+1)
            self.estimators_.append(CITreeRegressor(**self.params))

        # Train models
        n = X.shape[0]
        self.estimators_ = \
            Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_parallel_fit_regressor)(
                self.estimators_[i], X, y, n, i, self.n_estimators,
                self.bootstrap, self.bayes, self.verbose, self.random_state
                )
            for i in range(self.n_estimators)
            )

        # Accumulate feature importances (mean decrease impurity)
        self.feature_importances_ = np.sum([
                tree.feature_importances_ for tree in self.estimators_],
                axis=0
            )
        sum_fi = np.sum(self.feature_importances_)
        if sum_fi > 0: self.feature_importances_ /= sum_fi

        return self


    def predict(self, X):
        """Predicts labels for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        labels : 1d array-like
            Array of predicted labels
        """
        if self.verbose:
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        # Parallel prediction
        results = np.zeros(X.shape[0], dtype=np.float64)
        lock    = threading.Lock()
        Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_accumulate_prediction)(e.predict, X, results, lock)
            for e in self.estimators_)

        # Normalize predictions
        results /= len(self.estimators_)
        if len(results) == 1:
            return results[0]
        else:
            return results