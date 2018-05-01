from __future__ import division, print_function

from joblib import delayed, Parallel
import numbers
import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
import threading
import warnings
warnings.simplefilter('ignore')

# Package imports
from permutation import (permutation_test_dcor, permutation_test_pcor, 
                         permutation_test_dcor_parallel)
from scorers import c_dcor, gini_index, pcor, py_dcor
from utils import logger


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

    Returns
    -------
    None
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

    NOTE: Classification and regression trees are derived from this parent class
    
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

    selector : str
        Type of correlation for feature selection, 'pearson' for Pearson
        correlation, 'distance' for distance correlation, or 'hybrid' for
        a dynamic selection of Pearson/distance correlations

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    verbose : bool or int
        Controls verbosity of training and testing

    n_jobs : int
        Number of jobs for permutation testing
    
    Returns
    -------
    None
    """
    def __init__(self, min_samples_split=2, alpha=.05, max_depth=-1,
                 max_feats=-1, n_permutations=500, selector='pearson', 
                 early_stopping=False, verbose=0, n_jobs=-1):

        # Error checking
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be between (0, 1)" % alpha)
        if n_permutations < 0:
            raise ValueError("n_permutations (%d) should be > 0" % \
                             n_permutations)
        if selector not in ['distance', 'pearson', 'hybrid']:
            raise ValueError("selector (%s) should be distance, pearson, or " \
                             "hybrid" % selector)
        if max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))

        # Define attributes
        self.alpha             = float(alpha)
        self.min_samples_split = max(1, int(min_samples_split))
        self.n_permutations    = int(n_permutations)
        self.selector          = selector
        self.max_feats         = max_feats
        self.early_stopping    = early_stopping
        self.verbose           = verbose
        self.n_jobs            = n_jobs
        self.root              = None

        if max_depth == -1: 
            self.max_depth = np.inf
        else:
            self.max_depth = int(max(1, max_depth))


    def _selector_pcor(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a Pearson correlation
        
        Parameters
        ----------
        
        Returns
        -------
        """
        best_col, best_pval = None, np.inf

        # Iterate over columns
        for col in col_idx:
            pval = permutation_test_pcor(x=X[:, col], 
                                         y=y, 
                                         agg=np.concatenate([X[:, col], y]), 
                                         B=self.n_permutations)
            if pval < best_pval: 
                best_col, best_pval = col, pval
                
                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _selector_dcor(self, X, y, n, col_idx):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        best_col, best_pval = None, np.inf

        # Use Numba serial version for 'smaller' samples
        if n < 500:
            for col in col_idx:
                pval = permutation_test_dcor(x=X[:, col], 
                                             y=y, 
                                             agg=np.concatenate([X[:, col], y]), 
                                             n=n,
                                             B=self.n_permutations)
                if pval < best_pval: 
                    best_col, best_pval = col, pval

                    # If early stopping
                    if self.early_stopping and best_pval < self.alpha:
                        if self.verbose: logger("tree", "Early stopping")
                        return best_col, best_pval

        else:
            # Use parallel C version for 'larger' samples
            for col in col_idx:
                pval = permutation_test_dcor_parallel(
                                             x=X[:, col], 
                                             y=y, 
                                             agg=np.concatenate([X[:, col], y]), 
                                             n=n,
                                             n_jobs=self.n_jobs,
                                             B=self.n_permutations)
                if pval < best_pval: 
                    best_col, best_pval = col, pval

                    # If early stopping
                    if self.early_stopping and best_pval < self.alpha:
                        if self.verbose: logger("tree", "Early stopping")
                        return best_col, best_pval

        return best_col, best_pval


    def _selector_cor_hybrid(self, X, y, n, col_idx):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        best_col, best_pval = None, np.inf

        # Iterate over columns
        for col in col_idx:

            # First calculate correlation to determine which one is larger
            pearson  = abs(pcor(X[:, col], y))
            distance = py_dcor(X[:, col], y, n) if n < 500 else \
                       c_dcor(X[:, col], y, n)
            
            # Calculate permutation test based on correlation values
            if pearson >= distance:
                pval = permutation_test_pcor(x=X[:, col], 
                                             y=y, 
                                             agg=np.concatenate([X[:, col], y]), 
                                             B=self.n_permutations)
                if pval < best_pval: 
                    best_col, best_pval = col, pval
                    
                    # If early stopping
                    if self.early_stopping and best_pval < self.alpha:
                        if self.verbose: logger("tree", "Early stopping")
                        return best_col, best_pval

            else:
                # Use Numba serial version for 'smaller' samples
                if n < 500:
                    pval = permutation_test_dcor(x=X[:, col], 
                                                 y=y, 
                                                 agg=np.concatenate([X[:, col], y]), 
                                                 n=n,
                                                 B=self.n_permutations)
                    if pval < best_pval: 
                        best_col, best_pval = col, pval

                        # If early stopping
                        if self.early_stopping and best_pval < self.alpha:
                            if self.verbose: logger("tree", "Early stopping")
                            return best_col, best_pval

                else:
                    # Use parallel C version for 'larger' samples
                    pval = permutation_test_dcor_parallel(
                                                 x=X[:, col], 
                                                 y=y, 
                                                 agg=np.concatenate([X[:, col], y]), 
                                                 n=n,
                                                 n_jobs=self.n_jobs,
                                                 B=self.n_permutations)
                    if pval < best_pval: 
                        best_col, best_pval = col, pval

                        # If early stopping
                        if self.early_stopping and best_pval < self.alpha:
                            if self.verbose: logger("tree", "Early stopping")
                            return best_col, best_pval

        return best_col, best_pval


    def _selector(self, X, y, n, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a correlation function
        
        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        n : int
            Number of samples

        col_idx : list
            Columns of X to examine for feature selection
        
        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is 
            enabled then this may not be the absolute best column

        best_pval : float
            Best p-value corresponding to best column from feature selection
        """        
        if self.verbose > 1: 
            logger("selector", "Testing %d features" % len(col_idx))

        # Pearson correlation
        if self.selector == 'pearson':
            return self._selector_pcor(X, y, col_idx)

        # Distance correlation 
        elif self.selector == 'distance':
            return self._selector_dcor(X, y, n, col_idx)

        # Hybrid correlation
        else:
            s = int(n*(n-1)/2.)
            return self._selector_cor_hybrid(X, y, n, col_idx)


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
        n_samples, n_features = X.shape

        # Check for stopping criteria
        if n_samples >= self.min_samples_split and depth < self.max_depth:

            # Find column with strongest association with outcome
            col_idx       = np.random.choice(np.arange(n_features, dtype=int),
                                             size=self.max_feats, replace=False)
            col, col_pval = self._selector(X, y, n_samples, col_idx)
            if col_pval < self.alpha:

                # Find best split among selected variable
                impurity, threshold, left, right = self._splitter(X, y, n_samples, col)
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
        self.root = self._build_tree(X, y)
        sum_fi    = np.sum(self.feature_importances_)
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

        # Choose the feature that we will test

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
        
        Returns
        -------
        None
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
    min_samples_split : int
        ADD

    alpha : float
        ADD

    Returns
    -------
    """
    def __init__(self,
                 min_samples_split=2, 
                 alpha=.05, 
                 max_depth=-1,
                 max_feats=-1, 
                 n_permutations=100, 
                 selector='pearson', 
                 early_stopping=False, 
                 verbose=0, 
                 n_jobs=-1):

        super(CITreeClassifier, self).__init__(
                    min_samples_split=min_samples_split, 
                    alpha=alpha, 
                    max_depth=max_depth,
                    max_feats=max_feats, 
                    n_permutations=n_permutations, 
                    selector=selector, 
                    early_stopping=early_stopping, 
                    verbose=verbose, 
                    n_jobs=n_jobs)


    def _splitter(self, X, y, n, col):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose > 1: 
            logger("splitter", "Testing splits on feature %d" % col)
        
        best_impurity, best_threshold = 0.0, None
        left, right                   = None, None
        node_impurity                 = gini_index(y, self.labels_)
        for threshold in np.unique(X[:, col]):

            # Make split
            idx              = np.where(X[:, col] <= threshold, 1, 0)
            X_left, y_left   = X[idx==1], y[idx==1]
            X_right, y_right = X[idx==0], y[idx==0]
            n_left, n_right  = len(y_left), len(y_right)

            # Skip small splits
            if n_left < self.min_samples_split or n_right < self.min_samples_split:
                continue

            # Children impurity
            left_impurity  = gini_index(y_left, self.labels_)*(n_left/float(n))
            right_impurity = gini_index(y_right, self.labels_)*(n_right/float(n))

            # Impurity score and evaluate
            impurity = node_impurity - (left_impurity + right_impurity)

            if impurity > best_impurity:
                best_impurity  = impurity
                best_threshold = threshold
                left, right    = (X_left, y_left), (X_right, y_right)

        # Update feature importance
        self.feature_importances_[col] += best_impurity
        return best_impurity, best_threshold, left, right


    def _estimate_proba(self, y):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return np.array([np.mean(y == label) for label in self.labels_])


    def fit(self, X, y):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.feature_importances_ = np.zeros(X.shape[1])
        self.labels_              = np.unique(y)
        self.n_classes_           = len(self.labels_)
        self.node_estimate        = self._estimate_proba
        super(CITreeClassifier, self).fit(X, y)
        return self


    def predict_proba(self, X):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose: 
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        return np.array([self.predict_label(sample) for sample in X])


    def predict(self, X):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1) + self.labels_.min()


#####################
"""ENSEMBLE MODELS"""
#####################

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def sampled_idx(random_state, n_samples):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    random_instance = check_random_state(random_state)
    return random_instance.randint(0, n_samples, n_samples)


def unsampled_idx(random_state, n_samples):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    sample_idx     = sampled_idx(random_state, n_samples)
    sample_counts  = np.bincount(sample_idx, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    idx_range      = np.arange(n_samples, dtype=int)
    return idx_range[unsampled_mask]


def _parallel_fit(tree, X, y, n, tree_idx, n_estimators, bootstrap,
                  verbose, random_state):
    """This is a utility function for joblib's Parallel. It can't go locally in
    class, because joblib complains that it cannot pickle it when placed there.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Print status if conditions met
    if verbose and n_estimators >= 10:
        denom = n_estimators if verbose > 1 else 10
        if (tree_idx+1) % int(n_estimators/denom) == 0:
            logger("tree", "Building tree %d/%d" % (tree_idx+1, n_estimators))

    # Bootstrap sample if specified
    if bootstrap:
        idx = sampled_idx(random_state*(tree_idx+1), n)
        tree.fit(X[idx], y[idx])
    else:
        tree.fit(X, y)


def _accumulate_prediction(predict, X, out, lock):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)): out[i] += prediction[i]


class CIForestBase(object):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    def __init__(self, min_samples_split=2, alpha=.05, max_depth=-1,
                 n_estimators=100, max_feats='sqrt', n_permutations=250, 
                 selector='pearson', early_stopping=True, verbose=0, 
                 bootstrap=True, n_jobs=-1, random_state=None):

        # Error checking
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be between (0, 1)" % alpha)
        if n_permutations < 0:
            raise ValueError("n_permutations (%s) should be > 0" % \
                             str(n_permutations))
        if selector not in ['distance', 'pearson', 'hybrid']:
            raise ValueError("selector (%s) should be distance, pearson, or " \
                             "hybrid" % selector)
        if max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))
        if n_estimators < 0:
            raise ValueError("n_estimators (%s) must be > 0" % \
                             str(n_estimators))

        # Define attributes
        self.alpha             = float(alpha)
        self.min_samples_split = max(1, min_samples_split)
        self.n_permutations    = int(n_permutations)
        if max_depth == -1: 
            self.max_depth = max_depth
        else:
            self.max_depth = int(max(1, max_depth))

        self.n_estimators   = int(max(1, n_estimators))
        self.selector       = selector
        self.max_feats      = max_feats
        self.bootstrap      = bootstrap
        self.early_stopping = early_stopping
        self.n_jobs         = n_jobs
        self.verbose        = verbose


        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)

        # Package params for calling CITree{Classifier/Regressor}
        self.params = {
            'alpha': self.alpha, 
            'min_samples_split': self.min_samples_split,
            'n_permutations': self.n_permutations,
            'selector': self.selector,
            'max_feats': self.max_feats,
            'early_stopping': self.early_stopping,
            'verbose': 0,
            'n_jobs': 1
            }


    def fit(self, X, y):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            logger("tree", "Training ensemble with %d trees on %d samples" % \
                    (self.n_estimators, X.shape[0]))

        # Class information
        self.labels_    = np.unique(y)
        self.n_classes_ = len(self.labels_)

        # Instantiate base tree models
        self.estimators_ = \
            [self.Tree(**self.params) for _ in range(self.n_estimators)]

        # Train models
        n = X.shape[0]
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_parallel_fit)(
                self.estimators_[i], X, y, n, i, self.n_estimators, 
                self.bootstrap, self.verbose, self.random_state
                ) for i in range(self.n_estimators)
            )

        # Accumulate feature importances
        self.feature_importances_ = np.sum([
                tree.feature_importances_ for tree in self.estimators_],
                axis=0
            )
        sum_fi = np.sum(self.feature_importances_)
        if sum_fi > 0:
            self.feature_importances_ /= sum_fi

        return self


    def predict(self, X):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        raise NotImplementedError("ADD")


class CIForestClassifier(CIForestBase, BaseEstimator, ClassifierMixin):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    def __init__(self, 
                 min_samples_split=2, 
                 alpha=.05, 
                 max_depth=-1,
                 n_estimators=100, 
                 max_feats='sqrt', 
                 n_permutations=250, 
                 selector='pearson', 
                 early_stopping=False, 
                 verbose=0, 
                 bootstrap=True, 
                 n_jobs=-1, 
                 random_state=None):

        super(CIForestClassifier, self).__init__(
            min_samples_split=min_samples_split, 
            alpha=alpha, 
            max_depth=max_depth,
            n_estimators=n_estimators, 
            max_feats=max_feats, 
            n_permutations=n_permutations, 
            selector=selector, 
            early_stopping=early_stopping, 
            verbose=verbose, 
            bootstrap=bootstrap, 
            n_jobs=n_jobs, 
            random_state=random_state)


    def fit(self, X, y):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Alias to tree model used in parent class
        self.Tree = CITreeClassifier
        super(CIForestClassifier, self).fit(X, y)
        return self


    def predict_proba(self, X):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
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
        for proba in all_proba: proba /= len(self.estimators_)
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


    def predict(self, X):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1) + self.labels_.min()


if __name__ == '__main__':
    X   = np.random.normal(0, 1, (1000, 15))
    # y   = 1/(1+np.exp(-.9*X[:, 0]))
    # y   = np.random.binomial(1, y, 1000)
    y = np.zeros(1000)
    y[X[:,0]>0] = 1
    import pdb; pdb.set_trace()
    clf = CIForestClassifier()
    clf.fit(X, y)
    print(np.mean(clf.predict(X) == y))