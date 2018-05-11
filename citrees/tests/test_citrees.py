from __future__ import absolute_import, division, print_function

import numpy as np
from os.path import abspath, dirname
import sys
import unittest

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from citrees import (balanced_sampled_idx, balanced_unsampled_idx, 
                     normal_sampled_idx, normal_unsampled_idx,
                     stratify_sampled_idx, stratify_unsampled_idx, 
                     CIForestClassifier, CITreeClassifier)


class TestClassificationTrees(unittest.TestCase):

    def setUp(self):
        """Generate toy data"""
        
        self.n       = 500
        self.X       = np.arange(self.n).reshape(-1, 1)
        self.y       = np.ones(self.n)
        mask         = (self.X < int(self.n/3)).ravel()
        self.y[mask] = 0 

        # Indices for classes
        self.y0_idx = np.where(self.y==0)[0]
        self.y1_idx = np.where(self.y==1)[0]


    def test_CITreeClassifier(self):
        """Test for CITreeClassifier"""

        # Train simple model and check for 100% accuracy
        clf = CITreeClassifier().fit(self.X, self.y)
        acc = clf.score(self.X, self.y)
        msg = "Accuracy for CITreeClassifier (%.2f) should be 1.0 for " \
               "simple toy data" % acc
        self.assertEqual(acc, 1.0, msg=msg)


    def test_CIForestClassifier(self):
        """Test for CIForestClassifier"""

        # Train simple model and check for 100% accuracy
        clf = CIForestClassifier().fit(self.X, self.y)
        acc = clf.score(self.X, self.y)
        msg = "Accuracy for CIForestClassifier (%.2f) should be 1.0 for " \
               "simple toy data" % acc
        self.assertAlmostEqual(acc, 1.0, delta=.05, msg=msg)


    def test_stratify_sampling(self):
        """Test for stratified sampling in classification"""

        # REGULAR BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = stratify_sampled_idx(random_state=1718, y=self.y, bayes=False)
        unsampled = stratify_unsampled_idx(random_state=1718, y=self.y, bayes=False)
        
        # Get class counts
        class_counts = np.bincount(self.y.astype(int))

        # Check number of unique samples
        all_sampled         = np.concatenate(sampled)
        perc_unique_sampled = len(np.unique(all_sampled))/float(self.n)

        msg = "Bootstrap: Fraction of unique bootstrap samples (%.2f) " \
              "should be close to .632" % perc_unique_sampled
        self.assertAlmostEqual(perc_unique_sampled, .632, delta=.05, msg=msg)

        # Make sure both classes are sampled according to imbalanced distribution
        msg = "Bootstrap: Number of stratified samples for class 0 (%d) does " \
              "not match the number of original samples in class 0 (%d)" % \
              (len(sampled[0]), class_counts[0])
        self.assertEqual(len(sampled[0]), class_counts[0], msg=msg)

        msg = "Bootstrap: Number of stratified samples for class 1 (%d) does " \
              "not match the number of original samples in class 1 (%d)" % \
              (len(sampled[1]), class_counts[1])
        self.assertEqual(len(sampled[1]), class_counts[1], msg=msg)

        # Make sure sampled indices are part of correct classes
        msg = "Bootstrap: Stratified sampled indices for class 0 not all " \
               "members of class 0"
        if not set(sampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bootstrap: Stratified sampled indices for class 1 not all " \
              "members of class 1"
        if not set(sampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)
    
        # Make sure unsampled indices are part of correct classes
        msg = "Bootstrap: Stratified unsampled indices for class 0 not all " \
               "members of class 0"
        if not set(unsampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bootstrap: Stratified unsampled indices for class 1 not all " \
              "members of class 1"
        if not set(unsampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bootstrap: Stratified sampled indices for class 0 should not "\
              "be indices in the unsampled indices for class 0"
        if set(sampled[0]).issubset(set(unsampled[0])): 
            raise AssertionError(msg)

        msg = "Bootstrap: Stratified sampled indices for class 1 should not " \
               "be indices in the unsampled indices for class 1"
        if set(sampled[1]).issubset(set(unsampled[1])): 
            raise AssertionError(msg)


        # BAYES BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = stratify_sampled_idx(random_state=1718, y=self.y, bayes=True)
        unsampled = stratify_unsampled_idx(random_state=1718, y=self.y, bayes=True)

        # Check number of unique samples
        all_sampled         = np.concatenate(sampled)
        perc_unique_sampled = len(np.unique(all_sampled))/float(self.n)

        msg = "Bayes: Fraction of unique bootstrap samples (%.2f) " \
              "should be close to .5" % perc_unique_sampled
        self.assertAlmostEqual(perc_unique_sampled, .5, delta=.05, msg=msg)

        # Make sure both classes are sampled according to imbalanced distribution
        msg = "Bayes: Number of stratified samples for class 0 (%d) does " \
              "not match the number of original samples in class 0 (%d)" % \
              (len(sampled[0]), class_counts[0])
        self.assertEqual(len(sampled[0]), class_counts[0], msg=msg)

        msg = "Bayes: Number of stratified samples for class 1 (%d) does " \
              "not match the number of original samples in class 1 (%d)" % \
              (len(sampled[1]), class_counts[1])
        self.assertEqual(len(sampled[1]), class_counts[1], msg=msg)

        # Make sure sampled indices are part of correct classes
        msg = "Bayes: Stratified sampled indices for class 0 not all " \
               "members of class 0"
        if not set(sampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bayes: Stratified sampled indices for class 1 not all " \
              "members of class 1"
        if not set(sampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)
    
        # Make sure unsampled indices are part of correct classes
        msg = "Bayes: Stratified unsampled indices for class 0 not all " \
               "members of class 0"
        if not set(unsampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bayes: Stratified unsampled indices for class 1 not all " \
              "members of class 1"
        if not set(unsampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bayes: Stratified sampled indices for class 0 should not "\
              "be indices in the unsampled indices for class 0"
        if set(sampled[0]).issubset(set(unsampled[0])): 
            raise AssertionError(msg)

        msg = "Bayes: Stratified sampled indices for class 1 should not " \
               "be indices in the unsampled indices for class 1"
        if set(sampled[1]).issubset(set(unsampled[1])): 
            raise AssertionError(msg)


    def test_balanced_sampling(self):
        """Test for balanced sampling in classification"""

        # Get class counts
        class_counts = np.bincount(self.y.astype(int))
        min_class_p  = np.min(class_counts/float(self.n))

        # REGULAR BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = balanced_sampled_idx(random_state=1718, y=self.y, 
                                         bayes=False, min_class_p=min_class_p)
        unsampled = balanced_unsampled_idx(random_state=1718, y=self.y, 
                                           bayes=False, min_class_p=min_class_p)

        # Make sure both classes are sampled according to size of minority class
        msg = "Bootstrap: Number of balanced samples for class 0 (%d) does " \
              "not match the number of samples in the minority class (%d)" % \
              (len(sampled[0]), np.min(class_counts))
        self.assertEqual(len(sampled[0]), np.min(class_counts), msg=msg)

        msg = "Bootstrap: Number of balanced samples for class 1 (%d) does " \
              "not match the number of samples in the minority class (%d)" % \
              (len(sampled[0]), np.min(class_counts))
        self.assertEqual(len(sampled[1]), np.min(class_counts), msg=msg)

        # Make sure sampled indices are part of correct classes
        msg = "Bootstrap: Balanced sampled indices for class 0 not all " \
               "members of class 0"
        if not set(sampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bootstrap: Balanced sampled indices for class 1 not all " \
              "members of class 1"
        if not set(sampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)
    
        # Make sure unsampled indices are part of correct classes
        msg = "Bootstrap: Balanced unsampled indices for class 0 not all " \
               "members of class 0"
        if not set(unsampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bootstrap: Balanced unsampled indices for class 1 not all " \
              "members of class 1"
        if not set(unsampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bootstrap: Balanced sampled indices for class 0 should not "\
              "be indices in the unsampled indices for class 0"
        if set(sampled[0]).issubset(set(unsampled[0])): 
            raise AssertionError(msg)

        msg = "Bootstrap: Balanced sampled indices for class 1 should not " \
               "be indices in the unsampled indices for class 1"
        if set(sampled[1]).issubset(set(unsampled[1])): 
            raise AssertionError(msg)


        # BAYES BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = balanced_sampled_idx(random_state=1718, y=self.y, 
                                         bayes=True, min_class_p=min_class_p)
        unsampled = balanced_unsampled_idx(random_state=1718, y=self.y, 
                                           bayes=True, min_class_p=min_class_p)

        # Make sure both classes are sampled according to size of minority class
        msg = "Bayes: Number of balanced samples for class 0 (%d) does " \
              "not match the number of samples in the minority class (%d)" % \
              (len(sampled[0]), np.min(class_counts))
        self.assertEqual(len(sampled[0]), np.min(class_counts), msg=msg)

        msg = "Bayes: Number of balanced samples for class 1 (%d) does " \
              "not match the number of samples in the minority class (%d)" % \
              (len(sampled[0]), np.min(class_counts))
        self.assertEqual(len(sampled[1]), np.min(class_counts), msg=msg)

        # Make sure sampled indices are part of correct classes
        msg = "Bayes: Balanced sampled indices for class 0 not all " \
               "members of class 0"
        if not set(sampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bayes: Balanced sampled indices for class 1 not all " \
              "members of class 1"
        if not set(sampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)
    
        # Make sure unsampled indices are part of correct classes
        msg = "Bayes: Balanced unsampled indices for class 0 not all " \
               "members of class 0"
        if not set(unsampled[0]).issubset(set(self.y0_idx)): 
            raise AssertionError(msg)

        msg = "Bayes: Balanced unsampled indices for class 1 not all " \
              "members of class 1"
        if not set(unsampled[1]).issubset(set(self.y1_idx)): 
            raise AssertionError(msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bayes: Balanced sampled indices for class 0 should not "\
              "be indices in the unsampled indices for class 0"
        if set(sampled[0]).issubset(set(unsampled[0])): 
            raise AssertionError(msg)

        msg = "Bayes: Balanced sampled indices for class 1 should not " \
               "be indices in the unsampled indices for class 1"
        if set(sampled[1]).issubset(set(unsampled[1])): 
            raise AssertionError(msg)


    def test_normal_sampling(self):
        """Test for normal sampling in classification"""

        # REGULAR BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = normal_sampled_idx(random_state=1718, n=self.n, bayes=False)
        unsampled = normal_unsampled_idx(random_state=1718, n=self.n, bayes=False)

        # Check number of unique samples
        perc_unique_sampled = len(np.unique(sampled))/float(self.n)

        msg = "Bootstrap: Fraction of unique bootstrap samples (%.2f) " \
              "should be close to .632" % perc_unique_sampled
        self.assertAlmostEqual(perc_unique_sampled, .632, delta=.05, msg=msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bootstrap: Sampled indices should not be indices in the " \
              "unsampled indices"
        if set(sampled).issubset(unsampled): 
            raise AssertionError(msg)


        # BAYES BOOTSTRAPPING #

        # Get sampled and unsampled indices
        sampled   = normal_sampled_idx(random_state=1718, n=self.n, bayes=True)
        unsampled = normal_unsampled_idx(random_state=1718, n=self.n, bayes=True)

        # Check number of unique samples
        perc_unique_sampled = len(np.unique(sampled))/float(self.n)

        msg = "Bayes: Fraction of unique bootstrap samples (%.2f) " \
              "should be close to .5" % perc_unique_sampled
        self.assertAlmostEqual(perc_unique_sampled, .5, delta=.05, msg=msg)

        # Make sure unsampled indices are not part of sampled indices
        msg = "Bayes: Sampled indices should not be indices in the " \
              "unsampled indices"
        if set(sampled).issubset(unsampled): 
            raise AssertionError(msg)


if __name__ == '__main__':
    unittest.main()