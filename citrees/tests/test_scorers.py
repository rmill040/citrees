from __future__ import absolute_import, division, print_function

import numpy as np
from os.path import abspath, dirname
import sys
import unittest

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from scorers import *


class TestScorers(unittest.TestCase):

    def setUp(self):
        """Generate toy data and calculate Pearson correlation for comparison"""
        
        self.n       = 5000
        self.x       = np.random.normal(0, 1, self.n)
        self.y       = .2*self.x + np.random.normal(0, 1, self.n)
        self.weights = np.ones(self.n)/float(self.n)

        # Calculate pearson correlation and use as ground truth
        self.pearson_r = np.corrcoef(self.x, self.y, rowvar=0)[0, 1]


    def test_py_dcor(self):
        """Test for py_dcor"""

        # Compare against Pearson correlation
        dcor = py_dcor(self.x, self.y)
        diff = self.pearson_r - dcor
        msg  = "Python: Difference (%.4f) between Pearson correlation (%.4f) " \
               "and distance correlation (%.4f) too large" % \
                (diff, self.pearson_r, dcor)
        self.assertAlmostEqual(dcor, self.pearson_r, delta=.05, msg=msg)


    def test_py_wdcor(self):
        """Test for py_wdcor"""

        # Compare against Pearson correlation
        wdcor = py_wdcor(self.x, self.y, self.weights)
        diff  = self.pearson_r - wdcor
        msg   = "Python: Difference (%.4f) between Pearson correlation (%.4f) " \
                "and weighted distance correlation (%.4f) too large" % \
                (diff, self.pearson_r, wdcor)
        self.assertAlmostEqual(wdcor, self.pearson_r, delta=.05, msg=msg)


    def test_approx_wdcor(self):
        """Test for approx_dcor"""

        # Compare against Pearson correlation
        wdcor = approx_wdcor(self.x, self.y)
        diff  = self.pearson_r - wdcor
        msg   = "Difference (%.4f) between Pearson correlation (%.4f) " \
                "and approximate weighted distance correlation (%.4f) too " \
                "large" % (diff, self.pearson_r, wdcor)
        self.assertAlmostEqual(wdcor, self.pearson_r, delta=.05, msg=msg)


    def test_c_dcor(self):
        """Test for c_dcor"""

        # Compare against Pearson correlation
        dcor = c_dcor(self.x, self.y)
        diff = self.pearson_r - dcor
        msg  = "C: Difference (%.4f) between Pearson correlation (%.4f) " \
               "and distance correlation (%.4f) too large" % \
                (diff, self.pearson_r, dcor)
        self.assertAlmostEqual(dcor, self.pearson_r, delta=.05, msg=msg)


    def test_c_wdcor(self):
        """Test for c_wdcor"""

        # Compare against Pearson correlation
        wdcor = c_wdcor(self.x, self.y, self.weights)
        diff  = self.pearson_r - wdcor
        msg   = "C: Difference (%.4f) between Pearson correlation (%.4f) " \
                "and weighted distance correlation (%.4f) too large" % \
                (diff, self.pearson_r, wdcor)
        self.assertAlmostEqual(wdcor, self.pearson_r, delta=.05, msg=msg)


    def test_rdc(self):
        """Test for rdc"""

        # Compare against Pearson correlation
        cor  = rdc(self.x, self.y)
        diff = self.pearson_r - cor
        msg  = "Difference (%.4f) between Pearson correlation (%.4f) " \
               "and randomized dependence coefficient(%.4f) too large" % \
               (diff, self.pearson_r, cor)
        self.assertAlmostEqual(cor, self.pearson_r, delta=.05, msg=msg)


    def test_gini_index(self):
        """Test for gini_index"""

        gini_best      = np.array([1, 1])
        gini_worst     = np.array([0, 1]) 
        est_gini_best  = gini_index(gini_best, [0, 1])
        est_gini_worst = gini_index(gini_worst, [0, 1]) 
 
        msg = "Gini index (%.4f) should be 0.0" % est_gini_best
        self.assertAlmostEqual(est_gini_best, 0.0, delta=0.0, msg=msg)

        msg = "Gini index (%.4f) should be 0.5" % est_gini_worst
        self.assertAlmostEqual(est_gini_worst, 0.5, delta=0.0, msg=msg)


if __name__ == '__main__':
    unittest.main()
