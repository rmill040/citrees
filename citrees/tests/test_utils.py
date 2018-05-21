from __future__ import absolute_import, division, print_function

import numpy as np
from os.path import abspath, dirname
import sys
import unittest

# Add path to avoid relative imports
PATH = dirname(dirname(abspath(__file__)))
if PATH not in sys.path: sys.path.append(PATH)

from externals.six.moves import zip
from utils import auc_score, estimate_margin


class TestScorers(unittest.TestCase):

    def setUp(self):
        """Generate sample data"""
        
        y_probs1     = np.array([.1, .1, .9, .9])
        y_probs75    = np.array([.7, .1, .8, .6])
        y_probs50    = np.array([.9, .1, .6, .6])
        y_probs25    = np.array([.7, .3, .1, .4])
        y_probs0     = np.array([.9, .9, .1, .1])
        self.y_probs = [y_probs1, y_probs75, y_probs50, y_probs25, y_probs0]
        self.y_true  = np.array([0, 0, 1, 1])


    def test_auc_score(self):
        """Test for auc_score"""

        # Compare against true AUC
        for auc, y_probs in zip([1.0, .75, .50, .25, .00], self.y_probs):
            auc_est = auc_score(self.y_true, y_probs)
            diff    = auc - auc_est
            msg     = "Difference (%.2f) between true AUC (%.2f) and " \
                      "estimated AUC (%.2f) is too large" % \
                      (diff, auc, auc_est)

        self.assertEqual(auc, auc_est, msg=msg)


    def test_estimate_margin(self):
        """Test for estimate_margin"""

        # Sample data
        y_true = np.array([2, 0, 1, 1])

        y_probs = np.array([
                [.10, .20, .70], # True = 2, margin = .70-.20 =  .50
                [.90, .05, .05], # True = 0, margin = .90-.05 =  .85
                [.80, .10, .10], # True = 1, margin = .10-.80 = -.70
                [.35, .60, .05]  # True = 1, margin = .60-.35 =  .25
            ])

        # Calculate margins
        est_margin  = estimate_margin(y_probs, y_true)
        true_margin = np.array([.50, .85, -.70, .25])
        diff        = np.mean((est_margin-true_margin)**2)

        self.assertAlmostEqual(diff, 0.0, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
