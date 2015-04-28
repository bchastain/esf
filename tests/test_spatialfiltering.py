"""Unit test for spatialfiltering.py"""
import unittest
import pysal
import numpy as np

from esf.spatialfiltering import spatialfiltering


class TestSpatialfiltering(unittest.TestCase):

    def setUp(self):
        self.data = pysal.examples.get_path("columbus.dbf")
        self.nb = pysal.examples.get_path("columbus.gal")

    def test_spatialfiltering(self):
        """
        Run a series of tests based off R function results for comparison
        """
        dependent_var = "CRIME"
        independent_vars = ["INC", "HOVAL"]
        spatiallag = None
        neighbor_list = self.nb
        data = self.data
        style = "r"
        zero_policy = False
        tolerance = 0.1
        zero_value = 0.0001
        exact_EV = True
        symmetric = True
        alpha = None
        alternative = "two.sided"
        verbose = False
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Check with parameters as given in R SpatialFiltering docs.
        # Compare eigenvector selection list.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5, 3, 1, 10, 14])

        style = "v"
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using variance-stabilized weight matrix.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [4, 5, 1, 10, 14, 2, 11, 12, 16, 3])

        style = "b"
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using binary weight matrix.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [4, 3, 1, 12, 2, 11, 9, 16, 10, 5, 13, 17, 18])

        tolerance = 0.5
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using a different convergence tolerance.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [4, 3, 1, 12, 2, 11, 9, 16, 10])

        style = "r"
        tolerance = 0.1
        zero_value = 0.5
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using a different eigenvector exclusing threshold.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5, 3, 1, 2])

        zero_value = 0.0001
        alpha = 0.5
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try the alternative alpha-based stopping method instead of tolerance.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5, 3, 1, 10])

        exact_EV = False
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try not using exact expectation and variance of Moran's I.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5, 3, 1, 10])

        exact_EV = True
        alternative = "greater"
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using a greater-than alternative hypothesis.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5, 3, 1, 10, 14])

        alternative = "less"
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using a less-than alternative hypothesis.
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [5])


        alternative = "two.sided"
        alpha = None
        independent_vars = []
        spatiallag = ["INC", "HOVAL"]
        out, selVec = spatialfiltering(
            dependent_var, independent_vars, spatiallag, data,
            neighbor_list, style, zero_policy, tolerance, zero_value,
            exact_EV, symmetric, alpha, alternative, verbose
        )
        # Try using spatial lag model instead of SAR
        self.assertEqual(np.array(out[1:, 1]).astype(np.int).T[0].tolist(),
                         [6, 4, 1, 17, 5, 16, 15, 11, 19, 18])

if __name__ == '__main__':
    unittest.main()
