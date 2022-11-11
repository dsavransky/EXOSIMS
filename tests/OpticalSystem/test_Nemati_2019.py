import unittest
from tests.TestSupport.Info import resource_path
from EXOSIMS.OpticalSystem.Nemati_2019 import Nemati_2019
import numpy as np


class TestNemati2019(unittest.TestCase):

    """
    Sonny Rappaport, August 2021, Cornell

    This class tests particular methods Nemanti_2019.

    """

    def test_get_csv_values(self):

        """
        Tests whether get_csv_values returns the correct columns of data given
        the corresponding input
        """

        path = resource_path("test-scripts/nemati_2019_testCSV.csv")

        # the three columns represented in the test csv files
        test1 = [1, 1, 1, 1, 1]
        test2 = [2, 2, 2, 2, 2]
        test3 = [3, 3, 3, 3, 3]

        # test that three headers are correctly called in order
        output = Nemati_2019.get_csv_values(self, path, "test1", "test2", "test3")
        np.testing.assert_array_equal(output, [test1, test2, test3])

        # tests that two headers are correctly called in order
        output = Nemati_2019.get_csv_values(self, path, "test2", "test3")
        np.testing.assert_array_equal(output, [test2, test3])

        # test that a comment row is filtered out
        path2 = resource_path("test-scripts/nemati_2019_testCSV_comments.csv")
        output = Nemati_2019.get_csv_values(self, path2, "test1", "test2")
        np.testing.assert_array_equal(output, [test1, test2])
