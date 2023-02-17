import unittest

import numpy as np
import sys, os.path
import astropy.units as u
import inspect
import EXOSIMS.util.waypoint as wp

"""
waypoint.py module unit tests
Sonny Rappaport, June 2021 (in format of code for test_deltaMag)
General strategy: I put in arbitrary inputs and ensure that the dictionary 
generated has the correct sums. I do not check the plot/that a file has been
generated. 
"""


class Test_waypoint(unittest.TestCase):
    def test1(self):
        """Testing the waypoint function for various arbitrary inputs"""

        comps = []
        intTimes = [] * u.d
        self.assertDictEqual(
            wp.waypoint(comps, intTimes, 365, None, None),
            {"numStars": 0, "Total Completeness": 0, "Total intTime": 0},
        )

        comps = [1, 2, 3]
        intTimes = [1, 2, 3] * u.d
        self.assertDictEqual(
            wp.waypoint(comps, intTimes, 365, None, None),
            {"numStars": 3, "Total Completeness": 6, "Total intTime": 6 * u.d},
        )

        comps = [3, 3, 3]
        intTimes = [180, 180, 180] * u.d
        self.assertDictEqual(
            wp.waypoint(comps, intTimes, 365, None, None),
            {"numStars": 2, "Total Completeness": 6, "Total intTime": 360 * u.d},
        )


if __name__ == "__main__":
    unittest.main()
