#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""Utility unit tests

Michael Turmon, JPL, Apr. 2016
"""

import unittest
from EXOSIMS.util.eccanom import eccanom
import numpy as np


class TestUtilityMethods(unittest.TestCase):
    r"""Test utility functions."""

    def test_eccanom(self):
        r"""Test eccentric anomaly computation.

        Approach: Reference to pre-computed results from Matlab, plus
        additional check that solutions satisfy Kepler's equation.
        Tests restricted to eccentricity between 0.0 and 0.4."""

        # eccanom() appears to be a specialized version of newtonm.m from
        # Vallado's m-files.  eccanom() implements the case in the m-file
        # marked "elliptical", because it works for planets in elliptical orbits.

        print("eccanom()")

        # empty input should return empty output
        empty_input = np.array([])
        empty_output = eccanom(empty_input, 0.1)
        self.assertEqual(empty_output.shape, (0,))
        self.assertTrue(isinstance(empty_output, np.ndarray))

        # precomputed from newtonm.m in Vallado's Matlab source code
        tabulation = {
            # a few systematically-chosen values, a few random ones
            # (eccentricity in [0,0.4] and mean anomaly in [0,2 pi]
            # label    eccentricity     mean anomaly    ecc-anomaly    true-anomaly
            "syst-0": (0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000),
            "syst-1": (0.025000000000, 0.400000000000, 0.409964417354, 0.420045901819),
            "syst-2": (0.050000000000, 0.800000000000, 0.837136437398, 0.874924655639),
            "syst-3": (0.075000000000, 1.200000000000, 1.271669563802, 1.344211492229),
            "syst-4": (0.100000000000, 1.600000000000, 1.699177050713, 1.797889059320),
            "syst-5": (0.125000000000, 2.000000000000, 2.107429361800, 2.211834518809),
            "syst-6": (0.150000000000, 2.400000000000, 2.490864846892, 2.577019718783),
            "syst-7": (0.175000000000, 2.800000000000, 2.850264344358, 2.896965705233),
            "syst-8": (0.200000000000, 3.200000000000, 3.190268645286, -3.101846256861),
            "syst-9": (0.225000000000, 3.600000000000, 3.517416263407, -2.841371075544),
            "syst-10": (
                0.250000000000,
                4.000000000000,
                3.839370814447,
                -2.592284710937,
            ),
            "syst-11": (
                0.275000000000,
                4.400000000000,
                4.165158556909,
                -2.340284012237,
            ),
            "syst-12": (
                0.300000000000,
                4.800000000000,
                4.506345584831,
                -2.066225859687,
            ),
            "syst-13": (
                0.325000000000,
                5.200000000000,
                4.879529007461,
                -1.739297079458,
            ),
            "syst-14": (
                0.350000000000,
                5.600000000000,
                5.310823513733,
                -1.301816690532,
            ),
            "syst-15": (
                0.375000000000,
                6.000000000000,
                5.838779417089,
                -0.646704214263,
            ),
            "rand-1": (0.021228232647, 5.868452651315, 5.859729688192, -0.432264760951),
            "rand-2": (0.016968378871, 4.761021655110, 4.744061786583, -1.556088761716),
            "rand-3": (0.018578311703, 2.464435046216, 2.475908955821, 2.487300499232),
            "rand-4": (0.016386947254, 1.075597681642, 1.090127758250, 1.104713814823),
            "rand-5": (0.017651152200, 0.200011672644, 0.203580329847, 0.207180380010),
            "rand-6": (0.006923074624, 0.290103403226, 0.292096978801, 0.294097208267),
            "rand-7": (0.002428294531, 5.173938128028, 5.171761572640, -1.113601464490),
            "rand-8": (0.017370715574, 1.992394794033, 2.008130645020, 2.023809681520),
            "rand-9": (0.023755551221, 0.216431106906, 0.221653600172, 0.226938058952),
            "rand-10": (0.010968608991, 2.397402491437, 2.404772696274, 2.412113273400),
            "rand-11": (
                0.019137919704,
                4.996388335095,
                4.977921138154,
                -1.323779018431,
            ),
            "rand-12": (0.004671815114, 3.077280455596, 3.077579312617, 3.077877470783),
            "rand-13": (
                0.011139655018,
                4.060904408970,
                4.052106100441,
                -2.239847775747,
            ),
            "rand-14": (
                0.017734120771,
                4.741836271756,
                4.724103367769,
                -1.576817615338,
            ),
            "rand-15": (
                0.006900626925,
                4.270697872458,
                4.264477964900,
                -2.024918020450,
            ),
            "rand-16": (0.016377450099, 1.021719665350, 1.035808775845, 1.049957669774),
            "rand-17": (0.002974942039, 3.131313689041, 3.131344177237, 3.131374620007),
            "rand-18": (0.023993598963, 2.138706596562, 2.158672164638, 2.178507958606),
            "rand-19": (0.014631693774, 1.406251889782, 1.420719116738, 1.435202708137),
            "rand-20": (0.018781676483, 1.602809881387, 1.621567356335, 1.640316999728),
        }

        for _, value in tabulation.items():
            (
                ref_eccentricity,
                ref_mean_anomaly,
                ref_ecc_anomaly,
                _ref_true_anomaly,
            ) = value
            exo_ecc_anomaly = eccanom(ref_mean_anomaly, ref_eccentricity)
            # 1: ensure the output agrees with the tabulation
            self.assertAlmostEqual(exo_ecc_anomaly, ref_ecc_anomaly, delta=1e-6)
            # 2: additionally, ensure the Kepler relation:
            #   M = E - e sin E
            # is satisfied for the output of eccanom().
            # Here, e is the given eccentricity, E is the EXOSIMS eccentric
            # anomaly, and M is the mean anomaly.
            est_mean_anomaly = exo_ecc_anomaly - ref_eccentricity * np.sin(
                exo_ecc_anomaly
            )
            self.assertAlmostEqual(ref_mean_anomaly, est_mean_anomaly, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
