#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""Kepler State Propagation unit tests

Michael Turmon, JPL, Apr. 2016
"""

import unittest
from EXOSIMS.util import keplerSTM, keplerSTM_indprop
import numpy as np
from collections import namedtuple
import sys

# These target values are pasted from printed output of Matlab test routine
KnownResults = [
    {
        "name": "Mercury",
        "pos": [-0.093047, 0.295288, 0.034854],
        "vel": [9.23773452, 2.61287326, 0.46008903],
    },
    {
        "name": "Venus",
        "pos": [0.143513, 0.687112, 0.041329],
        "vel": [7.14057119, -1.94564186, -0.09038863],
    },
    {
        "name": "Earth",
        "pos": [-0.055181, -0.490804, 0.899096],
        "vel": [-0.91297753, -5.43159857, -2.10067777],
    },
    {
        "name": "Mars",
        "pos": [0.999611, -0.552948, -0.017243],
        "vel": [-3.28382927, -5.06514036, -0.17164179],
    },
    {
        "name": "Jupiter",
        "pos": [3.826163, 2.755374, 0.064499],
        "vel": [1.53970096, -2.39598942, -0.05357969],
    },
    {
        "name": "Saturn",
        "pos": [7.032313, 4.695869, 0.218432],
        "vel": [1.07881821, -1.85515408, -0.07894125],
    },
    {
        "name": "Uranus",
        "pos": [13.687179, 11.088116, 0.157498],
        "vel": [0.86590322, -1.18182375, -0.01633361],
    },
    {
        "name": "Neptune",
        "pos": [9.992835, 27.279418, 0.867159],
        "vel": [1.05637671, -0.43477705, -0.01262013],
    },
    {
        "name": "Pluto",
        "pos": [16.659255, 10.351263, 4.800579],
        "vel": [0.32709102, -1.37664357, -0.39479977],
    },
]

# convenient holder for the above results
PlanetInfo = namedtuple("PlanetInfo", ["name", "pos", "vel"])
PlanetState = [PlanetInfo(**d) for d in KnownResults]


class Consts(object):
    """A bag to hold constants.  They must agree with the Matlab test code!"""

    # nominal values -- for a solar system like ours
    oneAU = 1.51e11
    # [m]
    Msun = 2.0e30
    # [kg]
    # seconds-per-year, for tidy display of numbers
    sec_per_year = 3600 * 24 * 365
    # correct G
    G = 6.67384e-11
    # [m^3/kg/s/s]


# holds the bag of constants
consts = Consts()

names = [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
]
masses = np.array([0.330, 4.87, 5.97, 0.642, 1898, 568, 86.8, 102, 0.0146]) * 1e24
# [kg]
dists = (
    np.array([57.9, 108.2, 149.6, 227.9, 778.6, 1433.5, 2872.5, 4495.1, 5906.4]) * 1e9
)
# [m]
peris = (
    np.array([46.0, 107.5, 147.1, 206.6, 740.5, 1352.6, 2741.3, 4444.5, 4436.8]) * 1e9
)
# [m]
incs = np.array([7.0, 3.4, 99.0, 1.9, 1.3, 2.5, 0.8, 1.8, 17.2])
# [deg]
speeds = np.array([47.4, 35.0, 29.8, 24.1, 13.1, 9.7, 6.8, 5.4, 4.7]) * 1e3
# [m/s]
periods = np.array([88.0, 224.7, 365.2, 687.0, 4331, 10747, 30589, 59800, 90560]) * (
    24.0 * 3600
)
# [s]

# cosine/sine in degrees
cosd = lambda theta: np.cos(np.radians(theta))
sind = lambda theta: np.sin(np.radians(theta))


class TestKeplerSTM(unittest.TestCase):
    """
    Test method: Verify the Python code-under-test with pre-computed values from a Matlab
    re-implementation of the same algorithm.  We check both position and velocity for state
    propagation of solar-system-analog planets (9 planets), over about one orbital period.

    Specific tests vary which implementation (C or Python) and which of the two implemented
    algorithms are used
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_keplerSTM_init(self):
        """
        Test method: initialize all possible constructors and sanity check generated objects
        """

        test_str = "EXOSIMS.util.KeplerSTM_C.CyKeplerSTM"

        ps = keplerSTM.planSys(np.random.randn(6), np.random.rand(1))
        if test_str in sys.modules:
            self.assertTrue(ps.havec)
        else:
            self.assertFalse(ps.havec)
        self.assertEqual(ps.algOrder[0], ps.calcSTM)
        self.assertEqual(ps.nplanets, 1)

        ps2 = keplerSTM.planSys(
            np.random.randn(6), np.random.rand(1), noc=True, prefVallado=True
        )
        self.assertFalse(ps2.havec)
        self.assertEqual(ps2.algOrder[1], ps2.calcSTM)
        self.assertEqual(ps2.nplanets, 1)

        ps3 = keplerSTM_indprop.planSys(np.random.randn(6), np.random.rand(1))
        if test_str in sys.modules:
            self.assertTrue(ps3.havec)
        else:
            self.assertFalse(ps3.havec)
        self.assertEqual(ps3.nplanets, 1)

        ps4 = keplerSTM_indprop.planSys(np.random.randn(6), np.random.rand(1), noc=True)
        self.assertFalse(ps4.havec)
        self.assertEqual(ps4.nplanets, 1)

        with self.assertRaises(Exception):
            ps0 = keplerSTM.planSys(np.random.randn(6), np.random.rand(2))

        with self.assertRaises(Exception):
            ps0 = keplerSTM.planSys(np.random.randn(9), np.random.rand(1))

        with self.assertRaises(Exception):
            ps0 = keplerSTM_indprop.planSys(np.random.randn(6), np.random.rand(2))

        with self.assertRaises(Exception):
            ps0 = keplerSTM_indprop.planSys(np.random.randn(9), np.random.rand(1))

    def test_keplerSTM_propC(self):
        self.propagate_and_check(keplerSTM.planSys)

    def test_keplerSTM_proppy(self):
        self.propagate_and_check(keplerSTM.planSys, noc=True)

    def test_keplerSTM_prop_vallado(self):
        self.propagate_and_check(keplerSTM.planSys, noc=True, prefVallado=True)

    def test_keplerSTM_indpropC(self):
        self.propagate_and_check(keplerSTM_indprop.planSys)

    def test_keplerSTM_indproppy(self):
        self.propagate_and_check(keplerSTM_indprop.planSys, noc=True)

    def propagate_and_check(self, keplerObj, **kwargs):
        r"""Test Kepler State Propagation.

        Args:
            keplerObj: Which
        """
        ## 1: Define a planetary system

        # From http://nssdc.gsfc.nasa.gov/planetary/factsheet/
        #   Mercury/Venus/Earth/Mars/Jupiter/Saturn/Uranus/Neptune/Pluto
        #   Earth orbital inclination has been altered to make the test more comprehensive
        # Note: having physically correct values is *not important*.
        # Note: these constants *must* agree with the Matlab file.

        for i in range(len(names)):
            # initial position: x=0, y large, at peak inclination in the z direction
            # we use the perihelion, not mean distance, so the orbit is ensured to be elliptical
            #   units are AU
            x0p = [
                0,
                (peris[i] / consts.oneAU) * cosd(incs[i]),
                (peris[i] / consts.oneAU) * sind(incs[i]),
            ]
            # initial velocity: x increasing, y decreasing, z steady
            #   units are AU/s
            x0v = [
                (speeds[i] / consts.oneAU) * cosd(incs[i]),
                -(speeds[i] / consts.oneAU) * sind(incs[i]),
                0,
            ]

            # units of mu will come out in AU^3/s^2
            mu_bare = (
                (masses[i] + consts.Msun)
                * consts.G
                / (consts.oneAU * consts.oneAU * consts.oneAU)
            )
            # object under test needs nu.array inputs
            mu = np.array([mu_bare])

            # initial position: perihelion, at mean speed
            x0 = np.array([x0p[0], x0p[1], x0p[2], x0v[0], x0v[1], x0v[2]])

            # instantiate object
            ps = keplerObj(x0, mu, **kwargs)

            # go for about one orbital period
            dt = periods[i]
            ps.takeStep(dt)

            # extract position and velocity
            position = ps.x0[0:3]  # already in AU
            velocity = ps.x0[3:6] * consts.sec_per_year  # convert to AU/year

            # error tolerance
            tol = 1e-5

            # check each coordinate
            for coord in [0, 1, 2]:
                self.assertAlmostEqual(
                    position[coord], PlanetState[i].pos[coord], delta=tol
                )
                self.assertAlmostEqual(
                    velocity[coord], PlanetState[i].vel[coord], delta=tol
                )


if __name__ == "__main__":
    unittest.main()
