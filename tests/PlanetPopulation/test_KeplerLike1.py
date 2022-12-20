#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""KeplerLike1 module unit tests

Michael Turmon, JPL, May 2016

Sonny Rappaport, Cornell University, 2021 (specifically, fixing test_gen_sma method
which doesn't work on latest SciPY)
"""

import os
import unittest
import numpy as np
import astropy.units as u
from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
from tests.TestSupport.Utilities import RedirectStreams
import scipy.stats
import EXOSIMS.util.statsFun as sf


class TestKeplerLike1Methods(unittest.TestCase):
    r"""Test PlanetPopulation KeplerLike1 class."""

    def setUp(self):
        # allow the chatter on stdout during object creation to be suppressed
        self.dev_null = open(os.devnull, "w")

        # The chain of init methods for this object is:
        #    KeplerLike1.__init__ ->
        #    Prototype/PlanetPopulation.__init__
        # The last init then finally does something like this:
        #    PlanPhys = get_module(specs['modules']['PlanetPhysicalModel'], 'PlanetPhysicalModel')
        #    self.PlanetPhysicalModel = PlanPhys(**specs)
        # To make this work, when we __init__ the module under test here,
        # we need to supply a "specs" that has a proper value for:
        #    [modules][PlanetPhysicalModel]
        # so that get_module can find the value.
        # Like so:
        specs = {}
        specs["modules"] = {}
        specs["modules"][
            "PlanetPhysicalModel"
        ] = " "  # so the Prototype for PhysMod will be used

        # the with clause allows the chatter on stdout/stderr during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null, stderr=self.dev_null):
            self.fixture = KeplerLike1(**specs)

    def tearDown(self):
        self.dev_null.close()
        del self.fixture

    def test_gen_mass(self):
        r"""Test gen_mass method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that returned values are nonnegative.  Check that, for this power law, there
        are more small than large masses (for n large).
        """

        plan_pop = self.fixture
        n = 10000
        masses = plan_pop.gen_mass(n)

        self.assertEqual(len(masses), n)
        self.assertTrue(np.all(masses.value >= 0))
        self.assertTrue(np.all(np.isfinite(masses.value)))

        midpoint = np.mean(masses)
        self.assertGreater(
            np.count_nonzero(masses < midpoint), np.count_nonzero(masses > midpoint)
        )

        # test some illegal "n" values
        n_list_bad = [-1, "100", 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                masses = plan_pop.gen_mass(n)

    def test_gen_sma(self):
        r"""Test gen_sma method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that they are in the correct range and follow the distribution.

        Sonny Rappaport, July 2021, Cornell: Use the KS test to see test if gen_sma follows
        the correct distribution from plan_post.dist_sma.
        """

        plan_pop = self.fixture
        n = 10000
        sma = plan_pop.gen_sma(n)
        print(sma)

        ar = plan_pop.arange.to("AU").value
        # unitless range

        # ensure the units are length
        self.assertEqual((sma / u.km).decompose().unit, u.dimensionless_unscaled)
        # sma > 0
        self.assertTrue(np.all(sma.value >= 0))
        # sma >= arange[0], sma <= arange[1]
        self.assertTrue(np.all(sma - plan_pop.arange[0] >= 0))
        self.assertTrue(np.all(plan_pop.arange[1] - sma >= 0))

        sma = plan_pop.gen_sma(n).to("AU").value
        # take the generated samples and make them unitless

        expected_samples = sf.simpSample(plan_pop.dist_sma, n, ar[0], ar[1])
        # generate expected sample from plan.pop's dist_sma, range from 0 to the maximum range ar[1]

        ks_result = scipy.stats.kstest(expected_samples, sma)

        self.assertGreater(ks_result[1], 0.01)
        # assert that the p value is greater than .01

    def test_gen_radius(self):
        r"""Test gen_radius method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check distributional agreement.
        """
        plan_pop = self.fixture
        n = 10000
        radii = plan_pop.gen_radius(n)

        # ensure the units are length
        self.assertEqual((radii / u.km).decompose().unit, u.dimensionless_unscaled)
        # radius > 0
        self.assertTrue(np.all(radii.value > 0))
        self.assertTrue(np.all(np.isfinite(radii.value)))

        h = np.histogram(radii.to("earthRad").value, bins=plan_pop.Rs)
        np.testing.assert_allclose(
            plan_pop.Rvals.sum() * h[0] / float(n), plan_pop.Rvals, rtol=0.05
        )

    def test_gen_radius_nonorm(self):
        r"""Test gen_radius_nonorm method.

        Approach: Just checking to see that we generate the number of samples consistent
        with the occurrence rate given by the Kepler radius distribution

        """
        plan_pop = self.fixture

        n = 100
        ns = np.array([len(plan_pop.gen_radius_nonorm(n)) for j in range(1000)])

        np.testing.assert_allclose(
            np.mean(ns) / float(n), np.sum(plan_pop.Rvals), rtol=0.01
        )

    def test_gen_albedo(self):
        r"""Test gen_albedo method.

        Approach: Ensures the output is set, of the correct type and length.
        Check that they are in the correct range.
        """

        plan_pop = self.fixture
        n = 10000
        # call the routine
        albedos = plan_pop.gen_albedo(n)
        # check the type
        self.assertEqual(type(albedos), np.ndarray)
        self.assertEqual(len(albedos), n)
        # albedos >= 0
        self.assertTrue(np.all(albedos >= 0.0))
        # albedos <= 1
        self.assertTrue(np.all(albedos <= 1.0))
        # test some illegal "n" values
        n_list_bad = [-1, "100", 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                sma = plan_pop.gen_albedo(n)


if __name__ == "__main__":
    unittest.main()
