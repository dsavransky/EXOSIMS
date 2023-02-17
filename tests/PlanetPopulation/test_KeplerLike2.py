import os
import unittest
import numpy as np
import astropy.units as u
from EXOSIMS.PlanetPopulation.KeplerLike2 import KeplerLike2
from tests.TestSupport.Utilities import RedirectStreams
import scipy.stats
import EXOSIMS.util.statsFun as sf


class TestKeplerLike2Methods(unittest.TestCase):
    r"""Test PlanetPopulation KeplerLike1 class."""

    def setUp(self):
        # allow the chatter on stdout during object creation to be suppressed
        self.dev_null = open(os.devnull, "w")

        specs = {}
        specs["modules"] = {}
        specs["modules"][
            "PlanetPhysicalModel"
        ] = " "  # so the Prototype for PhysMod will be used

        # the with clause allows the chatter on stdout/stderr during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null, stderr=self.dev_null):
            self.fixture = KeplerLike2(**specs)

    def tearDown(self):
        self.dev_null.close()
        del self.fixture

    def test_gen_sma(self):
        r"""Test gen_sma method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that they are in the correct range and follow the distribution.
        """

        plan_pop = self.fixture
        n = 10000
        sma = plan_pop.gen_sma(n)

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
