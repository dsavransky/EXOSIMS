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

    # allow the chatter on stdout during object creation to be suppressed
    dev_null = open(os.devnull, 'w')

    def setUp(self):
        specs = {}
        specs['modules'] = {}
        specs['modules']['PlanetPhysicalModel'] = ' ' # so the Prototype for PhysMod will be used

        # the with clause allows the chatter on stdout/stderr during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null, stderr=self.dev_null):
            self.fixture = KeplerLike2(**specs)

    def tearDown(self):
        del self.fixture

    def test_gen_sma(self):
        r"""Test gen_sma method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that they are in the correct range and follow the distribution.
        """

        plan_pop = self.fixture
        n = 10000
        sma = plan_pop.gen_sma(n)

        # ensure the units are length
        self.assertEqual((sma/u.km).decompose().unit, u.dimensionless_unscaled)
        # sma > 0
        self.assertTrue(np.all(sma.value >= 0))
        # sma >= arange[0], sma <= arange[1]
        self.assertTrue(np.all(sma - plan_pop.arange[0] >= 0))
        self.assertTrue(np.all(plan_pop.arange[1] - sma >= 0))

        h = np.histogram(sma.to('AU').value,100,density=False)
        hx = np.diff(h[1])/2.+h[1][:-1]
        hp = plan_pop.dist_sma(hx)
        h_norm = sf.norm_array(h[0]) 
        hp_norm = sf.norm_array(hp)
        #because chisquare now requires the sum of the frequencies to be the same, normalize each sum to 1 and use that in the chi^2
        chi2 = scipy.stats.chisquare(h_norm,hp_norm)
        self.assertGreaterEqual(chi2[1],0.95)
