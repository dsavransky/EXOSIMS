#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""KeplerLike1 module unit tests

Michael Turmon, JPL, May 2016
"""

import sys
import os
import json
import csv
import logging
import StringIO
import unittest
from collections import defaultdict
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy import integrate,stats
from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
from tests.TestSupport.Utilities import RedirectStreams


class TestKeplerLike1Methods(unittest.TestCase):
    r"""Test PlanetPopulation KeplerLike1 class."""

    # allow the chatter on stdout during object creation to be suppressed
    dev_null = open(os.devnull, 'w')

    def setUp(self):
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
        specs['modules'] = {}
        specs['modules']['PlanetPhysicalModel'] = ' ' # so the Prototype for PhysMod will be used

        # the with clause allows the chatter on stdout/stderr during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null, stderr=self.dev_null):
            self.fixture = KeplerLike1(**specs)

    def tearDown(self):
        del self.fixture

    def validate_planet_population(self, plan_pop):
        r"""Consolidate simple validation of PlanetPopulation attributes in one place."""
        
        self.assertEqual(plan_pop._modtype, 'PlanetPopulation')
        self.assertIsInstance(plan_pop._outspec, dict)
        # check for presence of one class attribute
        self.assertIn('smadist', plan_pop.__dict__)
        self.assertIn('edist',   plan_pop.__dict__)


    #@unittest.skip('skipping init')
    def test_init_trivial(self):
        r"""Test of initialization and __init__ -- trivial setup/teardown test.
        """
        plan_pop = self.fixture
        # ensure basic module attributes are set up
        self.validate_planet_population(plan_pop)


    # @unittest.skip('gen_mass')
    def test_gen_mass(self):
        r"""Test gen_mass method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that returned values are nonnegative.  Check that, for this power law, there
        are more small than large masses (for n large).
        """

        print 'gen_mass()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
            # call the routine
            masses = plan_pop.gen_mass(n)
            # check the type
            self.assertEqual(type(masses), type(1.0 * u.kg))
            self.assertEqual(len(masses), n)
            # ensure the units are kg
            self.assertEqual(masses.unit, u.kg)
            # Note: since mass is computed from radius, user-set mass limits are
            # ignored.  So we cannot check against Mprange.
            self.assertTrue(np.all(masses.to(u.kg).value >= 0))
            self.assertTrue(np.all(np.isfinite(masses.to(u.kg).value)))
            # crude check on the shape (more small than large for this power law)
            if n >= 100:
                midpoint = np.mean(plan_pop.Mprange)
                self.assertGreater(np.count_nonzero(masses < midpoint),
                                   np.count_nonzero(masses > midpoint))
        # test some illegal "n" values
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                masses = plan_pop.gen_mass(n)


    # @unittest.skip('gen_sma')
    def test_gen_sma(self):
        r"""Test gen_sma method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that they are in the correct range.
        Check that, for this power law, there are more small than large SMA (for n large).
        Deliberately omitted for now: checking the frequency curve.
        """

        print 'gen_sma()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
            # call the routine
            sma = plan_pop.gen_sma(n)
            # check the type
            self.assertEqual(type(sma), type(1.0 * u.km))
            self.assertEqual(len(sma), n)
            # ensure the units are length
            self.assertEqual((sma/u.km).decompose().unit, u.dimensionless_unscaled)
            # sma > 0
            self.assertTrue(np.all(sma.value >= 0))
            # sma >= arange[0], sma <= arange[1]
            self.assertTrue(np.all(sma - plan_pop.arange[0] >= 0))
            self.assertTrue(np.all(plan_pop.arange[1] - sma >= 0))
            # crude check on the shape (sma is a power law
            # so we require more small than large)
            if n >= 100:
                midpoint = (np.min(sma) + np.max(sma)) * 0.5
                self.assertGreater(np.count_nonzero(sma < midpoint),
                                   np.count_nonzero(sma > midpoint))
        # test some illegal "n" values
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                sma = plan_pop.gen_sma(n)
            
    # @unittest.skip('gen_radius')
    def test_gen_radius(self):
        r"""Test gen_radius method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that there are more small than large radii (for n large).
        TODO: Check distributional agreement.
        """
        print 'gen_radius()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
            # call the given routine
            radii = plan_pop.gen_radius(n)
            # check the type
            self.assertEqual(type(radii), type(1.0 * u.km))
            self.assertEqual(len(radii), n)
            # ensure the units are length
            self.assertEqual((radii/u.km).decompose().unit, u.dimensionless_unscaled)
            # radius > 0
            self.assertTrue(np.all(radii.value > 0))
            self.assertTrue(np.all(np.isfinite(radii.value)))
            # crude check on the shape 
            #  (we require more small than large radii)
            if n >= 100:
                midpoint = (np.min(radii) + np.max(radii)) * 0.5
                self.assertGreater(np.count_nonzero(radii < midpoint),
                                   np.count_nonzero(radii > midpoint))
        # test some illegal "n" values
        # Note: as long as we're checking this, -1 should be illegal, but is passed thru
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                # call the given routine
                radii = plan_pop.gen_radius(n)
    
    # @unittest.skip('gen_radius_nonorm')
    def test_gen_radius_nonorm(self):
        r"""Test gen_radius_nonorm method.

        Approach: Ensures the output is set, of the correct type and units, and
        is above zero.
        Note: We do not check the specific distribution.
        """
        print 'gen_radius_nonorm()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
            # call the given routine
            radii = plan_pop.gen_radius_nonorm(n)
            # check the type
            self.assertEqual(type(radii), type(1.0 * u.km))
            # not in this case!
            # self.assertEqual(len(radii), n)
            # ensure the units are length
            self.assertEqual((radii/u.km).decompose().unit, u.dimensionless_unscaled)
            # radius > 0
            self.assertTrue(np.all(radii.value > 0))
            self.assertTrue(np.all(np.isfinite(radii.value)))
        # test some illegal "n" values
        # Note: as long as we're checking this, -1 should be illegal, but is passed thru
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                # call the given routine
                radii = plan_pop.gen_radius_nonorm(n)

    
    # @unittest.skip('gen_eccentricity')
    def test_gen_eccentricity(self):
        r"""Test gen_eccentricity method.

        Approach: Ensures the output is set, of the correct type and length.
        Check that they are in the correct range.
        """

        print 'gen_eccentricity()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
            # call the routine
            eccs = plan_pop.gen_eccen(n)
            # check the type
            self.assertEqual(type(eccs), np.ndarray)
            self.assertEqual(len(eccs), n)
            # eccs >= 0
            self.assertTrue(np.all(eccs >= 0.0))
            # eccs <= 1
            self.assertTrue(np.all(eccs <= 1.0))
            # eccs >= erange[0], eccs <= erange[1]
            self.assertTrue(np.all(eccs - plan_pop.erange[0] >= 0))
            self.assertTrue(np.all(plan_pop.erange[1] - eccs >= 0))
        # test some illegal "n" values
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                eccs = plan_pop.gen_eccen(n)

                
    # @unittest.skip('gen_albedo')
    def test_gen_albedo(self):
        r"""Test gen_albedo method.

        Approach: Ensures the output is set, of the correct type and length.
        Check that they are in the correct range.
        """

        print 'gen_albedo()'
        plan_pop = self.fixture
        n_list = [0, 1, 20, 100, 500, 1002]
        for n in n_list:
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
        n_list_bad = [-1, '100', 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                sma = plan_pop.gen_albedo(n)
                
    def test_gen_eccentricity_dist(self):
        r"""Test gen_eccentricity method -- distribution correctness.

        Approach: Take a large number of samples and ensure, using a
        Kolmogorov-Smirnov test, that they follow the expected law.
        """
        print 'gen_eccentricity: dist'
        plan_pop = self.fixture
        n_samples = 20000
        # get a lot of samples
        eccs = plan_pop.gen_eccen(n_samples)
        # perform the KS test (second arg is the density)
        ks_result = stats.kstest(eccs,
                                lambda x: pdf_to_cdf(x, plan_pop.edist, plan_pop.erange))
        # print 'KS test:', ks_result
        self.assertGreaterEqual(ks_result.pvalue, 0.005,
                                'KS test indicates bad eccentricity distribution')

    def test_gen_sma_dist(self):
        r"""Test gen_sma method -- distribution correctness.

        Approach: Take a large number of samples and ensure, using a
        Kolmogorov-Smirnov test, that they follow the expected law.
        """
        print 'gen_sma: dist'
        plan_pop = self.fixture
        n_samples = 20000
        # get a lot of samples
        smas = plan_pop.gen_sma(n_samples)
        # perform the KS test (second arg is the density)
        #   the distribution and the range need to be in AU
        ks_result = stats.kstest(smas.to(u.au).value,
                                lambda x: pdf_to_cdf(x, plan_pop.smadist, plan_pop.arange.to(u.au).value))
        # print 'KS test:', ks_result
        self.assertGreaterEqual(ks_result.pvalue, 0.005,
                                'KS test indicates bad SMA distribution')


def pdf_to_cdf(x, pdf, x_range):
    r"""Compute the cdf of a vector of given values `x', given a density `pdf' and a range x_range."""
    result = np.zeros(len(x))
    # ensure the distribution is normalized (cdf = 1 over range[0]..range[1])
    #   the method used by the utility sampler of EXOSIMS does not require any
    #   normalization, so the densities are typically un-normalized
    normalizer = integrate.quad(pdf, x_range[0], x_range[1])[0]
    # compute the cdf by integrating for each input x
    for i in range(len(x)):
        integral = integrate.quad(pdf, x_range[0], x[i])[0]
        result[i] = integral / normalizer
    return result


if __name__ == '__main__':
    unittest.main()
