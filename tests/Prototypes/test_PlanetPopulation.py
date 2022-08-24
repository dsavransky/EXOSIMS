r"""Test code for PlanetPopulation Prototype module within EXOSIMS.

Cate Liu, IPAC, 2016"""

import unittest
import EXOSIMS
from  EXOSIMS import MissionSim
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
from astropy import units as u
from astropy import constants as const
import os
import math
import scipy.stats
import EXOSIMS.util.statsFun as sf


class TestPlanetPopulation(unittest.TestCase):
    def setUp(self):

        self.spec = {"modules":{"PlanetPhysicalModel" : "PlanetPhysicalModel"}}
        
        pass
    
    def tearDown(self):
        pass
        

    def test_gen_angles(self):
        """
        Test generation of orientation angles.
        
        We expect long. and periapse to be uniformly distributed and
        inclination to be sinusoidally distributed.

        Edit made by Sonny Rappaport, Cornell, July 2021:
        SciPY update has broken this method, so use KS test to check inclination
        distribution and alter usage of chi^2 test for the uniform distributions
        """

        pp = PlanetPopulation(**self.spec)

        x = 100000
        I, O, w = pp.gen_angles(x)

        crit = scipy.stats.chi2.ppf(1-.01,99)
        #critical chi^2 value for .01 level significance , 100 degrees freedom
        # (100 buckets)

        #O & w are expected to be uniform
        for param,param_range in zip([O,w],[pp.Orange,pp.wrange]):
            h = np.histogram(param,100,density=False)
            #critical value chi^2: chi^2 must be smaller than this value for .01 signifiance
            chi2 = scipy.stats.chisquare(h[0])
            self.assertLess(chi2[0], crit)
            #assert that chi^2 is less than critical value 

        #I is expected to be sinusoidal
            
        sin_cdf = lambda x: -np.cos(x)/2+.5
        #cdf of the sin distribution for ks test
        ks_result = scipy.stats.kstest(I,sin_cdf)

        self.assertGreater(ks_result[1],.01)
        #assert that the p value is greater than .01 

    def test_gen_plan_params(self):
        """
        Test generation of planet orbital and phyiscal properties.

        We expect eccentricity and albedo to be uniformly distributed
        and sma and radius to be log-uniform

        Edit made by Sonny Rappaport, Cornell, July 2021:
        SciPY update has broken this method, so use KS test to check log-uniform
        distribution and alter usage of chi^2 test for the uniform distributions
        """
        pp = PlanetPopulation(**self.spec)

        x = 100000

        a, e, p, Rp = pp.gen_plan_params(x)

        crit = scipy.stats.chi2.ppf(1-.01,99)
        #critical chi^2 value for .01 level significance , 100 degrees freedom
        # (100 buckets)

        #expect e and p to be uniform
        for param,param_range in zip([e,p],[pp.erange,pp.prange]):
            h = np.histogram(param,100,density=False)
            #critical value chi^2: chi^2 must be smaller than this value for .01 signifiance
            chi2 = scipy.stats.chisquare(h[0])
            self.assertLess(chi2[0], crit)
            #assert that chi^2 is less than critical value 

        #expect a and Rp to be log-uniform
        for param,param_range in zip([a.value,Rp.value],[pp.arange.value,pp.Rprange.value]):

            expected = scipy.stats.loguniform.rvs(param_range[0],param_range[1],size=x)
            #use scipy's log-uniform distribution with appropriate bounds

            ks_result = scipy.stats.kstest(expected,param)

            self.assertGreater(ks_result[1],.01)
            #assert that the p value is greater than .01 


    def test_checkranges(self):
        """
        Test that check ranges is doing what it should do

        """

        pp = PlanetPopulation(arange=[10,1],**self.spec)
        self.assertTrue(pp.arange[0].value == 1)
        self.assertTrue(pp.arange[1].value == 10)
        
        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(prange=[-1,1],**self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(erange=[-1,1],**self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(arange=[0,1],**self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(Rprange=[0,1],**self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(Mprange=[0,1],**self.spec)


if __name__ == "__main__":
    unittest.main()
