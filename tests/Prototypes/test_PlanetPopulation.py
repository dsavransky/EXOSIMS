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

        Test method: chi squares
        """

        pp = PlanetPopulation(**self.spec)

        x = 100000
        I, O, w = pp.gen_angles(x)
        assert(I.min() >= pp.Irange[0])
        assert(I.max() <= pp.Irange[1])
        assert(O.min() >= pp.Orange[0])
        assert(O.max() <= pp.Orange[1])
        assert(w.min() >= pp.wrange[0])
        assert(w.max() <= pp.wrange[1])

        #O & w are expected to be uniform
        for param,param_range in zip([O,w],[pp.Orange,pp.wrange]):
            h = np.histogram(param,100,density=True)
            chi2 = scipy.stats.chisquare(h[0],[1.0/np.diff(param_range.value)[0]]*len(h[0]))
            self.assertGreater(chi2[1], 0.95)


        #I is expected to be sinusoidal
        hI = np.histogram(I.to(u.rad).value,100,density=True)
        Ix = np.diff(hI[1])/2.+hI[1][:-1]
        Ip = np.sin(Ix)/2

        Ichi2 = scipy.stats.chisquare(hI[0],Ip)
        assert(Ichi2[1] > 0.95)

    
    def test_gen_plan_params(self):
        """
        Test generation of planet orbital and phyiscal properties.

        We expect eccentricity and albedo to be uniformly distributed
        and sma and radius to be log-uniform

        Test method: chi squares
        """
        pp = PlanetPopulation(**self.spec)

        x = 10000

        a, e, p, Rp = pp.gen_plan_params(x)

        #expect e and p to be uniform
        for param,param_range in zip([e,p],[pp.erange,pp.prange]):
            assert(param.min() >= param_range[0])
            assert(param.max() <= param_range[1])

            h = np.histogram(param,100,density=True)
            chi2 = scipy.stats.chisquare(h[0],[1.0/np.diff(param_range)[0]]*len(h[0]))

            assert(chi2[1] > 0.95)

        #expect a and Rp to be log-uniform
        for param,param_range in zip([a.value,Rp.value],[pp.arange.value,pp.Rprange.value]):
            assert(param.min() >= param_range[0])
            assert(param.max() <= param_range[1])

            h = np.histogram(param,100,density=True)
            hx = np.diff(h[1])/2.+h[1][:-1]
            hp = 1.0/(hx*np.log(param_range[1]/param_range[0]))
            chi2 = scipy.stats.chisquare(h[0],hp)
            assert(chi2[1] > 0.95)
    


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
