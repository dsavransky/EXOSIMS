r"""Test code for EarthTwinHabZone3 module within EXOSIMS PlanetPopulation.

Sonny Rappaport, Cornell, July 2021"""

import unittest
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone3 import EarthTwinHabZone3
from tests.TestSupport.Info import resource_path
import os
import numpy as np
import astropy.units as u
import scipy.stats

class TestEarthTwinHabZone3(unittest.TestCase):

    def setUp(self):
        self.spec = {'modules':{'PlanetPhysicalModel': ''}}
        pass
    
    def tearDown(self):
        pass
    
    def test_gen_sma(self):
        r"""Tests generated semi-major axis

        Strategy: Uses chi^2 test to check that uniform distribution is indeed uniform
        
        """

        obj = EarthTwinHabZone3(**self.spec)

        x = 10000

        a = obj.gen_sma(x)


        crit = scipy.stats.chi2.ppf(1-.01,99)
        h = np.histogram(a.to('AU').value,100,density=False)

        crit = scipy.stats.chi2.ppf(1-.01,99)
        #critical value chi^2: chi^2 must be smaller than this value for .01 signifiance
        chi2 = scipy.stats.chisquare(h[0])
        self.assertLess(chi2[0], crit)
        #assert that chi^2 is less than critical value 
    
    
if __name__ == "__main__":
    unittest.main()
