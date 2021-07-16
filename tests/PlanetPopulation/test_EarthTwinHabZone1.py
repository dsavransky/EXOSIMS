r"""Test code for EarthTwinHabZone1 module within EXOSIMS PlanetPopulation.

Cate Liu, IPAC, 2016
"""

import unittest
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone1 import EarthTwinHabZone1
import os
import numpy as np
from astropy import units as u
import scipy.stats


class TestEarthTwinHabZone1(unittest.TestCase):
    
    def setUp(self):
        self.spec = {'modules':{'PlanetPhysicalModel': ''}}
        pass
    
    def tearDown(self):
        pass
    
    def test_gen_plan_params(self):
        r"""Test generated planet parameters:
        Expected: all 1 R_E, all p = 0.67, e = 0, and uniform a in arange

        Updated by Sonny Rapapport, Cornell 7/16/2021. Strategy: Generate the histogram of values for 
        the generated values. As the distribution should be uniform, just check that all the buckets
        in the histogram have approximately even distributions with a chi^2 test. 
        """

        obj = EarthTwinHabZone1(**self.spec)

        x = 10000

        a, e, p, Rp = obj.gen_plan_params(x)
        
        assert(np.all(e == 0))
        assert(np.all(p == 0.367))
        assert(np.all(Rp == 1.0*u.R_earth))

        h = np.histogram(a.to('AU').value,100,density=False)
        crit = scipy.stats.chi2.ppf(1-.01,99)
        #critical value chi^2: chi^2 must be smaller than this value for .01 signifiance
        chi2 = scipy.stats.chisquare(h[0])
        self.assertLess(chi2[0], crit)
        #assert that chi^2 is less than critical value 

    
if __name__ == "__main__":
    unittest.main()
