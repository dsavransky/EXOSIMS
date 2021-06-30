r"""Test code for EarthTwinHabZone1 module within EXOSIMS PlanetPopulation.

Cate Liu, IPAC, 2016"""

import unittest
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone1 import EarthTwinHabZone1
import os
import numpy as np
from astropy import units as u
import scipy.stats
import xmlrunner


class TestEarthTwinHabZone1(unittest.TestCase):
    
    def setUp(self):
        self.spec = {'modules':{'PlanetPhysicalModel': ''}}
        pass
    
    def tearDown(self):
        pass
    
    def test_gen_plan_params(self):
        r"""Test generated planet parameters:
        Expected: all 1 R_E, all p = 0.67, e = 0, and uniform a in arange
        """

        obj = EarthTwinHabZone1(**self.spec)

        x = 10000

        a, e, p, Rp = obj.gen_plan_params(x)
        
        assert(np.all(e == 0))
        assert(np.all(p == 0.367))
        assert(np.all(Rp == 1.0*u.R_earth))

        h = np.histogram(a.to('AU').value,100,density=True)
        chi2 = scipy.stats.chisquare(h[0],[1.0/np.diff(obj.arange.to('AU').value)[0]]*len(h[0]))
        self.assertGreater(chi2[1], 0.95)

    
if __name__ == '__main__':
    with open('../../../test-results.xml', 'wb') as output:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output=output),
            failfast=False, buffer=False, catchbreak=False)
