r"""Test code for EarthTwinHabZone2 module within EXOSIMS PlanetPopulation.

Cate Liu, IPAC, 2016"""

import unittest
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone2 import EarthTwinHabZone2
from tests.TestSupport.Info import resource_path
import os
import numpy as np
import astropy.units as u
import scipy.stats

class TestEarthTwinHabZone2(unittest.TestCase):

    def setUp(self):
        self.spec = {'modules':{'PlanetPhysicalModel': ''}}
        pass
    
    def tearDown(self):
        pass
    
    def test_gen_plan_params(self):
        r"""Test generated planet parameters:
        Expected: all 1 R_E, all p = 0.67, e = 0, and uniform a,e in arange,erange
        """

        obj = EarthTwinHabZone2(constrainOrbits=False,erange=[0.1,0.5],**self.spec)

        x = 10000

        a, e, p, Rp = obj.gen_plan_params(x)
        
        assert(np.all(p == 0.367))
        assert(np.all(Rp == 1.0*u.R_earth))
        

        for param,param_range in zip([a.value,e],[obj.arange.value,obj.erange]):
            assert(param.min() >= param_range[0])
            assert(param.max() <= param_range[1])

            h = np.histogram(param,100,density=True)
            chi2 = scipy.stats.chisquare(h[0],[1.0/np.diff(param_range)[0]]*len(h[0]))
            self.assertGreater(chi2[1], 0.95)

    def test_gen_plan_params2(self):
        r"""Test generated planet parameters with constrainOrbits defaulting to True
        Expected: a(1+e) < arange.max(); a(1-e) > arange.min()
        """

        
        obj = EarthTwinHabZone2(**self.spec)

        x = 10000

        a, e, p, Rp = obj.gen_plan_params(x)

        assert(np.all(a*(1+e) <= obj.arange[1]))
        assert(np.all(a*(1-e) >= obj.arange[0]))
    
    
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestEarthTwinHabZone2)
    #unittest.TextTestRunner().run(suite)   
    unittest.main()

