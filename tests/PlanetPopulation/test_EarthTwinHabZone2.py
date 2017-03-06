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

class TestEarthTwinHabZone2(unittest.TestCase):
    
    def setUp(self):
        if False:
            # this is a round-about way to generate the specs dictionary: disabled.
            scriptfile = resource_path('/test-scripts/ipac_testscript.json')
            self.sim = MissionSim.MissionSim(scriptfile)
            self.spec = self.sim.genOutSpec()
        else:
            # PlanetPopulation just needs a PlanetPhysicalModel: use the Prototype
            self.spec = {'modules':{'PlanetPhysicalModel': ''}}
    
    def tearDown(self):
        pass
    
    # Another idea for a test is to ensure that the properties guaranteed
    # by EarthTwinHabZone2 are correct: 1 R_Earth, 1 M_Eearth, 1 p_Earth
    # (see the docs for the module).
    # This is not yet done.

    def test_gensma(self):
        r"""Test gen_sma method.

        Approach: Generate SMA's, and check that their range is correct.
        The mean is also checked here, but it need not, in principle,
        agree, depending on the SMA distribution used.  For uniform,
        it works."""
        obj = EarthTwinHabZone2(**self.spec)
        
        res = obj.gen_sma(5000)
        res = res.value
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = obj.arange.value
        vmin = np.amin(v)
        vmax = np.amax(v)
        vmean = np.mean(v)

        np.testing.assert_allclose(resmin, vmin, rtol=0.01, atol=0.)
        np.testing.assert_allclose(resmax, vmax, rtol=0.01, atol=0.)
        np.testing.assert_allclose(resmean, vmean, rtol=0.2, atol=0.)
    
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestEarthTwinHabZone2)
    #unittest.TextTestRunner().run(suite)   
    unittest.main()

