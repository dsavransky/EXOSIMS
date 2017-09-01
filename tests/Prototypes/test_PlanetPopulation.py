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
        #scriptfile = os.path.join(EXOSIMS.__path__[0],'Prototypes','template_WFIRST_EarthTwinHabZone.json')
        #self.sim = MissionSim.MissionSim(scriptfile)
        #self.outspecs = self.sim.genOutSpec()
        #self.targetlist = self.sim.modules['TargetList']
        
        """
        self.spec={"arange": [0.01, 10.], "erange": [10.*np.finfo(np.float).eps ,  0.8],
                          "wrange": [0., 360.], "Orange": [0., 360.], "prange": [0.0004, 0.6],
                          "Rrange": [0.027*const.R_jup.to(u.km), 2.04*const.R_jup.to(u.km)],
                          "Mprange": [6.3e-5*const.M_jup, 28.5*const.M_jup]}
        """

        self.spec = {"modules":{"PlanetPhysicalModel" : "PlanetPhysicalModel"}}
        
        pass
    
    def tearDown(self):
        pass
        

    def test_gen_angles(self):
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
    
    def test_gen_plan_params_constrainOrbits(self):
        pp = PlanetPopulation(constrainOrbits=True,**self.spec)

        x = 10000

        a, e, p, Rp = pp.gen_plan_params(x)

        self.assertTrue(np.all(a*(1+e) <= pp.arange[1]))
        self.assertTrue(np.all(a*(1-e) >= pp.arange[0]))


    
    def test_gen_mass(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_mass(x)
        res = np.log(res.value)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.Mprange.value
        v = np.log(v)
        vmin = np.amin(v)
        vmax = np.amax(v)
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin, vmin, rtol=1e-01, atol=0.1)
        np.testing.assert_allclose(resmax, vmax, rtol=1e-01, atol=0.)
        np.testing.assert_allclose(resmean, vmean, rtol=1e-01, atol=0.)


if __name__ == "__main__":
    unittest.main()
