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
    
    def test_gen_sma(self):
        #pp = self.sim.modules['PlanetPopulation']
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_sma(x)
        res = np.log(res.value)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.arange.value  #inputs
        v = np.log(v)
        vmin = np.min(v)
        vmax = np.max(v)
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin, vmin, rtol=0.1, atol=0)
        np.testing.assert_allclose(resmax, vmax, rtol=0.1, atol=0)
        # the following assertion is reliant on the exponential distribution
        # currently used by PlanetPopulation
        np.testing.assert_allclose(resmean, vmean, rtol=0.1, atol=0)
        
        #count, bins, ignored = plt.hist(res, 50, normed=True)
        #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        #plt.show()
    
    def test_gen_eccen(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_eccen(x)
        
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.erange
        vmean = (v[1]-v[0])/2.-v[0]
        
        np.testing.assert_allclose(resmin, v[0], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmax, v[1], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmean, vmean, rtol=1e-01, atol=0.5)
        
        #count, bins, ignored = plt.hist(res, 50, normed=True)
        #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        #plt.show()
    
    def test_gen_eccen_from_sma(self):
        pp = PlanetPopulation(**self.spec)
        
        n = 5000
        #a = np.zeros(n) + 5.
        a = pp.gen_sma(n)
        res=pp.gen_eccen_from_sma(n,a)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        vmin = pp.erange[0]
        elim = np.min(np.vstack((1 - (pp.arange[0]/a).decompose().value,\
                (pp.arange[1]/a).decompose().value - 1)),axis=0)
        vmax = np.amax(elim)
        vmean = (vmax - vmin)/2.
        
        np.testing.assert_allclose(resmin, vmin, rtol=1e-02, atol=0.5)
        np.testing.assert_allclose(resmax, vmax, rtol=1e-02, atol=0.5)
        np.testing.assert_allclose(resmean, vmean, rtol=1e-02, atol=1.)
        
    def test_gen_w(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_w(x)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.wrange.value
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin.value, v[0], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmax.value, v[1], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmean.value, vmean, rtol=1e-01, atol=1.)
    
    def test_gen_O(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_O(x)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.Orange.value
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin.value, v[0], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmax.value, v[1], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmean.value, vmean, rtol=1e-01, atol=1.)
    
    def test_gen_radius(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_radius(x)
        res = np.log(res.value)
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.Rprange.value
        v = np.log(v)
        vmin = np.amin(v)
        vmax = np.amax(v)
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin, vmin, rtol=1e-01, atol=0.1)
        np.testing.assert_allclose(resmax, vmax, rtol=1e-01, atol=0.)
        np.testing.assert_allclose(resmean, vmean, rtol=1e-01, atol=0.)
    
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
    
    def test_gen_albedo(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_albedo(x)
        
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = pp.prange
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin, v[0], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmax, v[1], rtol=1e-01, atol=0.5)
        np.testing.assert_allclose(resmean, vmean, rtol=1e-01, atol=0.5)

    def test_gen_I(self):
        pp = PlanetPopulation(**self.spec)
        
        x = 5000
        res = pp.gen_I(x)

        res = np.cos(res)
        res = res.value
        
        resmin = np.amin(res)
        resmax = np.amax(res)
        resmean = np.mean(res)
        
        v = np.cos(pp.Irange)
        vmin = np.amin(v)
        vmax = np.amax(v)
        vmean = np.mean(v)
        
        np.testing.assert_allclose(resmin, vmin, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(resmax, vmax, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(resmean, vmean, rtol=0.2, atol=0.5)

        #count, bins, ignored = plt.hist(res, 50, normed=True)
        #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        #plt.show()
    

if __name__ == "__main__":
    unittest.main()
