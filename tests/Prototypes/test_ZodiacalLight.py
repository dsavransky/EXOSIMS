import unittest
import numpy as np
import os
import EXOSIMS
from  EXOSIMS import MissionSim
from EXOSIMS.Prototypes import ZodiacalLight
import numpy as np
from astropy import units as u
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams

r"""ZodiacalLight module unit tests

Paul Nunez, JPL, Aug. 2016
"""

scriptfile = resource_path('test-scripts/template_prototype_testing.json')

class Test_Zodiacal_prototype(unittest.TestCase):
    dev_null = open(os.devnull, 'w')


    def setUp(self):
        with RedirectStreams(stdout=self.dev_null):
            sim = MissionSim.MissionSim(scriptfile)
        self.targetlist = sim.modules['TargetList']
        self.nStars = sim.TargetList.nStars
        self.star_index = np.array(range(0, self.nStars))
        self.observatory = sim.Observatory
        self.mode = sim.OpticalSystem.observingModes[0]
        self.timekeeping = sim.TimeKeeping
        self.sim = sim
        assert self.nStars > 10, "Need at least 10 stars in the target list for the unit test."
        
    def test_fz_case(self):
        obj = ZodiacalLight.ZodiacalLight()
        r_sc = np.zeros((self.nStars,3)) * u.km

        # test the default case        
        expected = np.array([6.309e-10] * self.nStars)
        result = obj.fZ(self.observatory, self.targetlist, self.star_index, self.timekeeping.currentTimeAbs, self.mode )
        result_unitless = (result*u.arcsec**2).decompose()
        self.assertEqual(result_unitless.unit, u.dimensionless_unscaled)
        #diff = expected - result_unitless.value
        np.testing.assert_allclose(result_unitless.value, expected, rtol=1e-2, atol=0.)

        # re-run with another magnitude
        obj.magZ = 22.5
        expected = np.array([1e-9] * self.nStars)
        result = obj.fZ(self.observatory, self.targetlist, self.star_index, self.timekeeping.currentTimeAbs, self.mode )
        result_unitless = (result*u.arcsec**2).decompose()
        self.assertEqual(result_unitless.unit, u.dimensionless_unscaled)
        np.testing.assert_allclose(result_unitless.value, expected, rtol=1e-2, atol=0.)
    
    @unittest.skip('Test currently needs improvement in generality to work')
    def test_fEZ_lognormal(self):
        r'''Test fEZ inclination variation, and lognormal randomization.

        Currently needs work, see comments in the test code.'''
        obj = ZodiacalLight.ZodiacalLight()
        Inc = np.linspace(0.0, 180.0, self.nStars) * u.deg
        # test the default        
        # turmon: two problems:
        # (1) this test varies inclination, but "expected" below does not take
        # inclination into account
        # (2) the fEZ code takes the V-band magnitude of the star into account,
        # but this test does not.
        # Therefore, we cannot use this test as written.
        expected = np.array([2.377e-09] * self.nStars) 
        result = obj.fEZ(self.targetlist, self.star_index, Inc)
        result_unitless = (result*u.arcsec**2).decompose()
        self.assertEqual(result_unitless.unit, u.dimensionless_unscaled)
        np.testing.assert_allclose(result_unitless.value, expected, rtol=1e-2, atol=0)       
        
        # Test case of non-zero variance
        # turmon: the monte carlo approach used here is OK, but only if the star is held
        # constant across all monte carlo draws (or the per-star adjustment is
        # taken into account)
        obj.varEZ = 0.5
        fEZ = obj.fEZ(self.targetlist, self.star_index, Inc)

        # turmon: these averages might be better to do in the log domain, i.e.,
        # take mean(log(fEZ)) and compare to log(expected), and
        # take std(log(fEZ)) and compare to the std. of the lognormal
        np.testing.assert_allclose(2.377e-09, np.mean(fEZ*u.arcsec**2), rtol=1e-2, atol=0.01)
        np.testing.assert_allclose(1.121e-09, np.std(fEZ*u.arcsec**2), rtol=1e-2, atol=0.01)

        #note that nEZ was being changed into a numpy array. Should be a float.
        assert type(obj.magZ) is float
        assert type(obj.magEZ) is float
        assert type(obj.nEZ) is float
        assert type(obj.varEZ) is float

    def test_generate_fZ(self):
        r"""Test generate fZ method
        """
        # for mod in self.allmods:
        #     if 'generate_fZ' in mod.__dict__:

        # with RedirectStreams(stdout=self.dev_null):
        #     sim = mod(scriptfile=self.script)

        #Check if File Exists and if it does, delete it
        sim = self.sim
        try:
            sim.SurveySimulation.cachefname
        except:
            import pdb
            pdb.set_trace()

        if os.path.isfile(sim.SurveySimulation.cachefname+'starkfZ'):
            os.remove(sim.SurveySimulation.cachefname+'starkfZ')
        sInds = np.asarray([0])
        Obs = sim.Observatory
        TL = sim.TargetList
        currentTimeAbs = sim.TimeKeeping.currentTimeAbs
        OS = sim.OpticalSystem
        allModes = OS.observingModes
        mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        hashname = sim.SurveySimulation.cachefname
        sim.ZodiacalLight.fZ_startSaved = sim.ZodiacalLight.generate_fZ(Obs, TL, currentTimeAbs, mode, hashname)
        self.assertEqual(sim.ZodiacalLight.fZ_startSaved.shape[0],TL.nStars)
        #Should also check length of fZ_startSaved??
        self.assertEqual(sim.ZodiacalLight.fZ_startSaved.shape[1],1000)#This was arbitrarily selected.

    def test_calcfZmax(self):
        r"""Test calcfZmax method
        """
        # for mod in self.allmods:
        #     if 'calcfZmax' in mod.__dict__:

        # with RedirectStreams(stdout=self.dev_null):
        #     sim = mod(scriptfile=self.script)

        #Check if File Exists and if it does, delete it
        sim = self.sim
        if os.path.isfile(sim.SurveySimulation.cachefname+'starkfZ'):
            os.remove(sim.SurveySimulation.cachefname+'starkfZ')
        sInds = np.arange(5)
        Obs = sim.Observatory
        TL = sim.TargetList
        currentTimeAbs = sim.TimeKeeping.currentTimeAbs
        OS = sim.OpticalSystem
        allModes = OS.observingModes
        mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        hashname = sim.SurveySimulation.cachefname
        sim.ZodiacalLight.fZ_startSaved = sim.ZodiacalLight.generate_fZ(Obs, TL, currentTimeAbs, mode, hashname)
        valfZmax = np.zeros(sInds.shape[0])
        timefZmax = np.zeros(sInds.shape[0])
        [valfZmax, timefZmax] = sim.ZodiacalLight.calcfZmax(sInds, Obs, TL, currentTimeAbs, mode, hashname)
        self.assertTrue(len(valfZmax) == len(sInds))
        self.assertTrue(len(timefZmax) == len(sInds))
        self.assertTrue(valfZmax[0].unit == 1/u.arcsec**2)
        self.assertTrue(timefZmax[0].format == currentTimeAbs.format)

    def test_calcfZmin(self):
        r"""Test calcfZmin method
        """
        # for mod in self.allmods:
        #     if 'calcfZmin' in mod.__dict__:

        # with RedirectStreams(stdout=self.dev_null):
        #     sim = mod(scriptfile=self.script)
        sim = self.sim
        #Check if File Exists and if it does, delete it
        if os.path.isfile(sim.SurveySimulation.cachefname+'starkfZ'):
            os.remove(sim.SurveySimulation.cachefname+'starkfZ')
        sInds = np.asarray([0])
        Obs = sim.Observatory
        TL = sim.TargetList
        currentTimeAbs = sim.TimeKeeping.currentTimeAbs
        OS = sim.OpticalSystem
        allModes = OS.observingModes
        mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        hashname = sim.SurveySimulation.cachefname
        sim.ZodiacalLight.fZ_startSaved = sim.ZodiacalLight.generate_fZ(Obs, TL, currentTimeAbs, mode, hashname)
        [valfZmin, timefZmin] = sim.ZodiacalLight.calcfZmin(sInds, Obs, TL, currentTimeAbs, mode, hashname)
        self.assertTrue(len(valfZmin) == len(sInds))
        self.assertTrue(len(timefZmin) == len(sInds))
        self.assertTrue(valfZmin[0].unit == 1/u.arcsec**2)
        self.assertTrue(timefZmin[0].format == currentTimeAbs.format)

if __name__ == '__main__':
    unittest.main()

    
