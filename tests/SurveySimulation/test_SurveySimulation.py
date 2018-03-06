import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.SurveySimulation
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.get_module import get_module
import os
import pkgutil
import numpy as np
import astropy.units as u
import pdb
import os.path

class TestSurveySimulation(unittest.TestCase):
    """ 

    Global SurveySimulation tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """
    
    def setUp(self):

        self.dev_null = open(os.devnull, 'w')
        self.script = resource_path('test-scripts/simplest.json')
    
        modtype = getattr(SurveySimulation,'_modtype')
        pkg = EXOSIMS.SurveySimulation
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)

    def test_init(self):
        """
        Test of initialization and __init__.
        
        """

        exclude_mods=['KnownRVSurvey']

        required_modules = [
            'BackgroundSources', 'Completeness', 'Observatory', 'OpticalSystem',
            'PlanetPhysicalModel', 'PlanetPopulation', 'PostProcessing', 
            'SimulatedUniverse', 'TargetList', 'TimeKeeping', 'ZodiacalLight' ]
        
        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue
            
            with RedirectStreams(stdout=self.dev_null):
                sim = mod(scriptfile=self.script)

            self.assertIsInstance(sim._outspec, dict)
            # check for presence of a couple of class attributes
            self.assertIn('DRM', sim.__dict__)

            for rmod in required_modules:
                self.assertIn(rmod, sim.__dict__)
                self.assertEqual(getattr(sim,rmod)._modtype,rmod)

    def test_run_sim(self):
        r"""Test run_sim method.

        Approach: Ensures the simulation runs to completion and the output is set.
        """

        #expected contents of DRM:
        DRM_keys =  ['FA_char_status',
                     'char_mode',
                     'det_status',
                     'char_params',
                     'star_name',
                     'plan_inds',
                     'FA_char_dMag',
                     'OB_nb',
                     'char_fZ',
                     'det_SNR',
                     'FA_char_fEZ',
                     'char_status',
                     'det_mode',
                     'det_time',
                     'arrival_time',
                     'char_SNR',
                     'det_params',
                     'char_time',
                     'FA_char_SNR',
                     'det_fZ',
                     'FA_det_status',
                     'star_ind',
                     'FA_char_WA']


        for mod in self.allmods:
            if 'run_sim' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)
                    sim.run_sim()
                # check that the mission time is indeed elapsed
                self.assertGreaterEqual(sim.TimeKeeping.currentTimeNorm,
                                        sim.TimeKeeping.missionFinishNorm,
                                        'Mission did not run to completion for %s'%mod.__name__)
                # resulting DRM is a list...
                self.assertIsInstance(sim.DRM, list, 'DRM is not a list for %s'%mod.__name__)
                # ...and has nontrivial number of entries
                self.assertGreater(len(sim.DRM), 0, 'DRM is empty for %s'%mod.__name__)


                for key in DRM_keys:
                    self.assertIn(key,sim.DRM[0].keys(),'DRM is missing key %s for %s'%(key,mod.__name__))
   
    def test_next_target(self):
        r"""Test next_target method.

        Approach: Ensure the next target is selected OK, and is a valid integer.
        Deficiencies: We are not checking that the occulter slew works.
        """

        for mod in self.allmods:
            if 'next_target' in mod.__dict__:
        
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                DRM_out,sInd,intTime = sim.next_target(None, sim.OpticalSystem.observingModes[0])

                # result index is a scalar numpy ndarray, that is a valid integer
                # in a valid range
                self.assertIsInstance(sInd, int, 'sInd is not an integer for %s'%mod.__name__)
                self.assertEqual(sInd - int(sInd), 0, 'sInd is not an integer for %s'%mod.__name__)
                self.assertGreaterEqual(sInd, 0, 'sInd is not a valid index for %s'%mod.__name__)
                self.assertLess(sInd, sim.TargetList.nStars, 'sInd is not a valid index for %s'%mod.__name__)

                # resulting DRM is a dictionary -- contents unimportant
                self.assertIsInstance(DRM_out, dict, 'DRM_out is not a dict for %s'%mod.__name__)

    def test_choose_next_target(self):
        r"""Test choose_next_target method.

        Approach: Ensure the next target is a valid index for different cases: old_sInd is none,
        old_sInd in sInds, old_sInd not in sInds
        """

        for mod in self.allmods:
            if 'choose_next_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)
                
                #old sInd is None
                sInds = np.random.choice(sim.TargetList.nStars,size=int(sim.TargetList.nStars/2.0),replace=False)
                sInd = sim.choose_next_target(None,sInds,
                        np.array([1.0]*sim.TargetList.nStars)*u.d,
                        np.array([1.0]*len(sInds))*u.d)
                self.assertTrue(sInd in sInds,'sInd not in passed sInds for %s'%mod.__name__)

                #old sInd in sInds
                sInds = np.random.choice(sim.TargetList.nStars,size=int(sim.TargetList.nStars/2.0),replace=False)
                old_sInd = np.random.choice(sInds)
                _ = sim.observation_detection(old_sInd,1.0*u.d,sim.OpticalSystem.observingModes[0])
                sInd = sim.choose_next_target(old_sInd,sInds,
                        np.array([1.0]*sim.TargetList.nStars)*u.d,
                        np.array([1.0]*len(sInds))*u.d)

                self.assertTrue(sInd in sInds,'sInd not in passed sInds for %s'%mod.__name__)

                #old sInd not in sInds
                sInds = np.random.choice(sim.TargetList.nStars,size=int(sim.TargetList.nStars/2.0),replace=False)
                tmp = list(set(np.arange(sim.TargetList.nStars)) - set(sInds))
                old_sInd = np.random.choice(tmp)
                _ = sim.observation_detection(old_sInd,1.0*u.d,sim.OpticalSystem.observingModes[0])
                sInd = sim.choose_next_target(old_sInd,sInds,
                        np.array([1.0]*sim.TargetList.nStars)*u.d,
                        np.array([1.0]*len(sInds))*u.d)

                self.assertTrue(sInd in sInds,'sInd not in passed sInds for %s'%mod.__name__)

    def test_choose_revisit_target(self):
        r"""Test choose_revisit_target Method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                for sInd in [0,None]:
                    sInd = sim.choose_revisit_target(sInd)
                    try:
                        self.assertIsInstance(sInd, int)
                    except:
                        self.assertIsInstance(sInd, type(None))

    def test_observation_detection(self):
        r"""Test observation_detection method.

        Approach: Ensure that all outputs are set as expected
        """

        for mod in self.allmods:
            if 'observation_detection' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                #defualt settings should create dummy planet around first star
                sInd = 0
                pInds = np.where(sim.SimulatedUniverse.plan2star == sInd)[0]
                detected, fZ, systemParams, SNR, FA = sim.observation_detection(sInd,1.0*u.d,sim.OpticalSystem.observingModes[0])
                
                self.assertEqual(len(detected),len(pInds))
                self.assertIsInstance(detected[0],int)
                for s in SNR[detected == 1]:
                    self.assertGreaterEqual(s,sim.OpticalSystem.observingModes[0]['SNR'])
                self.assertIsInstance(FA, bool)    

    def test_scheduleRevisit(self):
        """Runs scheduleRevisit method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                sInd = [0]
                smin = None
                det = 0
                pInds = [0]
                sim.scheduleRevisit(sInd,smin,det,pInds)

    def test_observation_characterization(self):
        r"""Test observation_characterization method.

        Approach: Ensure all outputs are set as expected
        """


        for mod in self.allmods:
            if 'observation_characterization' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)


                #defualt settings should create dummy planet around first star
                sInd = 0
                pInds = np.where(sim.SimulatedUniverse.plan2star == sInd)[0]

                #in order to test for characterization, we need to have previously 
                #detected the planet, so let's do that first
                detected, fZ, systemParams, SNR, FA = sim.observation_detection(sInd,1.0*u.d,sim.OpticalSystem.observingModes[0])
                #now the characterization
                characterized, fZ, systemParams, SNR, intTime = sim.observation_characterization(sInd,sim.OpticalSystem.observingModes[0])

                self.assertEqual(len(characterized),len(pInds))
                self.assertIsInstance(characterized[0],int)
                for s in SNR[characterized == 1]:
                    self.assertGreaterEqual(s,sim.OpticalSystem.observingModes[0]['SNR'])
                
                self.assertLessEqual(intTime,sim.OpticalSystem.intCutoff)

    def test_calc_signal_noise(self):
        r"""Test calc_signal_noise method.

        Approach: Ensure that signal is greater than noise for dummy planet
        """

        for mod in self.allmods:
            if 'calc_signal_noise' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                S,N = sim.calc_signal_noise(np.array([0]), np.array([0]), 1.0*u.d, sim.OpticalSystem.observingModes[0],\
                        fZ = np.array([0.0])/u.arcsec**2, fEZ=np.array([0.0])/u.arcsec**2, dMag=np.array([20]), WA=np.array([0.5])*u.arcsec)

                self.assertGreaterEqual(S,N)

    def test_generate_fZ(self):
        r"""Test generate fZ method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                #Check if File Exists and if it does, delete it
                if os.path.isfile(sim.cachefname+'starkfZ'):
                    os.remove(sim.cachefname+'starkfZ')
                sInds = np.asarray([0])
                sim.fZ_startSaved = sim.generate_fZ(sInds)
                self.assertEqual(sim.fZ_startSaved.shape[0],1)

    def test_calcfZmax(self):
        """Test calcfZmax method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                #Check if File Exists and if it does, delete it
                if os.path.isfile(sim.cachefname+'starkfZ'):
                    os.remove(sim.cachefname+'starkfZ')
                sInds = np.arange(5)
                sim.fZ_startSaved = sim.generate_fZ(sInds)
                val = np.zeros(sInds.shape[0])
                inds = np.zeros(sInds.shape[0])
                [val, inds] = sim.calcfZmax(sInds)
                try:
                    self.assertIsInstance(inds, type(np.asarray([])))
                except:
                    self.assertIsInstance(inds, type(None))

    def test_calcfZmin(self):
        r"""Test calcfZmin method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                #Check if File Exists and if it does, delete it
                if os.path.isfile(sim.cachefname+'starkfZ'):
                    os.remove(sim.cachefname+'starkfZ')
                sInds = np.asarray([0])
                sim.fZ_startSaved = sim.generate_fZ(sInds)
                [val, inds] = sim.calcfZmin(sInds)
                try:
                    self.assertIsInstance(inds, type(np.asarray([])))
                except:
                    self.assertIsInstance(inds, type(None))

    def test_revisitFilter(self):
        r"""Test revisitFilter method
        """
        for mod in self.allmods:
            if 'choose_revisit_target' in mod.__dict__:

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(scriptfile=self.script)

                sInds = np.asarray([0])
                tovisit = np.zeros(sim.TargetList.nStars, dtype=bool)
                sim.revisitFilter(sInds,sim.TimeKeeping.currentTimeNorm)
                try:
                    self.assertIsInstance(sInds, np.ndarray)
                except:
                    self.assertIsInstance(sInds, type(list()))