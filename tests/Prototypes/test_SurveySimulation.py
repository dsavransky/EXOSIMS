#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""SurveySimulation module unit tests

Michael Turmon, JPL, Apr. 2016
"""

import sys
import os
import unittest
import StringIO
from collections import namedtuple
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np
import astropy.units as u
from astropy.time import Time
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams

SimpleScript = resource_path() + '/test-scripts/simplest.json'
# SimpleScript = 'test-scripts/simplest.json'

class TestSurveySimulationMethods(unittest.TestCase):
    r"""Test SurveySimulation class."""
    dev_null = open(os.devnull, 'w')

    def setUp(self):
        # print '[setup] ',
        self.fixture = SurveySimulation

    def tearDown(self):
        del self.fixture

    # @unittest.skip("Skipping init.")
    def test_init(self):
        r"""Test of initialization and __init__.
        """
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        self.assertEqual(sim._modtype, 'SurveySimulation')
        self.assertEqual(type(sim._outspec), type({}))
        # check for presence of a couple of class attributes
        self.assertIn('DRM', sim.__dict__)
        self.assertIn('OpticalSystem', sim.__dict__)

    #@unittest.skip("Skipping str.")
    def test_str(self):
        r"""Test __str__ method, for full coverage."""
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        # replace stdout and keep a reference
        original_stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        # call __str__ method
        result = sim.__str__()
        # examine what was printed
        contents = sys.stdout.getvalue()
        self.assertEqual(type(contents), type(''))
        self.assertIn('DRM', contents)
        sys.stdout.close()
        # it also returns a string, which is not necessary
        self.assertEqual(type(result), type(''))
        # put stdout back
        sys.stdout = original_stdout

    # @unittest.skip("Skipping run_sim.")
    def test_run_sim(self):
        r"""Test run_sim method.

        Approach: Ensures the simulation runs to completion and the output is set.
        """

        print 'run_sim()'
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        sim.run_sim()
        #print 'TargetList is', sim.TargetList
        if False:
            print 'PlanetPopulation is as follows:\n', sim.PlanetPopulation
            print 'SimulatedUniverse is as follows:\n', sim.SimulatedUniverse
        #print 'Time is', sim.TimeKeeping.currentTimeNorm
        #print 'DRM is', sim.DRM
        # check that the mission time is indeed elapsed
        self.assertGreaterEqual(sim.TimeKeeping.currentTimeNorm,
                                sim.TimeKeeping.missionFinishNorm)
        # resulting DRM is a list...
        self.assertIsInstance(sim.DRM, list)
        # ...and has nontrivial number of entries
        self.assertGreater(len(sim.DRM), 0)
        

    @unittest.skip("Skip initial target.")
    def test_initial_target(self):
        r"""Test initial_target method.

        Approach: Ensure the initial target selected OK, and is a valid integer.
        """

        print 'initial_target'
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        # allowable index range
        Nstar = sim.TargetList.Name.size

        # simplify: use run_sim to set up the sim object
        #sim.run_sim()
        #print 'Time ended at:', sim.TimeKeeping.currentTimeNorm, ' -- Now resetting.'
        #sim.TimeKeeping.currentTimeNorm = 0.0 * u.day

        sInd = sim.initial_target()
        # result is a scalar numpy ndarray.
        # It is an integer, in a valid range
        self.assertIsInstance(sInd, np.ndarray)
        self.assertEqual(sInd.size, 1)
        self.assertIsInstance(int(sInd), int)
        self.assertEqual(sInd - int(sInd), 0)
        self.assertGreaterEqual(sInd, 0)
        self.assertLess(sInd, Nstar)
    
    # @unittest.skip("Skip next target.")
    def test_next_target(self):
        r"""Test next_target method.

        Approach: Ensure the next target is selected OK, and is a valid integer.
        Deficiencies: We are not checking that the occulter slew works.
        """

        print 'next_target'
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        # allowable index range
        Nstar = sim.TargetList.Name.size

        DRM_in = {}
        (sInd, DRM_out) = sim.next_target(0, DRM_in)

        # result index is a scalar numpy ndarray, that is a valid integer
        # in a valid range
        self.assertIsInstance(sInd, int)
        # self.assertEqual(sInd.size, 1)
        # self.assertIsInstance(int(sInd), int)
        self.assertEqual(sInd - int(sInd), 0)
        self.assertGreaterEqual(sInd, 0)
        self.assertLess(sInd, Nstar)

        # resulting DRM is a dictionary -- contents unimportant
        self.assertIsInstance(DRM_out, dict)

    @unittest.skip("Skip observation_detection.")
    def test_observation_detection(self):
        r"""Test observation_detection method.

        Approach:
        """

        print 'observation_detection'
        sim = self.fixture(SimpleScript)
        # allowable index range
        Nstar = sim.TargetList.Name.size

        # FIXME: with empty pInds, is not exercising all functionality
        # Need to get some pInds in the range 0...sim.SimulatedUniverse.nPlans-1
        pInds = np.array([False] * sim.SimulatedUniverse.nPlans)
        pInds[0] = True
        sInd = np.array([0])
        DRM_in = {}
        planPosTime = np.array([[0]] * sim.SimulatedUniverse.nPlans)
        obs_poss, t_int, DRM_out = sim.observation_detection(pInds, sInd, DRM_in, planPosTime)

        # FIXME: not correct -- in general, length depends on entries in pInds
        self.assertIsInstance(obs_poss, bool)

        # integration time, one for each star
        self.assertIsInstance(t_int, np.ndarray)
        self.assertEqual(t_int.size, Nstar)
        self.assertTrue(np.all(t_int >= 0))

        # resulting DRM is a dictionary -- contents unimportant
        self.assertIsInstance(DRM_out, dict)

    @unittest.skip("Skip observation_characterization")
    def test_observation_characterization(self):
        r"""Test observation_characterization method.

        Approach:
        """

        print 'observation_characterization'
        sim = self.fixture(SimpleScript)
        # allowable index range
        Nstar = sim.TargetList.Name.size

        # with empty pInds, is not exercising all functionality
        obs_poss = np.array([])
        pInds = np.array([])
        sInd = np.array([0])
        spectra_in = np.array([])
        DRM_in = {}
        FA_in = False
        t_int = np.array([0.0]) * u.day 
        DRM_out, FA_out, spectra_out = sim.observation_characterization(
            obs_poss, pInds, sInd, spectra_in, DRM_in, FA_in, t_int)

        # resulting DRM is a dictionary -- contents unimportant
        self.assertIsInstance(DRM_out, dict)
        # False Alarm indicator
        self.assertIsInstance(FA_out, bool)
        # spectra: TBD
        self.assertIsInstance(spectra_out, np.ndarray)


    @unittest.skip("Skip det_data")
    def test_det_data(self):
        r"""Test det_data method.

        Approach: Check contents of returned dictionary
        """

        print 'det_data'
        sim = self.fixture(SimpleScript)

        s_in = 0.0
        DRM_in = {}
        FA = False
        DET = True
        MD = False
        sInd = np.array([0])
        pInds = np.array([])
        obs_poss = np.array([])
        observed = np.array([0])

        (s_out, DRM_out, observed) = sim.det_data(
            s_in, DRM_in, FA, DET, MD, sInd, pInds, obs_poss, observed)

        # resulting DRM is a dictionary...
        self.assertIsInstance(DRM_out, dict)
        # ...and contains 'det_status'
        self.assertIn('det_status', DRM_out)

    # @unittest.skip("Skip check_visible_end")
    def test_check_visible_end(self):
        r"""Test check_visible_end method.

        Approach:
        """
        pass

    # @unittest.skip("Skip sys_end_res")
    def test_sys_end_res(self):
        r"""Test sys_end_res method.

        Approach:
        """
        pass
    
if __name__ == '__main__':
    unittest.main()
