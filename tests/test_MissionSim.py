#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""MissionSim module unit tests

Michael Turmon, JPL, Apr. 2016
"""

import sys
import os
import json
import logging
import StringIO
import unittest
from EXOSIMS.MissionSim import MissionSim
import numpy as np
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Utilities import assertMethodIsCalled


SimpleScript = resource_path('test-scripts/simplest.json')
ErrorScript = resource_path('test-scripts/simplest-error.json')

class TestMissionSimMethods(unittest.TestCase):
    r"""Test MissionSim class."""

    # allow the chatter on stdout during object creation to be suppressed
    dev_null = open(os.devnull, 'w')

    # these modules are required to be in the mission module list,
    # and in the outspec module list
    required_modules = [
            'BackgroundSources', 'Completeness', 'Observatory', 'OpticalSystem',
            'PlanetPhysicalModel', 'PlanetPopulation', 'PostProcessing', 
            'SimulatedUniverse', 'SurveyEnsemble', 'SurveySimulation',
            'TargetList', 'TimeKeeping', 'ZodiacalLight' ]

    def setUp(self):
        # print '[setup] ',
        self.fixture = MissionSim

    def tearDown(self):
        del self.fixture

    def validate_object(self, mission):
        r"""Basic validation of mission object.

        Just a helper method which is used below a couple of times."""
        self.assertEqual(mission._modtype, 'MissionSim')
        self.assertEqual(type(mission._outspec), type({}))
        # check for presence of class attributes
        #   `modules' is the key for this class
        self.assertIn('modules', mission.__dict__)
        # ensure the required modules are in the mission object
        for module in self.required_modules:
            # 1: the module must be in the modules list
            self.assertIn(module, mission.modules)
            # 2: more controversially, the module must have been raised to be an
            # attribute of the mission object (e.g., mission.Observatory must
            # exist).
            self.assertIn(module, mission.__dict__)

    def validate_outspec(self, outspec, mission):
        r"""Validation of an output spec dictionary.

        Ensures that all _outspec keys (for each module) are present in
        the pooled outspec.  Also, for some value types, ensures the
        values are correct as well.
        This helper method is used below a couple of times."""
        self.assertIsInstance(outspec, dict)
        # enforce a couple more fundamental ones to be sure the outspec is OK
        for key in ['dMagLim', 'IWA', 'OWA', 'duration']:
            self.assertIn(key, outspec)
        #  `modules' must be in this dictionary
        self.assertIn('modules', outspec)
        # ensure each module name is in the outspec
        for module in self.required_modules:
            self.assertIn(module, outspec['modules'])
        # check that all individual module _outspec keys are in the outspec
        for module in mission.modules.values():
            for (key, value) in module._outspec.items():
                # 1: key is in the outspec
                self.assertIn(key, outspec)
                # 2: value matches in at least some cases
                if isinstance(value, (int, float, str)):
                    self.assertEqual(value, outspec[key])

    def test_init(self):
        r"""Test of initialization and __init__.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(SimpleScript)
        self.validate_object(mission)
            
    def test_init_fail(self):
        r"""Test of initialization and __init__ -- failure.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            with self.assertRaises(ValueError):
                mission = self.fixture(ErrorScript)
            
    def test_init_specs(self):
        r"""Test of initialization and __init__ -- specs dictionary.
        """
        script = open(SimpleScript).read()
        specs = json.loads(script)
        self.assertEqual(type(specs), type({}))
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(scriptfile=None, **specs)
        self.validate_object(mission)
            
    def test_init_file_no_file(self):
        r"""Test __init__ file handling -- various non-existent input files.
        """
        bad_files = ['/dev/null', '/tmp/this/file/is/not/there.json', '/tmp/this_file_is_not_there.json']
        for bad_file in bad_files:
            with self.assertRaises(AssertionError):
                mission = self.fixture(bad_file)
            
    def test_init_file_none(self):
        r"""Test __init__ file handling -- incomplete specs.
        
        Note that None is different than a non-existent file.
        """
        with self.assertRaises(ValueError):
            mission = self.fixture(None)
            
    def test_start_logging(self):
        r"""Test start_logging method.

        Method: Create a MissionSim with logging turned on.  Ensure the system logging.*
        methods are called correctly.  Ensure the log file exists and that it has an
        appropriate message in it already.
        """
        logfile = '/tmp/dummy_log_file.out'
        if os.path.isfile(logfile):
            os.remove(logfile)
        specs = {'logfile': logfile}
        try:
            with assertMethodIsCalled(logging, "FileHandler") as logging_mock:
                with RedirectStreams(stdout=self.dev_null):
                    mission = self.fixture(scriptfile=SimpleScript, **specs)
                self.validate_object(mission)
                del mission
                # ensure logging.FileHandler was called once, with the given file provided
                self.assertEqual(len(logging_mock.method_args), 1, 'Logger not opened.')
                self.assertEqual(len(logging_mock.method_args[0]), 1, 'Logger arguments incorrect.')
                self.assertEqual(logging_mock.method_args[0][0], logfile)
            # check the log file itself
            self.assertTrue(os.path.isfile(logfile), "Could not find log file `%s'" % logfile)
            sentinel = 'Starting log.'
            match = False
            with open(logfile) as fp:
                for l in fp.readlines():
                    if sentinel in l:
                        match = True
            self.assertTrue(match, "Found no evidence of log text `%s' in file `%s'" % (sentinel, logfile))
        finally:
            os.remove(logfile)
                                
    def test_start_logging_failure(self):
        r"""Test __init__ log file handling - failure.

        Method: Log to a file that cannot be created.  This is not an error, but
        produces a message to stdout.  Ensure the message is printed.
        """
        logfile = '/tmp/does/not/exist/dummy.out'
        specs = {'logfile': logfile}
        chatter = StringIO.StringIO()
        with RedirectStreams(stdout=chatter):
            mission = self.fixture(scriptfile=SimpleScript, **specs)
        self.assertTrue('Failed to open logfile' in chatter.getvalue())
        self.assertTrue(logfile in chatter.getvalue())
        self.assertFalse(os.path.isfile(logfile), "Unexpectedly-found log file `%s'" % logfile)
            
    # unittest.skip("Skipping random_seed_initialize.")
    def test_random_seed_initialize(self):
        r"""Test random_seed_initialize method (basic).

        Method: Ensure the MissionSim __init__ accepts the {seed: value} keyword.
        """
        seed = 123456
        specs = {'seed': seed}
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(scriptfile=SimpleScript, **specs)
        self.validate_object(mission)

    def test_random_seed_initialize_full(self):
        r"""Test random_seed_initialize method (longer).

        Method: Initialize with a specific random seed, and ensure the numpy
        RNG seeding mechanism was indeed called by the __init__ function.
        """
        # Note: the mission.__init__() method, in building up planets, etc., calls
        # the numpy RNG many times.  All these calls make it impossible to check the 
        # RNG state just after the object is again available to this test code.
        # So, to test, we have to ensure that np.random.seed is
        # called correctly instead of checking the RNG state after __init__ returns.

        # set up the seed, and the specs addition that includes it
        seed = 1234567890
        specs = {'seed': seed}

        # plugs in our own mock object to monitor calls to np.random.seed()
        with assertMethodIsCalled(np.random, "seed") as rng_seed_mock:
            # initialize the object
            with RedirectStreams(stdout=self.dev_null):
                mission = self.fixture(scriptfile=SimpleScript, **specs)
            # ensure np.random.seed was called once, with the given seed provided
            self.assertEqual(len(rng_seed_mock.method_args), 1, 'RNG was not seeded.')
            self.assertEqual(len(rng_seed_mock.method_args[0]), 1, 'RNG seed arguments incorrect.')
            self.assertEqual(rng_seed_mock.method_args[0][0], seed)
            # keyword argument should be an empty dictionary
            self.assertEqual(len(rng_seed_mock.method_kwargs), 1, 'RNG seeded with unexpected arguments.')
            self.assertDictEqual(rng_seed_mock.method_kwargs[0], {}, 'RNG seeded with unexpected arguments.')
        # might as well validate the object
        self.validate_object(mission)

    def test_genoutspec(self):
        r"""Test of the genOutSpec method (results output).

        Method: This is the key test of genOutSpec.  We set up a mission
        object, write it to JSON using genOutSpec, and then both check the
        compiled outspec dictionary, and re-load the generated JSON and test
        it back against the mission object.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(SimpleScript)
        self.validate_object(mission)
        out_filename = '/tmp/dummy_gen_outspec.json'
        outspec_orig = mission.genOutSpec(out_filename)
        # ensure the compiled outspec is correct and complete
        self.validate_outspec(outspec_orig, mission)
        # ensure the JSON file was written
        self.assertTrue(os.path.isfile(out_filename), "Could not find outspec file `%s'" % out_filename)
        # re-load the JSON file
        with open(out_filename, 'r') as fp:
            outspec_new = json.load(fp)
        # ensure all keys are present
        self.assertListEqual(sorted(outspec_orig.keys()),
                             sorted(outspec_new.keys()))
        # furthermore, ensure the re-loaded outspec is OK
        # this is a rather stringent test
        self.validate_outspec(outspec_new, mission)
        os.remove(out_filename)
        
    def test_genoutspec_badfile(self):
        r"""Test of the genOutSpec method (bad filename).

        Method: This test ensures that bad filenames cause an exception.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(SimpleScript)
        self.validate_object(mission)
        out_filename = '/tmp/file/cannot/be/written/spec.json'
        with self.assertRaises(IOError):
            mission.genOutSpec(out_filename)
            
    def test_genoutspec_nofile(self):
        r"""Test of the genOutSpec method (empty filename).

        Method: This checks that the compiled outspec dictionary
        is correct and complete.  No file is written.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(SimpleScript)
        self.validate_object(mission)
        # compile the output spec
        outspec_orig = mission.genOutSpec(None)
        # ensure output spec is OK
        self.validate_outspec(outspec_orig, mission)
            
    
if __name__ == '__main__':
    unittest.main()
