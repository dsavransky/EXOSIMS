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
import json
from collections import namedtuple
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np
import astropy.units as u
from astropy.time import Time
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import pdb

SimpleScript = resource_path('test-scripts/simplest.json')
ErrorScript = resource_path('test-scripts/simplest-error.json')

class TestSurveySimulationMethods(unittest.TestCase):
    r"""Test SurveySimulation class."""
    dev_null = open(os.devnull, 'w')

    required_modules = [
            'BackgroundSources', 'Completeness', 'Observatory', 'OpticalSystem',
            'PlanetPhysicalModel', 'PlanetPopulation', 'PostProcessing', 
            'SimulatedUniverse', 'TargetList', 'TimeKeeping', 'ZodiacalLight' ]

    def setUp(self):
        # print '[setup] ',
        self.fixture = SurveySimulation

    def tearDown(self):
        del self.fixture


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

    def test_init_fail(self):
        r"""Test of initialization and __init__ -- failure.
        """
        with RedirectStreams(stdout=self.dev_null):
            with self.assertRaises(ValueError):
                sim = self.fixture(ErrorScript)
            
    def test_init_specs(self):
        r"""Test of initialization and __init__ -- specs dictionary.
        """
        script = open(SimpleScript).read()
        specs = json.loads(script)
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(scriptfile=None, **specs)
        
        for rmod in self.required_modules:
            self.assertIn(rmod, sim.__dict__)
            self.assertEqual(getattr(sim,rmod)._modtype,rmod)
            
    def test_init_file_no_file(self):
        r"""Test __init__ file handling -- various non-existent input files.
        """
        bad_files = ['/dev/null', '/tmp/this/file/is/not/there.json', '/tmp/this_file_is_not_there.json']
        for bad_file in bad_files:
            with self.assertRaises(AssertionError):
                sim = self.fixture(bad_file)
            
    def test_init_file_none(self):
        r"""Test __init__ file handling -- incomplete specs.
        
        Note that None is different than a non-existent file.
        """
        with self.assertRaises(KeyError):
            sim = self.fixture(None)

    def test_choose_next_target(self):
        r"""Test choose_next_target method.

        Approach: Ensure the next target has max completeness as required in prototype
        """
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        
        #to make this non-trivial, overwrite comp0 with random values:
        comprand = np.random.rand(sim.TargetList.nStars)
        sim.TargetList.comp0 = comprand.copy()
        sInd = sim.choose_next_target(None,np.arange(sim.TargetList.nStars),np.array([1.0]*sim.TargetList.nStars)*u.d,\
                np.array([1.0]*sim.TargetList.nStars)*u.d)
        self.assertEqual(comprand[sInd],comprand.max())

    def test_reset_sim(self):
        r"""Test reset_sim method.

        Approach: Ensures the simulation resets completely
        """

        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
            sim.run_sim()
        
        self.assertGreater(len(sim.DRM), 0)
        self.assertGreater(sim.TimeKeeping.currentTimeNorm,0.0*u.d)

        sim.reset_sim()
        self.assertEqual(sim.TimeKeeping.currentTimeNorm,0.0*u.d)
        self.assertEqual(len(sim.DRM), 0)

    def validate_outspec(self, outspec, sim):
        r"""Validation of an output spec dictionary.

        Ensures that all _outspec keys (for each module) are present in
        the pooled outspec.  Also, for some value types, ensures the
        values are correct as well.
        This helper method is used below a couple of times."""
        self.assertIsInstance(outspec, dict)
        # enforce a couple more fundamental ones to be sure the outspec is OK
        for key in ['dMagLim', 'IWA', 'OWA', 'OBduration']:
            self.assertIn(key, outspec)
        #  modules' must be in this dictionary
        self.assertIn('modules', outspec)
        # ensure each module name is in the outspec
        for module in self.required_modules:
            self.assertIn(module, outspec['modules'])
        # check that all individual module _outspec keys are in the outspec
        for module in sim.modules.values():
            for (key, value) in module._outspec.items():
                # 1: key is in the outspec
                self.assertIn(key, outspec)
                # 2: value matches in at least some cases
                if isinstance(value, (int, float, str)):
                    self.assertEqual(value, outspec[key])
    
    def test_genoutspec(self):
        r"""Test of the genOutSpec method (results output).

        Method: This is the key test of genOutSpec.  We set up a sim
        object, write it to JSON using genOutSpec, and then both check the
        compiled outspec dictionary, and re-load the generated JSON and test
        it back against the sim object.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
 
        out_filename = 'dummy_gen_outspec.json'
        outspec_orig = sim.genOutSpec(out_filename)
        # ensure the compiled outspec is correct and complete
        self.validate_outspec(outspec_orig, sim)
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
        self.validate_outspec(outspec_new, sim)
        os.remove(out_filename)
        
    def test_genoutspec_badfile(self):
        r"""Test of the genOutSpec method (bad filename).

        Method: This test ensures that bad filenames cause an exception.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        out_filename = '/tmp/file/cannot/be/written/spec.json'
        with self.assertRaises(IOError):
            sim.genOutSpec(out_filename)
            
    def test_genoutspec_nofile(self):
        r"""Test of the genOutSpec method (empty filename).

        Method: This checks that the compiled outspec dictionary
        is correct and complete.  No file is written.
        """
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
        # compile the output spec
        outspec_orig = sim.genOutSpec(None)
        # ensure output spec is OK
        self.validate_outspec(outspec_orig, sim)

if __name__ == '__main__':
    unittest.main()
