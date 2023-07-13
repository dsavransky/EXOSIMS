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

import os
import json
import unittest
from EXOSIMS.MissionSim import MissionSim
import numpy as np
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Utilities import assertMethodIsCalled


SimpleScript = resource_path("test-scripts/simplest.json")
ErrorScript = resource_path("test-scripts/simplest-error.json")


class TestMissionSimMethods(unittest.TestCase):
    r"""Test MissionSim class."""

    # these modules are required to be in the mission module list,
    # and in the outspec module list
    required_modules = [
        "BackgroundSources",
        "Completeness",
        "Observatory",
        "OpticalSystem",
        "PlanetPhysicalModel",
        "PlanetPopulation",
        "PostProcessing",
        "SimulatedUniverse",
        "SurveyEnsemble",
        "SurveySimulation",
        "TargetList",
        "TimeKeeping",
        "ZodiacalLight",
    ]

    def setUp(self):
        # print '[setup] ',
        self.fixture = MissionSim
        # allow the chatter on stdout during object creation to be suppressed
        self.dev_null = open(os.devnull, "w")

    def tearDown(self):
        del self.fixture
        self.dev_null.close()

    def validate_object(self, mission):
        r"""Basic validation of mission object.

        Just a helper method which is used below a couple of times."""
        self.assertEqual(mission._modtype, "MissionSim")
        self.assertEqual(type(mission._outspec), type({}))
        # check for presence of class attributes
        #   `modules' is the key for this class
        self.assertIn("modules", mission.__dict__)
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
        for key in ["intCutoff_dMag", "IWA", "OWA", "duration"]:
            self.assertIn(key, outspec)
        #  `modules' must be in this dictionary
        self.assertIn("modules", outspec)
        # ensure each module name is in the outspec
        for module in self.required_modules:
            self.assertIn(module, outspec["modules"])
        # check that all individual module _outspec keys are in the outspec
        for module in mission.modules.values():
            for (key, value) in module._outspec.items():
                # 1: key is in the outspec
                self.assertIn(key, outspec)
                # 2: value matches in at least some cases
                if isinstance(value, (int, float, str)):
                    self.assertEqual(value, outspec[key])

    def test_init(self):
        r"""Test of initialization and __init__."""
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(SimpleScript)
        self.validate_object(mission)

    def test_init_fail(self):
        r"""Test of initialization and __init__ -- failure."""
        # the with clause allows the chatter on stdout during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null):
            with self.assertRaises(ValueError):
                mission = self.fixture(ErrorScript)

    def test_init_specs(self):
        r"""Test of initialization and __init__ -- specs dictionary."""
        with open(SimpleScript) as script:
            specs = json.loads(script.read())
        self.assertEqual(type(specs), type({}))
        with RedirectStreams(stdout=self.dev_null):
            mission = self.fixture(scriptfile=None, **specs)
        self.validate_object(mission)

    def test_init_file_no_file(self):
        r"""Test __init__ file handling -- various non-existent input files."""
        bad_files = [
            "/dev/null",
            "/tmp/this/file/is/not/there.json",
            "/tmp/this_file_is_not_there.json",
        ]
        for bad_file in bad_files:
            with self.assertRaises(AssertionError):
                mission = self.fixture(bad_file)

    def test_init_file_none(self):
        r"""Test __init__ file handling -- incomplete specs.

        Note that None is different than a non-existent file.
        """
        with self.assertRaises(ValueError):
            mission = self.fixture(None)

    def test_random_seed_initialize(self):
        r"""Test random_seed_initialize method (basic).

        Method: Ensure the MissionSim __init__ accepts the {seed: value} keyword.
        """
        seed = 123456
        specs = {"seed": seed}
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
        specs = {"seed": seed}

        # plugs in our own mock object to monitor calls to np.random.seed()
        with assertMethodIsCalled(np.random, "seed") as rng_seed_mock:
            # initialize the object
            with RedirectStreams(stdout=self.dev_null):
                mission = self.fixture(scriptfile=SimpleScript, **specs)
            # ensure np.random.seed was called once, with the given seed provided
            self.assertEqual(len(rng_seed_mock.method_args), 1, "RNG was not seeded.")
            self.assertEqual(
                len(rng_seed_mock.method_args[0]), 1, "RNG seed arguments incorrect."
            )
            self.assertEqual(rng_seed_mock.method_args[0][0], seed)
            # keyword argument should be an empty dictionary
            self.assertEqual(
                len(rng_seed_mock.method_kwargs),
                1,
                "RNG seeded with unexpected arguments.",
            )
            self.assertDictEqual(
                rng_seed_mock.method_kwargs[0],
                {},
                "RNG seeded with unexpected arguments.",
            )
        # might as well validate the object
        self.validate_object(mission)

    def test_reset_sim(self):
        r"""Test reset_sim method.

        Approach: Ensures the simulation resets completely
        """

        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
            sim.run_sim()

        self.assertGreater(len(sim.SurveySimulation.DRM), 0)
        self.assertGreater(sim.TimeKeeping.currentTimeNorm.value, 0.0)

        sim.reset_sim()
        self.assertEqual(sim.TimeKeeping.currentTimeNorm.value, 0.0)
        self.assertEqual(len(sim.SurveySimulation.DRM), 0)

    def test_run_ensemble(self):
        r"""Test run_ensemble method.

        Approach: Ensures that ensemble runs and returns all DRM dicts
        """

        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
            n = 10
            res = sim.run_ensemble(n)

        self.assertEqual(len(res), n)

    def test_DRM2array(self):
        """Test DRM2array method.

        Approach: Ensure that array is properly generated for known keys
        """
        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
            sim.run_sim()

        res = sim.DRM2array("star_ind")
        self.assertIsInstance(res, np.ndarray)

        res = sim.DRM2array("plan_inds")
        self.assertIsInstance(res, np.ndarray)

        res = sim.DRM2array("FA_det_status")
        self.assertIsInstance(res, np.ndarray)

        res = sim.DRM2array("det_WA", DRM=sim.SurveySimulation.DRM)
        self.assertIsInstance(res, np.ndarray)

        with self.assertRaises(AssertionError):
            res = sim.DRM2array("nosuchkeyexists")

    def test_filter_status(self):
        """Test filter_status method.

        Approach: Test outputs from sample run
        """

        with RedirectStreams(stdout=self.dev_null):
            sim = self.fixture(SimpleScript)
            sim.run_sim()

        res = sim.filter_status("det_SNR", 0)
        self.assertIsInstance(res, np.ndarray)

        res = sim.filter_status("det_SNR", 0, DRM=sim.SurveySimulation.DRM)
        self.assertIsInstance(res, np.ndarray)

        with self.assertRaises(AssertionError):
            res = sim.filter_status("det_SNR", 0, obsMode="nosuchmode")


if __name__ == "__main__":
    unittest.main()
