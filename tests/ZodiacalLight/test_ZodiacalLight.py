import unittest
from tests.TestSupport.Info import resource_path
import EXOSIMS
import EXOSIMS.Prototypes.ZodiacalLight
import EXOSIMS.ZodiacalLight
from EXOSIMS import MissionSim
import astropy.units as u
import pkgutil
from EXOSIMS.util.get_module import get_module
import numpy as np
import os
import json
from tests.TestSupport.Utilities import RedirectStreams
import sys
from io import StringIO
import copy


"""ZodiacalLight module unit tests
based on previous tests by Paul Nunez, JPL"""


class TestZodiacalLight(unittest.TestCase):
    """

    Global ZodiacalLight tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        # self.script = resource_path("test-scripts/template_prototype_testing.json")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())
        self.spec["ntargs"] = 10  # generate fake targets list with 10 stars

        with RedirectStreams(stdout=self.dev_null):
            self.sim = MissionSim.MissionSim(**copy.deepcopy(self.spec))
        self.TL = self.sim.TargetList
        self.nStars = self.TL.nStars
        self.star_index = np.array(range(0, self.nStars))
        self.Obs = self.sim.Observatory
        self.mode = self.sim.OpticalSystem.observingModes[0]
        self.TK = self.sim.TimeKeeping
        self.unit = 1.0 / u.arcsec**2
        self.JEZ0_unit = 1 * u.ph / u.s / u.m**2 / u.arcsec**2

        modtype = getattr(EXOSIMS.Prototypes.ZodiacalLight.ZodiacalLight, "_modtype")
        pkg = EXOSIMS.ZodiacalLight
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not is_pkg:
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_fZ(self):
        """
        Test that fZ returns correct type and units.
        """

        for mod in self.allmods:
            if "fZ" in mod.__dict__:
                obj = mod()
                fZs = obj.fZ(
                    self.Obs,
                    self.TL,
                    self.star_index,
                    self.TK.currentTimeAbs,
                    self.mode,
                )
                self.assertEqual(
                    len(fZs),
                    self.nStars,
                    "fZ does not return same number of values as nStars for {}".format(
                        mod.__name__
                    ),
                )
                self.assertEqual(
                    fZs.unit,
                    self.unit,
                    "fZ does not return 1/arcsec**2 for {}".format(mod.__name__),
                )

    def test_calc_JEZ0(self):
        """
        Test that fEZ returns correct shape and units.
        """
        exclude_mods = []

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue
            if "fEZ" in mod.__dict__:
                obj = mod()
                # use 3 System
                flambda = np.random.uniform(0.0, 5.0, 3)
                I = np.random.uniform(0.0, 180.0, 3) * u.deg
                bandwidth = 10 * u.nm
                JEZ0s = obj.calc_JEZ0(self.TL.MV[:3], I, flambda, bandwidth)
                self.assertEqual(
                    len(JEZs),
                    3,
                    (
                        "calc_JEZ0 does not return same number of values as stars tested "
                        f"for {mod.__name__}"
                    ),
                )
                self.assertEqual(
                    JEZ0s.unit,
                    self.JEZ0_unit,
                    "calc_JEZ0 does not return ph/s/m2/arcsec2 for {}".format(
                        mod.__name__
                    ),
                )

    def test_generate_fZ(self):
        """
        Test generate fZ method
        """

        for mod in self.allmods:
            if "generate_fZ" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                # Check if File Exists and if it does, delete it
                # if os.path.isfile(self.sim.SurveySimulation.cachefname+'starkfZ'):
                #    os.remove(self.sim.SurveySimulation.cachefname+'starkfZ')
                OS = self.sim.OpticalSystem
                allModes = OS.observingModes
                mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
                hashname = self.sim.SurveySimulation.cachefname
                obj.generate_fZ(self.Obs, self.TL, self.TK, mode, hashname)
                self.assertEqual(
                    self.sim.ZodiacalLight.fZMap[mode["syst"]["name"]].shape[0],
                    self.nStars,
                )
                if self.sim.SurveySimulation.koTimes is None:
                    times = self.sim.ZodiacalLight.fZTimes
                else:
                    times = self.sim.SurveySimulation.koTimes
                self.assertEqual(
                    self.sim.ZodiacalLight.fZMap[mode["syst"]["name"]].shape[1],
                    len(times),
                )  # This was arbitrarily selected.

    def test_calcfZmax(self):
        """
        Test calcfZmax method
        """

        for mod in self.allmods:
            if "calcfZmax" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()

                # Check if File Exists and if it does, delete it
                if os.path.isfile(self.sim.SurveySimulation.cachefname + "fZmax"):
                    os.remove(self.sim.SurveySimulation.cachefname + "fZmax")
                sInds = np.asarray([0])
                currentTimeAbs = self.sim.TimeKeeping.currentTimeAbs
                OS = self.sim.OpticalSystem
                allModes = OS.observingModes
                mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
                hashname = self.sim.SurveySimulation.cachefname
                obj.generate_fZ(self.Obs, self.TL, self.TK, mode, hashname)
                valfZmax = np.zeros(sInds.shape[0])
                timefZmax = np.zeros(sInds.shape[0])
                [valfZmax, timefZmax] = obj.calcfZmax(
                    sInds, self.Obs, self.TL, self.TK, mode, hashname
                )
                self.assertTrue(len(valfZmax) == len(sInds))
                self.assertTrue(len(timefZmax) == len(sInds))
                self.assertTrue(valfZmax[0].unit == self.unit)
                self.assertTrue(timefZmax[0].format == currentTimeAbs.format)

    def test_calcfZmin(self):
        """
        Test calcfZmin method
        """

        for mod in self.allmods:
            if "calcfZmin" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                sInds = np.asarray([0])
                currentTimeAbs = self.TK.currentTimeAbs
                OS = self.sim.OpticalSystem
                allModes = OS.observingModes
                mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
                hashname = self.sim.SurveySimulation.cachefname
                obj.generate_fZ(self.Obs, self.TL, self.TK, mode, hashname)
                fZmins, fZtypes = obj.calcfZmin(
                    sInds, self.Obs, self.TL, self.TK, mode, hashname
                )
                [valfZmin, timefZmin] = obj.extractfZmin(fZmins, sInds)
                self.assertTrue(len(valfZmin) == len(sInds))
                self.assertTrue(len(timefZmin) == len(sInds))
                self.assertTrue(valfZmin[0].unit == 1 / u.arcsec**2)
                self.assertTrue(timefZmin[0].format == currentTimeAbs.format)

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """
        atts_list = ["magZ", "magEZ", "varEZ", "fZ0", "fEZ0"]
        exclude_mods = []

        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue
            if mod.__name__ in exclude_mods:
                continue
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**self.spec)
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            # call __str__ method
            result = obj.__str__()
            # examine what was printed
            contents = sys.stdout.getvalue()
            self.assertEqual(type(contents), type(""))
            # attributes from ICD
            for att in atts_list:
                self.assertIn(
                    att, contents, "{} missing for {}".format(att, mod.__name__)
                )
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
