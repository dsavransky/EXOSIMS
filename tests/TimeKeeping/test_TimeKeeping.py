import unittest
from tests.TestSupport.Info import resource_path
import EXOSIMS
import EXOSIMS.Prototypes.TimeKeeping
import EXOSIMS.TimeKeeping
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import pkgutil
import os
import json
import sys
import copy
from EXOSIMS.util.get_module import get_module
from astropy.time import Time
import astropy.units as u
from tests.TestSupport.Utilities import RedirectStreams
from io import StringIO


class TestTime(unittest.TestCase):
    """

    Global TimeKeeping tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

        modtype = getattr(EXOSIMS.Prototypes.TimeKeeping.TimeKeeping, "_modtype")
        pkg = EXOSIMS.TimeKeeping
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

        self.script1 = resource_path("test-scripts/simplest.json")

        everymodtype = getattr(SurveySimulation, "_modtype")
        self.everymods = [get_module(everymodtype)]

    def tearDown(self):
        self.dev_null.close()

    def test_init(self):
        """
        Test of initialization and __init__.
        """

        req_atts = [
            "missionStart",
            "missionPortion",
            "missionLife",
            "missionFinishAbs",
            "currentTimeNorm",
            "currentTimeAbs",
            "OBnumber",
            "OBduration",
            "OBstartTimes",
            "OBendTimes",
            "cachedir",
        ]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

            # verify that all attributes are there
            for att in req_atts:
                self.assertTrue(
                    hasattr(obj, att),
                    "Missing attribute {} for {}".format(att, mod.__name__),
                )

    def test_init_OB(self):
        """
        Test OB method to ensure proper output and change of attributes. Assumes
        (for now) that the csv file formatted the same for all inputs. Checks
        to see that the outputs have the correct types.
        """

        OBduration = 10

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

                # type(r_sc), type(1.0 * u.km)

                # 1) Load Observing Blocks from File
                # File Located in: EXOSIMS/EXOSIMS/Scripts/sampleOB.csv
                obj.init_OB("sampleOB.csv", OBduration * u.d)
                self.assertEqual(type(obj.OBduration), type(OBduration * u.d))
                self.assertEqual(type(obj.OBnumber), type(1))
                self.assertEqual(type(obj.OBstartTimes), type([0] * u.d))
                self.assertEqual(type(obj.OBendTimes), type([0] * u.d))

                # 2) Automatically construct OB from OBduration, missionLife,
                # and missionPortion
                OBduration = 10
                obj.missionLife = 100 * u.d
                obj.missionPortion = 0.1
                obj.init_OB(str(None), OBduration * u.d)
                self.assertEqual(type(obj.OBduration), type(OBduration * u.d))
                self.assertEqual(type(obj.OBnumber), type(1))
                self.assertEqual(type(obj.OBstartTimes), type([0] * u.d))
                self.assertEqual(type(obj.OBendTimes), type([0] * u.d))

    def test_allocate_time(self):
        """
        Test allocate_time to ensure proper output
        """

        dt = 10 * u.d
        timeString = "2010-01-01 00:00:00"
        arbTime = Time(timeString)

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

                addExoplanetObsTime = False
                self.assertEqual(
                    type(obj.allocate_time(dt, addExoplanetObsTime)), type(False)
                )
                self.assertEqual(type(obj.currentTimeNorm), type(1 * u.d))
                self.assertEqual(type(obj.currentTimeAbs), type(arbTime))
                self.assertEqual(type(obj.exoplanetObsTime), type(1 * u.d))

                addExoplanetObsTime = True
                self.assertEqual(
                    type(obj.allocate_time(dt, addExoplanetObsTime)), type(False)
                )
                self.assertEqual(type(obj.currentTimeNorm), type(1 * u.d))
                self.assertEqual(type(obj.currentTimeAbs), type(arbTime))
                self.assertEqual(type(obj.exoplanetObsTime), type(1 * u.d))

    def test_advanceToAbsTime(self):
        """
        Test advanceToAbsTime to ensure proper output
        """

        timeString = "2010-01-01 00:00:00"
        arbTime = Time(timeString)
        # arbTime.format = 'mjd'

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

                addExoplanetObsTime = False
                self.assertEqual(
                    type(obj.advanceToAbsTime(arbTime, addExoplanetObsTime)),
                    type(False),
                )
                self.assertEqual(type(obj.currentTimeNorm), type(1 * u.d))
                self.assertEqual(type(obj.currentTimeAbs), type(arbTime))
                self.assertEqual(type(obj.exoplanetObsTime), type(1 * u.d))

                self.assertEqual(
                    type(obj.advanceToAbsTime(arbTime, addExoplanetObsTime)),
                    type(False),
                )
                self.assertEqual(type(obj.currentTimeNorm), type(1 * u.d))
                self.assertEqual(type(obj.currentTimeAbs), type(arbTime))
                self.assertEqual(type(obj.exoplanetObsTime), type(1 * u.d))

    def test_advancetToStartOfNextOB(self):
        """
        Test advancetToStartOfNextOB to ensure proper output types.
        """

        timeString = "2010-01-01 00:00:00"
        arbTime = Time(timeString)

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

                obj.OBstartTimes = [0, 10, 20, 30] * u.d
                obj.OBendTimes = [5, 15, 25, 35] * u.d
                obj.OBnumber = 0

                obj.advancetToStartOfNextOB()

                self.assertEqual(type(obj.OBnumber), type(1))
                self.assertEqual(type(obj.currentTimeNorm), type(1 * u.d))
                self.assertEqual(type(obj.currentTimeAbs), type(arbTime))

    def test_mission_is_over(self):
        """
        Test mission_is_over method to ensure proper output types.

        """

        sim = self.everymods[0](scriptfile=self.script1)
        allModes = sim.OpticalSystem.observingModes
        Obs = sim.Observatory
        OS = sim.OpticalSystem
        det_mode_list = list(filter(lambda mode: mode["detectionMode"], allModes))

        for mod in self.allmods:
            for det_mode in det_mode_list:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**copy.deepcopy(self.spec))

                    self.assertEqual(
                        type(obj.mission_is_over(OS, Obs, det_mode)), type(False)
                    )

    def test_get_ObsDetectionMaxIntTime(self):
        """
        Test get_ObsDetectionMaxIntTime method to ensure proper output types.
        """

        sim = self.everymods[0](scriptfile=self.script1)
        allModes = sim.OpticalSystem.observingModes
        Obs = sim.Observatory
        det_mode_list = list(filter(lambda mode: mode["detectionMode"], allModes))

        for mod in self.allmods:
            for det_mode in det_mode_list:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**copy.deepcopy(self.spec))

                    self.assertEqual(
                        type(obj.get_ObsDetectionMaxIntTime(Obs, det_mode)[0]),
                        type(1 * u.d),
                    )
                    self.assertEqual(
                        type(obj.get_ObsDetectionMaxIntTime(Obs, det_mode)[1]),
                        type(1 * u.d),
                    )
                    self.assertEqual(
                        type(obj.get_ObsDetectionMaxIntTime(Obs, det_mode)[2]),
                        type(1 * u.d),
                    )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """

        atts_list = [
            "missionStart",
            "missionPortion",
            "missionLife",
            "missionFinishAbs",
            "currentTimeNorm",
            "currentTimeAbs",
            "OBnumber",
            "OBduration",
            "OBstartTimes",
            "OBendTimes",
            "cachedir",
        ]
        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
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
