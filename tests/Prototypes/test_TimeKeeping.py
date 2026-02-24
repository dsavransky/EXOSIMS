#!/usr/local/bin/python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""TimeKeeping module unit tests
Michael Turmon, JPL, Mar/Apr 2016
"""

import unittest
from EXOSIMS.Prototypes.TimeKeeping import TimeKeeping
from tests.TestSupport.Utilities import RedirectStreams
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from tests.TestSupport.Info import resource_path
from EXOSIMS.util.get_module import get_module
import os
import numpy as np
import astropy.units as u


class TestTimeKeepingMethods(unittest.TestCase):
    r"""Test TimeKeeping class."""

    def setUp(self):
        # print '[setup] ',
        # do not instantiate it
        self.fixture = TimeKeeping

        self.dev_null = open(os.devnull, "w")
        self.script1 = resource_path("test-scripts/simplest.json")
        self.script2 = resource_path("test-scripts/simplest_initOB.json")

        modtype = getattr(SurveySimulation, "_modtype")
        self.allmods = [get_module(modtype)]

    def tearDown(self):
        self.dev_null.close()

    def test_init(self):
        r"""Test of initialization and __init__."""
        tk = self.fixture()
        self.assertEqual(tk.currentTimeNorm.to(u.day).value, 0.0)
        self.assertEqual(type(tk._outspec), type({}))
        # check for presence of one class attribute
        self.assertGreater(tk.missionLife.value, 0.0)

        exclude_mods = [
            "KnownRVSurvey",
            "ZodiacalLight",
            "BackgroundSources",
            "Completeness" "PlanetPhysicalModel",
            "PlanetPopulation",
            "PostProcessing",
        ]

        required_modules = [
            "Observatory",
            "OpticalSystem",
            "SimulatedUniverse",
            "TargetList",
            "TimeKeeping",
        ]

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue

            with RedirectStreams(stdout=self.dev_null):
                sim = mod(scriptfile=self.script1)

            self.assertIsInstance(sim._outspec, dict)
            # check for presence of a couple of class attributes
            self.assertIn("DRM", sim.__dict__)

            for rmod in required_modules:
                self.assertIn(rmod, sim.__dict__)
                self.assertEqual(getattr(sim, rmod)._modtype, rmod)

    def test_initOB(self):
        r"""Test init_OB method
        Strategy is to test Observing Blocks loaded from a file, then test
        automatically defined OB
        """
        tk = self.fixture()

        # 1) Load Observing Blocks from File
        OBduration = np.inf
        tk.init_OB(
            "sampleOB.csv", OBduration * u.d
        )  # File Located in: EXOSIMS/EXOSIMS/Scripts/sampleOB.csv
        self.assertTrue(tk.OBduration == OBduration * u.d)
        self.assertTrue(tk.OBnumber == 0)
        self.assertTrue(
            set(tk.OBstartTimes)
            == set([0, 40, 80, 120, 160, 200, 240, 280, 320, 360] * u.d)
        )
        self.assertTrue(
            set(tk.OBendTimes)
            == set([20, 60, 100, 140, 180, 220, 260, 300, 340, 380] * u.d)
        )

        # 2) Automatically construct OB from OBduration, missionLife, and
        # missionPortion SINGLE BLOCK
        OBduration = 10
        tk.missionLife = 100 * u.d
        tk.missionPortion = 0.1
        tk.init_OB(str(None), OBduration * u.d)
        self.assertTrue(tk.OBduration == OBduration * u.d)
        self.assertTrue(tk.OBnumber == 0)
        self.assertTrue(len(tk.OBendTimes) == 1)
        self.assertTrue(len(tk.OBstartTimes) == 1)
        self.assertTrue(tk.OBstartTimes[0] == 0 * u.d)
        self.assertTrue(tk.OBendTimes[0] == OBduration * u.d)

        # 3) Automatically construct OB from OBduration, missionLife, and
        # missionPortion TWO BLOCK
        OBduration = 10
        tk.missionLife = 100 * u.d
        tk.missionPortion = 0.2
        tk.init_OB(str(None), OBduration * u.d)
        self.assertTrue(tk.OBduration == OBduration * u.d)
        self.assertTrue(tk.OBnumber == 0)
        self.assertTrue(len(tk.OBendTimes) == 2)
        self.assertTrue(len(tk.OBstartTimes) == 2)
        self.assertTrue(set(tk.OBstartTimes) == set([0, 50] * u.d))
        self.assertTrue(set(tk.OBendTimes) == set([OBduration, 50 + OBduration] * u.d))

    def test_allocate_time(self):
        r"""Test allocate_time method.
        Approach: Ensure erraneous time allocations fail and time allocations
        exceeding mission constraints fail
        """
        tk = self.fixture(OBduration=10.0)

        # 1) dt = 0: All time allocation should fail
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.allocate_time(0 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.allocate_time(0 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 2) dt < 0: All time allocation should fail
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.allocate_time(-1 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.allocate_time(-1 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 3) Exceeds missionLife: All time allocation should fail
        tk.missionLife = 365 * u.d
        tk.currentTimeNorm = tk.missionLife.to("day") - 1 * u.d
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.allocate_time(2 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.allocate_time(2 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart

        # 4) Exceeds current OB: All time allocation should fail
        tk.OBendTimes = [20] * u.d
        tk.OBnumber = 0
        tk.currentTimeNorm = tk.OBendTimes[tk.OBnumber] - 1 * u.d
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.allocate_time(2 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.allocate_time(2 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart

        # 5a) Exceeds exoplanetObsTime: All time allocation should fail with add
        # Exoplanet Obs Time is True
        tk.missionLife = 10 * u.d
        tk.missionPortion = 0.2
        tk.OBendTimes = [10] * u.d
        tk.exoplanetObsTime = tk.missionLife.to("day") * tk.missionPortion - 1 * u.d
        tk.currentTimeNorm = tk.missionLife.to("day") * tk.missionPortion - 1 * u.d
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.allocate_time(2 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 5b) allocate_time with addExoplanetObsTime == False flag
        self.assertTrue(tk.allocate_time(2 * u.d, False))
        self.assertFalse(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertFalse(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 6a) allocate_time successful under nominal conditions with
        # addExoplanetObsTime == True
        tk.missionLife = 20 * u.d
        tk.missionPortion = 1
        tk.OBendTimes = [20] * u.d
        tk.OBnumber = 0
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        tk.exoplanetObsTime = 0 * u.d
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertTrue(tk.allocate_time(2 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs + 2 * u.d)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm + 2 * u.d)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime + 2 * u.d)

        # 6b) allocate_time successful under nominal conditions with
        # addExoplanetObsTime == True
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertTrue(tk.allocate_time(2 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs + 2 * u.d)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm + 2 * u.d)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

    def test_mission_is_over(self):
        r"""Test mission_is_over method.
        Approach: Allocate time until mission completes.  Check that the mission
        terminated at the right time.
        """
        life = 0.1 * u.year
        tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=1.0)
        sim = self.allmods[0](scriptfile=self.script1)
        allModes = sim.OpticalSystem.observingModes
        Obs = sim.Observatory
        OS = sim.OpticalSystem
        det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]

        # 1) mission not over
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        self.assertFalse(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should not be over at mission start.",
        )

        # 2) exoplanetObsTime exceeded
        # set exoplanetObsTime to failure condition
        tk.exoplanetObsTime = 1.1 * tk.missionLife.to("day") * tk.missionPortion
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over when exoplanetObsTime is exceeded.",
        )
        # reset exoplanetObsTime
        tk.exoplanetObsTime = 0.0 * tk.missionLife.to("day") * tk.missionPortion

        # 3) missionLife exceeded
        tk.currentTimeNorm = 1.1 * tk.missionLife.to("day")
        tk.currentTimeAbs = tk.missionStart + 1.1 * tk.missionLife.to("day")
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over when missionLife exceeded.",
        )
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart

        # 4) OBendTimes Exceeded
        tk.OBendTimes = [10] * u.d
        tk.OBnumber = 0
        tk.currentTimeNorm = tk.OBendTimes[tk.OBnumber] + 1 * u.d
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over past end of last observing block.",
        )
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart

        # 5) Fuel Exceeded
        OS.haveOcculter = True
        Obs.scMass = Obs.dryMass - 1 * u.kg
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over when main tank is dry.",
        )

        Obs.twotanks = True
        Obs.slewMass = 1 * u.kg
        Obs.skMass = -1 * u.kg
        Obs.scMass = Obs.slewMass + Obs.skMass + Obs.dryMass
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over when stationkeeping tank is dry.",
        )

        Obs.slewMass = -1 * u.kg
        Obs.skMass = 1 * u.kg
        Obs.scMass = Obs.slewMass + Obs.skMass + Obs.dryMass
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            "Mission should be over when slew tank is dry.",
        )

        # 6) Refueing
        Obs.twotanks = False
        Obs.scMass = Obs.dryMass - 1 * u.kg
        Obs.allowRefueling = True
        Obs.external_fuel_mass = Obs.maxFuelMass + 1 * u.kg
        self.assertFalse(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should not be over when main tank is dry but "
                "refueling is allowed."
            ),
        )
        # external tank should now be empty
        Obs.scMass = Obs.dryMass.copy()
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should be over when main tank is dry, refueling is "
                "allowed, but external tank is also dry."
            ),
        )

        # and now with two tanks
        Obs.twotanks = True
        Obs.slewMass = 1 * u.kg
        Obs.skMass = -1 * u.kg
        Obs.scMass = Obs.slewMass + Obs.skMass + Obs.dryMass
        Obs.external_fuel_mass = 1000 * u.kg
        Obs.skMaxFuelMass = 999 * u.kg
        self.assertFalse(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should not be over when stationkeeping tank is dry "
                "but refueling is allowed."
            ),
        )
        # external tank should now be empty
        Obs.skMass = 0 * u.kg
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should be over when statinokeeping tank is dry, "
                "refueling is allowed, but external tank is also dry."
            ),
        )

        Obs.slewMass = -1 * u.kg
        Obs.skMass = 1 * u.kg
        Obs.scMass = Obs.slewMass + Obs.skMass + Obs.dryMass
        Obs.external_fuel_mass = 1000 * u.kg
        Obs.slewMaxFuelMass = 999 * u.kg
        self.assertFalse(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should not be over when slew tank is dry "
                "but refueling is allowed."
            ),
        )
        # external tank should now be empty
        Obs.slewMass = 0 * u.kg
        self.assertTrue(
            tk.mission_is_over(OS, Obs, det_mode),
            (
                "Mission should be over when slew tank is dry, "
                "refueling is allowed, but external tank is also dry."
            ),
        )

    def test_advancetToStartOfNextOB(self):
        r"""Test advancetToStartOfNextOB method

        Strategy is to call the method once and ensure it advances the Observing Block
        """
        tk = self.fixture()

        tk.OBstartTimes = [0, 10, 20, 30] * u.d
        tk.OBendTimes = [5, 15, 25, 35] * u.d
        tk.OBnumber = 0

        # 1) Set current Time to End of OB
        tk.currentTimeNorm = tk.OBendTimes[0]
        tk.currentTimeAbs = tk.missionStart + tk.currentTimeNorm
        tmpOBnumber = tk.OBnumber
        tk.advancetToStartOfNextOB()
        self.assertTrue(tmpOBnumber + 1 == tk.OBnumber)
        self.assertTrue(tk.currentTimeNorm == tk.OBstartTimes[tk.OBnumber])
        self.assertTrue(
            tk.currentTimeAbs == tk.OBstartTimes[tk.OBnumber] + tk.missionStart
        )

    def test_advanceToAbsTime(self):
        r"""Test advanceToAbsTime Method"""
        tk = self.fixture()

        # 1) Check tAbs > currentTimeAbs (time to advance to is in future)
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.advanceToAbsTime(tk.currentTimeAbs - 1 * u.d, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.advanceToAbsTime(tk.currentTimeAbs - 1 * u.d, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 2) Check tAbs != currentTimeAbs (time to advance to is not currentTime)
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        self.assertFalse(tk.advanceToAbsTime(tk.currentTimeAbs, True))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertFalse(tk.advanceToAbsTime(tk.currentTimeAbs, False))
        self.assertTrue(tk.currentTimeAbs == tmpcurrentTimeAbs)
        self.assertTrue(tk.currentTimeNorm == tmpcurrentTimeNorm)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)

        # 3a) Check Use Case 1 and 3: addExoplanetObsTime == True
        # and exoplanetObsTime could be added
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [10] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.OBendTimes[0] + tk.missionStart
        self.assertTrue(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(tk.exoplanetObsTime == (tAbs - tmpcurrentTimeAbs).value * u.d)
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 3b) Check Use Case 1 and 3: addExoplanetObsTime == True
        # and exoplanetObsTime could NOT be added
        tk.missionLife = 1 * u.year
        tk.missionPortion = 0.5
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [365] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.OBendTimes[0] + tk.missionStart
        self.assertFalse(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(
            tk.exoplanetObsTime == tk.missionLife.to("day") * tk.missionPortion
        )
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 3c) Check Use Case 1 and 3: addExoplanetObsTime == False
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [10] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.OBendTimes[0] + tk.missionStart
        self.assertTrue(tk.advanceToAbsTime(tAbs, False))
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 4a) Check Use Case 2 and 4: addExoplanetObsTime == True
        # time advancement exceeds missionFinishAbs but NOT exoplanetObsTime
        tk.missionLife = 1 * u.year
        tk.missionFinishAbs = tk.missionStart + tk.missionLife.to("day")
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart + 300 * u.d
        tk.currentTimeNorm = 300 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [400] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.currentTimeAbs + 70 * u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(
            tk.exoplanetObsTime
            == (tk.missionLife.to("day") - tmpcurrentTimeNorm).to("day")
        )
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 4b) Check Use Case 2 and 4: addExoplanetObsTime == True
        # advancement exceeds exoplanet obs time and missionFinishAbs
        tk.missionLife = 1 * u.year
        tk.missionPortion = 0.05
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart + 340 * u.d
        tk.currentTimeNorm = 340 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [400] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.currentTimeAbs + 30 * u.d
        self.assertFalse(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(
            tk.exoplanetObsTime == tk.missionLife.to("day") * tk.missionPortion
        )
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 4c) Check Use Case 2 and 4: addExoplanetObsTime == False
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0] * u.d
        tk.OBendTimes = [400] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 10 * u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, False))
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertTrue(tk.currentTimeNorm == (tAbs - tk.missionStart).value * u.d)
        self.assertTrue(tk.currentTimeAbs == tAbs)

        # 5a) Check Use Case 5 and 7: addExoplanetObsTime == True
        # exoplanetObsTime is less than limit
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 17.5 * u.d  # 7.5*u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(tk.exoplanetObsTime == 10 * u.d)
        self.assertTrue(tk.OBnumber == 2)
        self.assertTrue(tk.currentTimeNorm == tk.OBstartTimes[tk.OBnumber])
        self.assertTrue(
            tk.currentTimeAbs == tk.OBstartTimes[tk.OBnumber] + tk.missionStart
        )

        # 5b) Check Use Case 5 and 7: addExoplanetObsTime == True
        # exoplanetObsTime is more than limit
        tk.missionLife = 1 * u.year
        tk.missionPortion = 0.025
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 37.5 * u.d
        self.assertFalse(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(tk.OBnumber == 4)
        self.assertTrue(
            tk.exoplanetObsTime == tk.missionLife.to("day") * tk.missionPortion
        )
        self.assertTrue(tk.currentTimeNorm == tk.OBstartTimes[tk.OBnumber])
        self.assertTrue(
            tk.currentTimeAbs == tk.OBstartTimes[tk.OBnumber] + tk.missionStart
        )

        # 5c) Check Use Case 5 and 7: addExoplanetObsTime == False
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 7.5 * u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, False))
        self.assertTrue(tk.OBnumber == 1)
        self.assertTrue(tk.exoplanetObsTime == tmpexoplanetObsTime)
        self.assertTrue(tk.currentTimeNorm == tk.OBstartTimes[tk.OBnumber])
        self.assertTrue(
            tk.currentTimeAbs == tk.OBstartTimes[tk.OBnumber] + tk.missionStart
        )

        # 6a) Check Use Case 6 and 8: addExoplanetObsTime == True
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 12.5 * u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(tk.OBnumber == 1)
        self.assertAlmostEqual(tk.exoplanetObsTime.value, (5.0 + 2.5))
        self.assertTrue(tk.currentTimeNorm == 12.5 * u.d)
        self.assertTrue(tk.currentTimeAbs == 12.5 * u.d + tk.missionStart)

        # 6b) Check Use Case 6 and 8: addExoplanetObsTime == True
        tk.missionLife = 40 * u.d
        tk.missionPortion = 0.25
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 22.5 * u.d
        self.assertFalse(tk.advanceToAbsTime(tAbs, True))
        self.assertTrue(tk.OBnumber == 2)
        self.assertTrue(
            tk.exoplanetObsTime == tk.missionLife.to("day") * tk.missionPortion
        )
        self.assertTrue(tk.currentTimeNorm == 22.5 * u.d)
        self.assertTrue(tk.currentTimeAbs == 22.5 * u.d + tk.missionStart)

        # 6c) Check Use Case 6 and 8: addExoplanetObsTime == False
        tk.missionLife = 20 * u.d
        tk.missionPortion = 0.5
        tk.exoplanetObsTime = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.currentTimeNorm = 0 * u.d
        tk.OBnumber = 0
        tk.OBstartTimes = [0, 10, 20, 30, 40, 50] * u.d
        tk.OBendTimes = [5, 15, 25, 35, 45, 55] * u.d
        tmpcurrentTimeNorm = tk.currentTimeNorm.copy()
        tmpcurrentTimeAbs = tk.currentTimeAbs.copy()
        tmpexoplanetObsTime = tk.exoplanetObsTime.copy()
        tAbs = tk.missionStart + 12.5 * u.d
        self.assertTrue(tk.advanceToAbsTime(tAbs, False))
        self.assertTrue(tk.OBnumber == 1)
        self.assertTrue(tk.exoplanetObsTime == 0 * u.d)
        self.assertTrue(tk.currentTimeNorm == 12.5 * u.d)
        self.assertTrue(tk.currentTimeAbs == 12.5 * u.d + tk.missionStart)

    def test_get_ObsDetectionMaxIntTime(self):
        r"""Test get_ObsDetectionMaxIntTime Method
        Strategy is to assign varions initial time conditions and check if we can
        allocate the time we expect we can. Will do one successful allocation and one
        unsuccessful allocation for each
        """
        life = 1.0 * u.year
        tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=1.0)
        sim = self.allmods[0](scriptfile=self.script1)
        allModes = sim.OpticalSystem.observingModes
        Obs = sim.Observatory
        mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]

        # 1) Does returned times enable allocation to succeed
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.exoplanetObsTime = 0 * u.d
        tk.OBendTimes = [1.0] * u.year
        tk.OBstartTimes = [0] * u.d
        MITOBT, MITEOT, MITML = tk.get_ObsDetectionMaxIntTime(Obs, mode)
        intTime = min([MITOBT, MITEOT, MITML])
        extraTime = intTime * (mode["timeMultiplier"] - 1)
        self.assertTrue(
            tk.allocate_time(
                intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], True
            )
        )

        # 2) Returned time ends mission at exoplanetObsTime = missionLife*missionPortion
        tk.missionLife = 1 * u.year
        tk.missionPortion = 0.1
        tk.allocated_time_d = tk.missionLife.to_value(u.d) * tk.missionPortion
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.exoplanetObsTime = 0 * u.d
        tk.OBendTimes = [1.0] * u.year
        tk.OBstartTimes = [0] * u.d
        MITOBT, MITEOT, MITML = tk.get_ObsDetectionMaxIntTime(Obs, mode)
        intTime = min([MITOBT, MITEOT, MITML])
        extraTime = intTime * (mode["timeMultiplier"] - 1)
        self.assertTrue(
            tk.allocate_time(
                intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], True
            )
        )  # was allocation successful
        self.assertTrue(
            tk.exoplanetObsTime == tk.missionLife.to("day") * tk.missionPortion
        )

        # 3) Returned time ends mission at missionLife
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.allocated_time_d = tk.missionLife.to_value(u.d) * tk.missionPortion
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.exoplanetObsTime = 0 * u.d
        tk.OBendTimes = [1.1] * u.year
        tk.OBstartTimes = [0] * u.d
        MITOBT, MITEOT, MITML = tk.get_ObsDetectionMaxIntTime(Obs, mode)
        intTime = min([MITOBT, MITML])
        extraTime = intTime * (mode["timeMultiplier"] - 1)
        self.assertTrue(
            tk.allocate_time(
                intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], False
            )
        )  # was allocation successful
        self.assertTrue(tk.missionLife.to("day") == tk.currentTimeNorm)
        self.assertTrue(tk.missionLife.to("day") == tk.currentTimeAbs - tk.missionStart)

        # 4) Returned time ends mission at OBendTime
        tk.missionLife = 1 * u.year
        tk.missionPortion = 1
        tk.allocated_time_d = tk.missionLife.to_value(u.d) * tk.missionPortion
        tk.currentTimeNorm = 0 * u.d
        tk.currentTimeAbs = tk.missionStart
        tk.exoplanetObsTime = 0 * u.d
        tk.OBendTimes = [0.5] * u.year
        tk.OBstartTimes = [0] * u.d
        MITOBT, MITEOT, MITML = tk.get_ObsDetectionMaxIntTime(Obs, mode)
        intTime = min([MITOBT, MITEOT, MITML])
        extraTime = intTime * (mode["timeMultiplier"] - 1)
        self.assertTrue(
            tk.allocate_time(
                intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], True
            )
        )  # was allocation successful
        self.assertTrue(tk.OBendTimes[tk.OBnumber] == tk.currentTimeNorm)
        self.assertTrue(
            tk.OBendTimes[tk.OBnumber] == tk.currentTimeAbs - tk.missionStart
        )


if __name__ == "__main__":
    unittest.main()
