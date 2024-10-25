import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.SurveySimulation
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.get_module import get_module
import os
import json
import copy
import pkgutil
import numpy as np
import astropy.units as u
import os.path
import sys
from io import StringIO


class TestSurveySimulation(unittest.TestCase):
    """

    Global SurveySimulation tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/simplest.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

        modtype = getattr(SurveySimulation, "_modtype")
        pkg = EXOSIMS.SurveySimulation
        with RedirectStreams(stdout=self.dev_null):
            self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not is_pkg:
                with RedirectStreams(stdout=self.dev_null):
                    mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_init(self):
        """
        Test of initialization and __init__.

        """

        exclude_mods = ["SS_char_only2"]

        required_modules = [
            "BackgroundSources",
            "Completeness",
            "Observatory",
            "OpticalSystem",
            "PlanetPhysicalModel",
            "PlanetPopulation",
            "PostProcessing",
            "SimulatedUniverse",
            "TargetList",
            "TimeKeeping",
            "ZodiacalLight",
        ]

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue

            spec = copy.deepcopy(self.spec)
            if "tieredScheduler" in mod.__name__:
                self.script = resource_path("test-scripts/simplest_occ.json")
                with open(self.script) as f:
                    spec = json.loads(f.read())
                spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

            if "KnownRV" in mod.__name__:
                spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
                spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
                spec["modules"]["SimulatedUniverse"] = "KnownRVPlanetsUniverse"

            with RedirectStreams(stdout=self.dev_null):
                sim = mod(**spec)

            self.assertIsInstance(sim._outspec, dict)
            # check for presence of a couple of class attributes
            self.assertIn("DRM", sim.__dict__)

            for rmod in required_modules:
                self.assertIn(rmod, sim.__dict__)
                self.assertEqual(getattr(sim, rmod)._modtype, rmod)

    def test_run_sim(self):
        r"""Test run_sim method.

        Approach: Ensures the simulation runs to completion and the output is set.
        """

        # expected contents of DRM:
        All_DRM_keys = [
            "FA_char_status",
            "char_mode",
            "det_status",
            "char_params",
            "star_name",
            "plan_inds",
            "FA_char_dMag",
            "OB_nb",
            "char_fZ",
            "det_SNR",
            "FA_char_JEZ",
            "char_status",
            "det_mode",
            "det_time",
            "arrival_time",
            "char_SNR",
            "det_params",
            "char_time",
            "FA_char_SNR",
            "det_fZ",
            "FA_det_status",
            "star_ind",
            "FA_char_WA",
        ]
        det_only_DRM_keys = [
            "det_status",
            "star_name",
            "plan_inds",
            "OB_nb",
            "det_SNR",
            "det_mode",
            "det_time",
            "arrival_time",
            "det_params",
            "det_fZ",
            "star_ind",
        ]
        TS_DRM_keys = ["plan_inds", "OB_nb", "arrival_time", "star_ind"]

        exclude_mods = [
            "SS_char_only",
            "SS_char_only2",
            "SS_det_only",
            "tieredScheduler_SLSQP",
            "tieredScheduler_DD_SLSQP",
            "linearJScheduler_orbitChar",
        ]
        exclude_mod_type = "sotoSS"

        for mod in self.allmods:
            if mod.__name__ in exclude_mods or exclude_mod_type in mod.__name__:
                continue

            spec = copy.deepcopy(self.spec)
            if "tieredScheduler" in mod.__name__:
                self.script = resource_path("test-scripts/simplest_occ.json")
                with open(self.script) as f:
                    spec = json.loads(f.read())
                spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

            if (
                "linearJScheduler" in mod.__name__
                or "ExoC" in mod.__name__
                or "coroOnly" in mod.__name__
            ):
                self.script = resource_path("test-scripts/simplest_3DDPC.json")
                with open(self.script) as f:
                    spec = json.loads(f.read())

            if "KnownRV" in mod.__name__:
                spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
                spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
                spec["modules"]["SimulatedUniverse"] = "KnownRVPlanetsUniverse"

            if "run_sim" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)
                    sim.run_sim()
                    # check that a mission constraint has been exceeded
                    allModes = sim.OpticalSystem.observingModes
                    mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
                    exoplanetObsTimeCondition = (
                        sim.TimeKeeping.exoplanetObsTime
                        + sim.Observatory.settlingTime
                        + mode["syst"]["ohTime"]
                        >= sim.TimeKeeping.missionLife * sim.TimeKeeping.missionPortion
                    )
                    missionLifeCondition = (
                        sim.TimeKeeping.currentTimeNorm
                        + sim.Observatory.settlingTime
                        + mode["syst"]["ohTime"]
                        >= sim.TimeKeeping.missionLife
                    )
                    OBcondition = (
                        sim.TimeKeeping.OBendTimes[sim.TimeKeeping.OBnumber]
                        <= sim.TimeKeeping.currentTimeNorm
                        + sim.Observatory.settlingTime
                        + mode["syst"]["ohTime"]
                    )

                self.assertTrue(
                    exoplanetObsTimeCondition or (missionLifeCondition or OBcondition),
                    "Mission did not run to completion for %s" % mod.__name__,
                )

                # resulting DRM is a list...
                self.assertIsInstance(
                    sim.DRM, list, "DRM is not a list for %s" % mod.__name__
                )
                # ...and has nontrivial number of entries
                self.assertGreater(
                    len(sim.DRM), 0, "DRM is empty for %s" % mod.__name__
                )

                if (
                    "det_only" in mod.__name__
                    or "ExoC" in mod.__name__
                    or "linearJScheduler_orbitChar" in mod.__name__
                ):
                    for key in det_only_DRM_keys:
                        self.assertIn(
                            key,
                            sim.DRM[0],
                            "DRM is missing key %s for %s" % (key, mod.__name__),
                        )
                elif "tieredScheduler" in mod.__name__ or "coroOnly" in mod.__name__:
                    for key in TS_DRM_keys:
                        self.assertIn(
                            key,
                            sim.DRM[0],
                            "DRM is missing key %s for %s" % (key, mod.__name__),
                        )
                elif "DDPC" in mod.__name__:
                    for keys in All_DRM_keys:
                        self.assertIn(
                            key,
                            list(sim.DRM[0]["det_info"][0].keys())
                            + list(sim.DRM[0]["char_info"][0].keys())
                            + list(sim.DRM[0].keys()),
                            "DRM is missing key %s for %s" % (key, mod.__name__),
                        )
                else:
                    for key in All_DRM_keys:
                        self.assertIn(
                            key,
                            sim.DRM[0],
                            "DRM is missing key %s for %s" % (key, mod.__name__),
                        )

    def test_next_target(self):
        r"""Test next_target method.

        Approach: Ensure the next target is selected OK, and is a valid integer.
        Deficiencies: We are not checking that the occulter slew works.
        """

        exclude_mods = ["SS_det_only", "linearJScheduler_orbitChar"]
        exclude_mod_type = "sotoSS"

        for mod in self.allmods:
            if mod.__name__ in exclude_mods or exclude_mod_type in mod.__name__:
                continue

            if "next_target" in mod.__dict__:
                if "tieredScheduler" in mod.__name__:
                    with RedirectStreams(stdout=self.dev_null):
                        if "tieredScheduler_DD_SS" in mod.__name__:
                            self.script = resource_path(
                                "test-scripts/simplest_occSS.json"
                            )
                            with open(self.script) as f:
                                spec = json.loads(f.read())
                        else:
                            self.script = resource_path(
                                "test-scripts/simplest_occ.json"
                            )
                            with open(self.script) as f:
                                spec = json.loads(f.read())
                        spec["occHIPs"] = resource_path(
                            "SurveySimulation/top100stars.txt"
                        )
                        sim = mod(**spec)
                        if "tieredScheduler_DD" in mod.__name__:
                            (
                                DRM_out,
                                sInd,
                                occ_sInd,
                                intTime,
                                sd,
                                occ_sInds,
                                det_mode,
                            ) = sim.next_target(
                                None,
                                None,
                                sim.OpticalSystem.observingModes,
                                sim.OpticalSystem.observingModes[0],
                            )
                        else:
                            (
                                DRM_out,
                                sInd,
                                occ_sInd,
                                intTime,
                                sd,
                                occ_sInds,
                            ) = sim.next_target(
                                None,
                                None,
                                sim.OpticalSystem.observingModes[0],
                                sim.OpticalSystem.observingModes[0],
                            )
                        if occ_sInd is not None:
                            self.assertIsInstance(
                                occ_sInd,
                                (int, np.int8, np.int16, np.int32, np.int64),
                                "occ_sInd is not an integer for %s" % mod.__name__,
                            )
                            self.assertEqual(
                                occ_sInd - int(occ_sInd),
                                0,
                                "occ_sInd is not an integer for %s" % (mod.__name__),
                            )
                            self.assertGreaterEqual(
                                occ_sInd,
                                0,
                                "occ_sInd is not a valid index for %s" % mod.__name__,
                            )
                            self.assertLess(
                                occ_sInd,
                                sim.TargetList.nStars,
                                "occ_sInd is not a valid index for %s" % mod.__name__,
                            )
                elif (
                    "linearJScheduler_DDPC" in mod.__name__
                    or "linearJScheduler_3DDPC" in mod.__name__
                ):
                    self.script = resource_path("test-scripts/simplest_3DDPC.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    with RedirectStreams(stdout=self.dev_null):
                        sim = mod(**spec)
                        DRM_out, sInd, intTime, waitTime, det_mode = sim.next_target(
                            None, sim.OpticalSystem.observingModes
                        )
                elif "coroOnlyScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_3DDPC.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    with RedirectStreams(stdout=self.dev_null):
                        sim = mod(**spec)
                        DRM_out, sInd, intTime, waitTime, det_mode = sim.next_target(
                            None,
                            sim.OpticalSystem.observingModes,
                            sim.OpticalSystem.observingModes,
                        )
                elif "linearJScheduler_orbitChar" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_3DDPC.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    with RedirectStreams(stdout=self.dev_null):
                        sim = mod(**spec)
                        DRM_out, sInd, intTime, waitTime = sim.next_target(
                            None,
                            sim.OpticalSystem.observingModes[0],
                            sim.OpticalSystem.observingModes[0],
                        )
                else:
                    with RedirectStreams(stdout=self.dev_null):
                        sim = mod(scriptfile=self.script)

                        DRM_out, sInd, intTime, waitTime = sim.next_target(
                            None, sim.OpticalSystem.observingModes[0]
                        )

                # result index is a scalar numpy ndarray, that is a valid integer
                # in a valid range
                if sInd is not None:
                    self.assertIsInstance(
                        sInd,
                        (int, np.int8, np.int16, np.int32, np.int64),
                        "sInd is not an integer for %s" % mod.__name__,
                    )
                    self.assertEqual(
                        sInd - int(sInd),
                        0,
                        "sInd is not an integer for %s" % mod.__name__,
                    )
                    self.assertGreaterEqual(
                        sInd, 0, "sInd is not a valid index for %s" % mod.__name__
                    )
                    self.assertLess(
                        sInd,
                        sim.TargetList.nStars,
                        "sInd is not a valid index for %s" % mod.__name__,
                    )

                # resulting DRM is a dictionary -- contents unimportant
                self.assertIsInstance(
                    DRM_out, dict, "DRM_out is not a dict for %s" % mod.__name__
                )

    def test_choose_next_target(self):
        r"""Test choose_next_target method.

        Approach: Ensure the next target is a valid index for different cases:
        old_sInd is none, old_sInd in sInds, old_sInd not in sInds
        """

        exclude_mods = ["SS_char_only", "SS_char_only2", "SS_det_only"]
        exclude_mod_type = "sotoSS"

        for mod in self.allmods:
            if mod.__name__ in exclude_mods or exclude_mod_type in mod.__name__:
                continue

            if "choose_next_target" in mod.__dict__:

                spec = copy.deepcopy(self.spec)
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")
                if "KnownRV" in mod.__name__:
                    spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
                    spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
                    spec["modules"]["SimulatedUniverse"] = "KnownRVPlanetsUniverse"
                if ("occulterJScheduler" in mod.__name__) or (
                    "linearJScheduler" in mod.__name__
                ):
                    spec["starlightSuppressionSystems"] = [
                        {
                            "name": "occulter",
                            "occulter": True,
                            "lam": 550,
                            "BW": 0.10,
                            "IWA": 0.1,
                            "OWA": 0,
                            "occ_trans": 1,
                        }
                    ]
                    spec["nSteps"] = 2
                    spec["modules"]["Observatory"] = "SotoStarshade"

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)

                    # old sInd is None
                    sInds = np.array([0, 1, 2])
                    sInd, waitTime = sim.choose_next_target(
                        None,
                        sInds,
                        np.array([1.0] * sim.TargetList.nStars) * u.d,
                        np.ones((len(sInds),)) * u.d,
                    )

                    self.assertTrue(
                        (sInd in sInds) or (sInd is None),
                        "sInd not in passed sInds for %s" % mod.__name__,
                    )

                    # old sInd in sInds
                    sInds = np.random.choice(
                        sim.TargetList.nStars,
                        size=int(sim.TargetList.nStars / 2.0),
                        replace=False,
                    )
                    old_sInd = np.random.choice(sInds)
                    _ = sim.observation_detection(
                        old_sInd, 1.0 * u.d, sim.OpticalSystem.observingModes[0]
                    )
                    sInd, waitTime = sim.choose_next_target(
                        old_sInd,
                        sInds,
                        np.array([1.0] * sim.TargetList.nStars) * u.d,
                        np.array([1.0] * len(sInds)) * u.d,
                    )

                    self.assertTrue(
                        (sInd in sInds) or (sInd is None),
                        "sInd not in passed sInds for %s" % mod.__name__,
                    )

                    # old sInd not in sInds
                    sInds = np.random.choice(
                        sim.TargetList.nStars,
                        size=int(sim.TargetList.nStars / 2.0),
                        replace=False,
                    )
                    tmp = list(set(np.arange(sim.TargetList.nStars)) - set(sInds))
                    old_sInd = np.random.choice(tmp)
                    _ = sim.observation_detection(
                        old_sInd, 1.0 * u.d, sim.OpticalSystem.observingModes[0]
                    )
                    sInd, waitTime = sim.choose_next_target(
                        old_sInd,
                        sInds,
                        np.array([1.0] * sim.TargetList.nStars) * u.d,
                        np.array([1.0] * len(sInds)) * u.d,
                    )

                    self.assertTrue(
                        (sInd in sInds) or (sInd is None),
                        "sInd not in passed sInds for %s" % mod.__name__,
                    )

    def test_calc_targ_intTime(self):
        """Test calc_targ_intTime method.
        Checks that proper outputs are given (length and units).
        """

        exclude_mods = []

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue
            if "calc_targ_intTime" in mod.__dict__:

                spec = copy.deepcopy(self.spec)
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")
                if "KnownRV" in mod.__name__:
                    spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
                    spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
                    spec["modules"]["SimulatedUniverse"] = "KnownRVPlanetsUniverse"

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)
                    startTimes = (
                        sim.TimeKeeping.currentTimeAbs.copy()
                        + np.zeros(sim.TargetList.nStars) * u.d
                    )
                    sInds = np.arange(sim.TargetList.nStars)
                    mode = list(
                        filter(
                            lambda mode: mode["detectionMode"],
                            sim.OpticalSystem.observingModes,
                        )
                    )[0]
                    intTimes = sim.calc_targ_intTime(sInds, startTimes, mode)
                self.assertTrue(
                    len(intTimes) == len(sInds),
                    "calc_targ_intTime returns incorrect number of intTimes for %s"
                    % mod.__name__,
                )
                self.assertTrue(
                    intTimes.unit == u.d,
                    "calc_targ_intTime returns incorrect unit for %s" % mod.__name__,
                )

    def test_observation_detection(self):
        r"""Test observation_detection method.

        Approach: Ensure that all outputs are set as expected
        """

        exclude_mods = []
        exclude_mod_type = "sotoSS"

        for mod in self.allmods:
            if (mod.__name__ in exclude_mods) or (exclude_mod_type in mod.__name__):
                continue
            if "observation_detection" in mod.__dict__:

                spec = copy.deepcopy(self.spec)
                spec["ntargs"] = 1
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)

                    # default settings should create dummy planet around first star
                    sInd = 0
                    pInds = np.where(sim.SimulatedUniverse.plan2star == sInd)[0]
                    detected, fZ, JEZ, systemParams, SNR, FA = (
                        sim.observation_detection(
                            sInd, 1.0 * u.d, sim.OpticalSystem.observingModes[0]
                        )
                    )

                self.assertEqual(
                    len(detected),
                    len(pInds),
                    "len(detected) != len(pInds) for %s" % mod.__name__,
                )
                self.assertIsInstance(
                    detected[0],
                    (int, np.int32, np.int64, np.int_),
                    "detected elements not ints for %s" % mod.__name__,
                )
                for s in SNR[detected == 1]:
                    self.assertGreaterEqual(
                        s,
                        sim.OpticalSystem.observingModes[0]["SNR"],
                        "detection SNR < mode requirement for %s" % mod.__name__,
                    )
                self.assertIsInstance(
                    FA, bool, "False Alarm not boolean for %s" % mod.__name__
                )

    def test_scheduleRevisit(self):
        """Runs scheduleRevisit method"""

        exclude_mods = []

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue

            if "scheduleRevisit" in mod.__dict__:
                spec = copy.deepcopy(self.spec)
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)
                    sInd = 0
                    smin = None
                    det = 0
                    pInds = [0]
                    sim.scheduleRevisit(sInd, smin, det, pInds)

    def test_observation_characterization(self):
        r"""Test observation_characterization method.

        Approach: Ensure all outputs are set as expected
        """

        exclude_mods = ["SS_char_only", "SS_char_only2"]
        exclude_mod_type = "sotoSS"

        for mod in self.allmods:
            if (mod.__name__ in exclude_mods) or (exclude_mod_type in mod.__name__):
                continue

            if "observation_characterization" in mod.__dict__:
                if ("tieredScheduler" in mod.__name__) or ("ExoC" in mod.__name__):
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")
                elif ("DDPC" in mod.__name__) or ("coroOnly" in mod.__name__):
                    self.script = resource_path("test-scripts/simplest_3DDPC.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                else:
                    spec = copy.deepcopy(self.spec)
                spec["ntargs"] = 1

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)

                    # default settings should create dummy planet around first star
                    sInd = 0
                    pInds = np.where(sim.SimulatedUniverse.plan2star == sInd)[0]

                    # in order to test for characterization, we need to have previously
                    # detected the planet, so let's do that first
                    spectroModes = list(
                        filter(
                            lambda mode: "spec" in mode["inst"]["name"],
                            sim.OpticalSystem.observingModes,
                        )
                    )
                    detected, fZ, JEZ, systemParams, SNR, FA = (
                        sim.observation_detection(
                            sInd, 1.0 * u.d, sim.OpticalSystem.observingModes[0]
                        )
                    )
                    # now the characterization
                    if "ExoC" in mod.__name__ or "coroOnly" in mod.__name__:
                        (
                            characterized,
                            fZ,
                            JEZ,
                            systemParams,
                            SNR,
                            intTime,
                        ) = sim.observation_characterization(
                            sInd, sim.OpticalSystem.observingModes[0], 0
                        )

                        self.assertEqual(
                            len(characterized),
                            len(pInds),
                            "len(characerized) != len(pInds) for %s" % mod.__name__,
                        )
                        self.assertIsInstance(
                            characterized[0],
                            (int, np.int32, np.int64, np.int_),
                            "characterized elements not ints for %s" % mod.__name__,
                        )
                        for s in SNR[characterized == 1]:
                            self.assertGreaterEqual(
                                s,
                                sim.OpticalSystem.observingModes[0]["SNR"],
                                "char SNR less than required for %s" % mod.__name__,
                            )

                    elif "PC" in mod.__name__:
                        (
                            characterized,
                            fZ,
                            JEZ,
                            systemParams,
                            SNR,
                            intTime,
                        ) = sim.observation_characterization(sInd, spectroModes)
                        self.assertEqual(
                            len(characterized[0, :]),
                            len(spectroModes),
                            "len(characerized_modes) != len(spectroModes) for %s"
                            % mod.__name__,
                        )
                        self.assertEqual(
                            len(characterized[:, 0]),
                            len(pInds),
                            "len(characerized) != len(pInds) for %s" % mod.__name__,
                        )
                        self.assertIsInstance(
                            characterized[0, 0],
                            (int, np.int32, np.int64, np.int_),
                            "characterized elements not ints for %s" % mod.__name__,
                        )
                        for s in SNR[:, 0][characterized[:, 0] == 1]:
                            self.assertGreaterEqual(
                                s,
                                sim.OpticalSystem.observingModes[0]["SNR"],
                                "char SNR less than required for %s" % mod.__name__,
                            )

                    else:
                        (
                            characterized,
                            fZ,
                            JEZ,
                            systemParams,
                            SNR,
                            intTime,
                        ) = sim.observation_characterization(
                            sInd, sim.OpticalSystem.observingModes[0]
                        )

                        self.assertEqual(
                            len(characterized),
                            len(pInds),
                            "len(characerized) != len(pInds) for %s" % mod.__name__,
                        )
                        self.assertIsInstance(
                            characterized[0],
                            (int, np.int32, np.int64, np.int_),
                            "characterized elements not ints for %s" % mod.__name__,
                        )
                        for s in SNR[characterized == 1]:
                            self.assertGreaterEqual(
                                s,
                                sim.OpticalSystem.observingModes[0]["SNR"],
                                "char SNR less than required for %s" % mod.__name__,
                            )

                if intTime is not None:
                    self.assertLessEqual(
                        intTime,
                        sim.OpticalSystem.intCutoff,
                        "char intTime greater than cutoff for %s" % mod.__name__,
                    )

    def test_calc_signal_noise(self):
        r"""Test calc_signal_noise method.

        Approach: Ensure that signal is greater than noise for dummy planet
        """

        exclude_mods = []

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue
            if "calc_signal_noise" in mod.__dict__:

                spec = copy.deepcopy(self.spec)
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)

                    S, N = sim.calc_signal_noise(
                        np.array([0]),
                        np.array([0]),
                        1.0 * u.d,
                        sim.OpticalSystem.observingModes[0],
                        fZ=np.array([0.0]) / u.arcsec**2,
                        JEZ=np.array([0.0]) * u.ph / u.s / u.m**2 / u.arcsec**2,
                        dMag=np.array([20]),
                        WA=np.array([0.5]) * u.arcsec,
                    )

                self.assertGreaterEqual(S, N)

    def test_revisitFilter(self):
        r"""Test revisitFilter method"""

        exclude_mods = []
        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue

            if "revisitFilter" in mod.__dict__:
                spec = copy.deepcopy(self.spec)
                if "tieredScheduler" in mod.__name__:
                    self.script = resource_path("test-scripts/simplest_occ.json")
                    with open(self.script) as f:
                        spec = json.loads(f.read())
                    spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")
                with RedirectStreams(stdout=self.dev_null):
                    sim = mod(**spec)

                    sInds = np.asarray([0])
                    tovisit = np.zeros(sim.TargetList.nStars, dtype=bool)

                    sInds = sim.revisitFilter(sInds, sim.TimeKeeping.currentTimeNorm)
                try:
                    self.assertIsInstance(sInds, np.ndarray)
                except:
                    self.assertIsInstance(sInds, type(list()))

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """
        atts_list = ["DRM", "seed", "starVisits"]

        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            spec = copy.deepcopy(self.spec)
            if "tieredScheduler" in mod.__name__:
                self.script = resource_path("test-scripts/simplest_occ.json")
                with open(self.script) as f:
                    spec = json.loads(f.read())
                spec["occHIPs"] = resource_path("SurveySimulation/top100stars.txt")

            if "KnownRV" in mod.__name__:
                spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
                spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
                spec["modules"]["SimulatedUniverse"] = "KnownRVPlanetsUniverse"

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**spec)
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


if __name__ == "__main__":
    unittest.main()
