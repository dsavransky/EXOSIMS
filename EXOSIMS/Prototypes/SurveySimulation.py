# -*- coding: utf-8 -*-
import copy
import hashlib
import inspect
import json
import logging
import os
import random as py_random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.time import Time

from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util._numpy_compat import copy_if_needed


Logger = logging.getLogger(__name__)


class SurveySimulation(object):
    """:ref:`SurveySimulation` Prototype

    Args:

        scriptfile (str, optional):
            JSON script file.  If not set, assumes that dictionary has been
            passed through specs. Defaults to None\
        ntFlux (int):
            Number of intervals to split integration into for computing
            total SNR. When greater than 1, SNR is effectively computed as a
            Reimann sum. Defaults to 1
        nVisitsMax (int):
            Maximum number of observations (in detection mode) per star.
            Defaults to 5
        charMargin (float):
            Integration time margin for characterization. Defaults to 0.15
        dt_max (float):
            Maximum time for revisit window (in days). Defaults to 1.
        record_counts_path (str, optional):
            If set, write out photon count info to file specified by this keyword.
            Defaults to None.
        revisit_wait (float, optional):
            If set, it is the minimum time (in days) to wait before revisiting current target. Defaults to None.
        nokoMap (bool):
            Skip generating keepout map. Only useful if you're not planning on
            actually running a mission simulation. Defaults to False.
        nofZ (bool):
            Skip precomputing zodical light minima.  Only useful if you're not
            planning on actually running a mission simulation.  Defaults to False.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        defaultAddExoplanetObsTime (bool):
            If True, time advancement when no targets are observable will add
            to exoplanetObsTime (i.e., wasting time is counted against you).
            Defaults to True
        find_known_RV (bool):
            Identify stars with known planets. Defaults to False
        include_known_RV (str, optional):
            Path to file including known planets to include. Defaults to None
        make_debug_bird_plots (bool, optional):
            If True, makes completeness bird plots for every observation that
            are saved in the cache directory
        debug_plot_path (str, optional):
            Path to save the debug plots in, must be set if
            make_debug_bird_plots is True
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        absTimefZmin (astropy.time.core.Time):
            Absolute time of local zodi minima
        BackgroundSources (:ref:`BackgroundSources`):
            BackgroundSources object
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        cachefname (str):
            Base filename for cache files.
        charMargin (float):
            Integration time margin for characterization.
        Completeness (:ref:`Completeness`):
            Completeness object
        count_lines (list):
            Photon counts.  Only used when ``record_counts_path`` is set
        defaultAddExoplanetObsTime (bool):
            If True, time advancement when no targets are observable will add
            to exoplanetObsTime (i.e., wasting time is counted against you).
        DRM (list):
            The mission simulation.  List of observation dictionaries.
        dt_max (astropy.units.quantity.Quantity):
            Maximum time for revisit window.
        revisit_wait (astropy.units.quantity.Quantity):
            Minimum time (in days) to wait before revisiting.
        find_known_RV (bool):
            Identify stars with known planets.
        fullSpectra (numpy.ndarray(bool)):
            Array of booleans indicating whether a planet's spectrum has been
            fully observed.
        fZmins (dict):
            Dictionary of local zodi minimum value candidates for each observing mode
        fZtypes (dict):
            Dictionary of type of local zodi minimum candidates for each observing mode
        include_known_RV (str, optional):
            Path to file including known planets to include.
        intTimeFilterInds (numpy.ndarray(ind)):
            Indices of targets where integration times fall below cutoff value
        intTimesIntTimeFilter (astropy.units.quantity.Quantity):
            Default integration times for pre-filtering targets.
        known_earths (numpy.ndarray):
            Indices of Earth-like planets
        known_rocky (list):
            Indices of rocky planets
        known_stars (list):
            Stars with known planets
        koMaps (dict):
            Keepout Maps
        koTimes (astropy.time.core.Time):
            Times corresponding to keepout map array entries.
        lastDetected (numpy.ndarray):
            ntarg x 4. For each target, contains 4 lists with planets' detected
            status (boolean),
            exozodi brightness (in units of 1/arcsec2), delta magnitude,
            and working angles (in units of arcsec)
        lastObsTimes (astropy.units.quantity.Quantity):
            Contains the last observation start time for future completeness update
            in units of day
        logger (logging.Logger):
            Logger object
        modules (dict):
            Modules dictionary.
        ntFlux (int):
            Number of intervals to split integration into for computing
            total SNR. When greater than 1, SNR is effectively computed as a
            Reimann sum.
        nVisitsMax (int):
            Maximum number of observations (in detection mode) per star.
        Observatory (:ref:`Observatory`):
            Observatory object.
        OpticalSystem (:ref:`OpticalSystem`):
            Optical system object.
        partialSpectra (numpy.ndarray):
            Array of booleans indicating whether a planet's spectrum has been
            partially observed.
        PlanetPhysicalModel (:ref:`PlanetPhysicalModel`):
            Planet pysical model object
        PlanetPopulation (:ref:`PlanetPopulation`):
            Planet population object
        PostProcessing (:ref:`PostProcessing`):
            Postprocessing object
        propagTimes (astropy.units.quantity.Quantity):
            Contains the current time at each target system.
        record_counts_path (str, optional):
            If set, write out photon count info to file specified by this keyword.
        seed (int):
            Seed for random number generation
        SimulatedUniverse (:ref:`SimulatedUniverse`):
            Simulated universe object
        StarCatalog (:ref:`StarCatalog`):
            Star catalog object (only if ``keepStarCatalog`` input is True.
        starExtended (numpy.ndarray):
            TBD
        starRevisit (numpy.ndarray):
            ntargs x 2. Contains indices of targets to revisit and revisit times
            of these targets in units of day
        starVisits (numpy.ndarray):
            ntargs x 1. Contains the number of times each target was visited
        TargetList (:ref:`TargetList`):
            TargetList object.
        TimeKeeping (:ref:`TimeKeeping`):
            Timekeeping object
        valfZmin (astropy.units.quantity.Quantity):
            Minimum local zodi for each target.
        ZodiacalLight (:ref:`ZodiacalLight`):
            Zodiacal light object.

    """

    _modtype = "SurveySimulation"

    def __init__(
        self,
        scriptfile=None,
        ntFlux=1,
        nVisitsMax=5,
        charMargin=0.15,
        dt_max=1.0,
        record_counts_path=None,
        revisit_wait=None,
        nokoMap=False,
        nofZ=False,
        cachedir=None,
        defaultAddExoplanetObsTime=True,
        find_known_RV=False,
        include_known_RV=None,
        make_debug_bird_plots=False,
        debug_plot_path=None,
        **specs,
    ):

        # start the outspec
        self._outspec = {}

        # if a script file is provided read it in. If not set, assumes that
        # dictionary has been passed through specs.
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file." % scriptfile

            try:
                with open(scriptfile) as ff:
                    script = ff.read()
                specs.update(json.loads(script))
            except ValueError:
                sys.stderr.write("Script file `%s' is not valid JSON." % scriptfile)
                # must re-raise, or the error will be masked
                raise

            # modules array must be present
            if "modules" not in specs:
                raise ValueError("No modules field found in script.")

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # count dict contains all of the C info for each star index
        self.record_counts_path = record_counts_path
        self.count_lines = []
        self._outspec["record_counts_path"] = record_counts_path

        # mission simulation logger
        self.logger = specs.get("logger", logging.getLogger(__name__))

        # set up numpy random number (generate it if not in specs)
        self.seed = int(specs.get("seed", py_random.randint(1, int(1e9))))
        self.vprint("Numpy random seed is: %s" % self.seed)
        np.random.seed(self.seed)
        self._outspec["seed"] = self.seed

        # cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        # N.B.: cachedir is going to be used by everything, so let's make sure that
        # it doesn't get popped out of specs
        specs["cachedir"] = self.cachedir

        # if any of the modules is a string, assume that they are all strings
        # and we need to initalize
        if isinstance(next(iter(specs["modules"].values())), str):

            # import desired module names (prototype or specific)
            self.SimulatedUniverse = get_module(
                specs["modules"]["SimulatedUniverse"], "SimulatedUniverse"
            )(**specs)
            self.Observatory = get_module(
                specs["modules"]["Observatory"], "Observatory"
            )(**specs)
            self.TimeKeeping = get_module(
                specs["modules"]["TimeKeeping"], "TimeKeeping"
            )(**specs)

            # bring inherited class objects to top level of Survey Simulation
            SU = self.SimulatedUniverse
            self.StarCatalog = SU.StarCatalog
            self.PlanetPopulation = SU.PlanetPopulation
            self.PlanetPhysicalModel = SU.PlanetPhysicalModel
            self.OpticalSystem = SU.OpticalSystem
            self.ZodiacalLight = SU.ZodiacalLight
            self.BackgroundSources = SU.BackgroundSources
            self.PostProcessing = SU.PostProcessing
            self.Completeness = SU.Completeness
            self.TargetList = SU.TargetList

        else:
            # these are the modules that must be present if passing instantiated objects
            neededObjMods = [
                "PlanetPopulation",
                "PlanetPhysicalModel",
                "OpticalSystem",
                "ZodiacalLight",
                "BackgroundSources",
                "PostProcessing",
                "Completeness",
                "TargetList",
                "SimulatedUniverse",
                "Observatory",
                "TimeKeeping",
            ]

            # ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs["modules"]:
                    raise ValueError(
                        "%s module is required but was not provided." % modName
                    )

            for modName in specs["modules"]:
                assert specs["modules"][modName]._modtype == modName, (
                    "Provided instance of %s has incorrect modtype." % modName
                )

                setattr(self, modName, specs["modules"][modName])

        # create a dictionary of all modules, except StarCatalog
        self.modules = {}
        self.modules["PlanetPopulation"] = self.PlanetPopulation
        self.modules["PlanetPhysicalModel"] = self.PlanetPhysicalModel
        self.modules["OpticalSystem"] = self.OpticalSystem
        self.modules["ZodiacalLight"] = self.ZodiacalLight
        self.modules["BackgroundSources"] = self.BackgroundSources
        self.modules["PostProcessing"] = self.PostProcessing
        self.modules["Completeness"] = self.Completeness
        self.modules["TargetList"] = self.TargetList
        self.modules["SimulatedUniverse"] = self.SimulatedUniverse
        self.modules["Observatory"] = self.Observatory
        self.modules["TimeKeeping"] = self.TimeKeeping
        # add yourself to modules list for bookkeeping purposes
        self.modules["SurveySimulation"] = self

        # observation time sampling
        self.ntFlux = int(ntFlux)
        self._outspec["ntFlux"] = self.ntFlux

        # maximum number of observations per star
        self.nVisitsMax = int(nVisitsMax)
        self._outspec["nVisitsMax"] = self.nVisitsMax

        # integration time margin for characterization
        self.charMargin = float(charMargin)
        self._outspec["charMargin"] = self.charMargin

        # maximum time for revisit window
        self.dt_max = float(dt_max) * u.week
        self._outspec["dt_max"] = self.dt_max.value

        # minimum time for revisit window
        if revisit_wait is not None:
            self.revisit_wait = revisit_wait * u.d
        else:
            self.revisit_wait = revisit_wait

        # list of detected earth-like planets aroung promoted stars
        self.known_earths = np.array([])

        self.find_known_RV = find_known_RV
        self._outspec["find_known_RV"] = find_known_RV
        self._outspec["include_known_RV"] = include_known_RV
        if self.find_known_RV:
            # select specific knonw RV stars if a file exists
            if include_known_RV is not None:
                if os.path.isfile(include_known_RV):
                    with open(include_known_RV, "r") as rv_file:
                        self.include_known_RV = [
                            hip.strip() for hip in rv_file.read().split(",")
                        ]
                        self.vprint(
                            "Including known RV stars: {}".format(self.include_known_RV)
                        )
                else:
                    self.include_known_RV = None
                    self.vprint(
                        "WARNING: Known RV file: {} does not exist!!".format(
                            include_known_RV
                        )
                    )
            else:
                self.include_known_RV = None
            self.known_stars, self.known_rocky = self.find_known_plans()
        else:
            self.include_known_RV = None
            self.known_stars = []
            self.known_rocky = []

        # defaultAddExoplanetObsTime Tells us time advanced when no targets available
        # counts agains exoplanetObsTime (when True)
        self.defaultAddExoplanetObsTime = defaultAddExoplanetObsTime
        self._outspec["defaultAddExoplanetObsTime"] = defaultAddExoplanetObsTime

        # If inputs are scalars, save scalars to outspec, otherwise save full lists
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping

        # initialize arrays updated in run_sim()
        self.initializeStorageArrays()

        # Generate File Hashnames and loction
        self.cachefname = self.generateHashfName(specs)

        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]

        # getting keepout map for entire mission
        startTime = self.TimeKeeping.missionStart.copy()
        endTime = self.TimeKeeping.missionFinishAbs.copy()

        nSystems = len(allModes)
        systNames = np.unique(
            [allModes[x]["syst"]["name"] for x in np.arange(nSystems)]
        )
        systOrder = np.argsort(systNames)
        koStr = ["koAngles_Sun", "koAngles_Moon", "koAngles_Earth", "koAngles_Small"]
        koangles = np.zeros([len(systNames), 4, 2])

        for x in systOrder:
            rel_mode = list(
                filter(lambda mode: mode["syst"]["name"] == systNames[x], allModes)
            )[0]
            koangles[x] = np.asarray([rel_mode["syst"][k] for k in koStr])

        self._outspec["nokoMap"] = nokoMap
        if not (nokoMap):
            koMaps, self.koTimes = self.Observatory.generate_koMap(
                TL, startTime, endTime, koangles
            )
            self.koMaps = {}
            for x, n in zip(systOrder, systNames[systOrder]):
                print(n)
                self.koMaps[n] = koMaps[x, :, :]

        self._outspec["nofZ"] = nofZ

        # TODO: do we still want a nofZ option?  probably not.
        self.fZmins = {}
        self.fZtypes = {}
        for x, n in zip(systOrder, systNames[systOrder]):
            self.fZmins[n] = np.array([])
            self.fZtypes[n] = np.array([])

        # TODO: do we need to do this for all modes? doing det only breaks other
        # schedulers, but maybe there's a better approach here.
        sInds = np.arange(TL.nStars)  # Initialize some sInds array
        for mode in allModes:
            # This instantiates fZMap arrays for every starlight suppresion system
            # that is actually used in a mode
            modeHashName = (
                f'{self.cachefname[0:-1]}_{TL.nStars}_{mode["syst"]["name"]}.'
            )

            if (np.size(self.fZmins[mode["syst"]["name"]]) == 0) or (
                np.size(self.fZtypes[mode["syst"]["name"]]) == 0
            ):
                self.ZodiacalLight.generate_fZ(
                    self.Observatory,
                    TL,
                    self.TimeKeeping,
                    mode,
                    modeHashName,
                    self.koTimes,
                )

                (
                    self.fZmins[mode["syst"]["name"]],
                    self.fZtypes[mode["syst"]["name"]],
                ) = self.ZodiacalLight.calcfZmin(
                    sInds,
                    self.Observatory,
                    TL,
                    self.TimeKeeping,
                    mode,
                    modeHashName,
                    self.koMaps[mode["syst"]["name"]],
                    self.koTimes,
                )

        # Precalculating intTimeFilter for coronagraph
        # find fZmin to use in intTimeFilter
        self.valfZmin, self.absTimefZmin = self.ZodiacalLight.extractfZmin(
            self.fZmins[det_mode["syst"]["name"]], sInds, self.koTimes
        )
        fEZ = self.ZodiacalLight.fEZ0  # grabbing fEZ0
        dMag = TL.int_dMag[sInds]  # grabbing dMag
        WA = TL.int_WA[sInds]  # grabbing WA
        self.intTimesIntTimeFilter = (
            self.OpticalSystem.calc_intTime(
                TL, sInds, self.valfZmin, fEZ, dMag, WA, det_mode, TK=TK
            )
            * det_mode["timeMultiplier"]
        )  # intTimes to filter by
        # These indices are acceptable for use simulating
        self.intTimeFilterInds = np.where(
            (
                (self.intTimesIntTimeFilter > 0)
                & (self.intTimesIntTimeFilter <= self.OpticalSystem.intCutoff)
            )
        )[0]

        self.make_debug_bird_plots = make_debug_bird_plots
        if self.make_debug_bird_plots:

            assert (
                debug_plot_path is not None
            ), "debug_plot_path must be set by input if make_debug_bird_plots is True"
            self.obs_plot_path = Path(f"{debug_plot_path}/{self.seed}")
            # Make directory if it doesn't exist
            if not self.obs_plot_path.exists():
                vprint(f"Making plot directory: {self.obs_plot_path}")
                self.obs_plot_path.mkdir(parents=True, exist_ok=True)
            self.obs_n_counter = 0

    def initializeStorageArrays(self):
        """
        Initialize all storage arrays based on # of stars and targets
        """

        self.DRM = []
        self.fullSpectra = np.zeros(self.SimulatedUniverse.nPlans, dtype=int)
        self.partialSpectra = np.zeros(self.SimulatedUniverse.nPlans, dtype=int)
        self.propagTimes = np.zeros(self.TargetList.nStars) * u.d
        self.lastObsTimes = np.zeros(self.TargetList.nStars) * u.d
        self.starVisits = np.zeros(
            self.TargetList.nStars, dtype=int
        )  # contains the number of times each star was visited
        self.starRevisit = np.array([])
        self.starExtended = np.array([], dtype=int)
        self.lastDetected = np.empty((self.TargetList.nStars, 4), dtype=object)

    def __str__(self):
        """String representation of the Survey Simulation object

        When the command 'print' is used on the Survey Simulation object, this
        method will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Survey Simulation class object attributes"

    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter:
            self.currentSep = Obs.occulterSep

        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(
            filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
        )
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s: survey beginning." % (TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        while not TK.mission_is_over(OS, Obs, det_mode):

            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd  # used to save sInd if returned sInd is None
            DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)

            if sInd is not None:
                ObsNum += (
                    1  # we're making an observation so increment observation number
                )

                if OS.haveOcculter:
                    # advance to start of observation (add slew time for selected target
                    _ = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)

                # beginning of observation, start to populate DRM
                DRM["star_ind"] = sInd
                DRM["star_name"] = TL.Name[sInd]
                DRM["arrival_time"] = TK.currentTimeNorm.to("day").copy()
                DRM["OB_nb"] = TK.OBnumber
                DRM["ObsNum"] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM["plan_inds"] = pInds.astype(int)
                log_obs = (
                    "  Observation #%s, star ind %s (of %s) with %s planet(s), "
                    + "mission time at Obs start: %s, exoplanetObsTime: %s"
                ) % (
                    ObsNum,
                    sInd,
                    TL.nStars,
                    len(pInds),
                    TK.currentTimeNorm.to("day").copy().round(2),
                    TK.exoplanetObsTime.to("day").copy().round(2),
                )
                self.logger.info(log_obs)
                self.vprint(log_obs)

                # PERFORM DETECTION and populate revisit list attribute
                (
                    detected,
                    det_fZ,
                    det_systemParams,
                    det_SNR,
                    FA,
                ) = self.observation_detection(sInd, det_intTime.copy(), det_mode)
                # update the occulter wet mass
                if OS.haveOcculter:
                    DRM = self.update_occulter_mass(
                        DRM, sInd, det_intTime.copy(), "det"
                    )
                # populate the DRM with detection results
                DRM["det_time"] = det_intTime.to("day")
                DRM["det_status"] = detected
                DRM["det_SNR"] = det_SNR
                DRM["det_fZ"] = det_fZ.to("1/arcsec2")
                DRM["det_params"] = det_systemParams

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                if char_mode["SNR"] not in [0, np.inf]:
                    (
                        characterized,
                        char_fZ,
                        char_systemParams,
                        char_SNR,
                        char_intTime,
                    ) = self.observation_characterization(sInd, char_mode)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if FA else len(pInds)
                    characterized = np.zeros(lenChar, dtype=float)
                    char_SNR = np.zeros(lenChar, dtype=float)
                    char_fZ = 0.0 / u.arcsec**2
                    char_systemParams = SU.dump_system_params(sInd)
                assert char_intTime != 0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter and (char_intTime is not None):
                    DRM = self.update_occulter_mass(DRM, sInd, char_intTime, "char")
                # populate the DRM with characterization results
                DRM["char_time"] = (
                    char_intTime.to("day") if char_intTime is not None else 0.0 * u.day
                )
                DRM["char_status"] = characterized[:-1] if FA else characterized
                DRM["char_SNR"] = char_SNR[:-1] if FA else char_SNR
                DRM["char_fZ"] = char_fZ.to("1/arcsec2")
                DRM["char_params"] = char_systemParams
                # populate the DRM with FA results
                DRM["FA_det_status"] = int(FA)
                DRM["FA_char_status"] = characterized[-1] if FA else 0
                DRM["FA_char_SNR"] = char_SNR[-1] if FA else 0.0
                DRM["FA_char_fEZ"] = (
                    self.lastDetected[sInd, 1][-1] / u.arcsec**2
                    if FA
                    else 0.0 / u.arcsec**2
                )
                DRM["FA_char_dMag"] = self.lastDetected[sInd, 2][-1] if FA else 0.0
                DRM["FA_char_WA"] = (
                    self.lastDetected[sInd, 3][-1] * u.arcsec if FA else 0.0 * u.arcsec
                )

                # populate the DRM with observation modes
                DRM["det_mode"] = dict(det_mode)
                del DRM["det_mode"]["inst"], DRM["det_mode"]["syst"]
                DRM["char_mode"] = dict(char_mode)
                del DRM["char_mode"]["inst"], DRM["char_mode"]["syst"]
                DRM["exoplanetObsTime"] = TK.exoplanetObsTime.copy()

                # append result values to self.DRM
                self.DRM.append(DRM)

                # handle case of inf OBs and missionPortion < 1
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1.0):
                    self.arbitrary_time_advancement(
                        TK.currentTimeNorm.to("day").copy() - DRM["arrival_time"]
                    )

            else:  # sInd == None
                sInd = old_sInd  # Retain the last observed star
                if (
                    TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]
                ):  # currentTime is at end of OB
                    # Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(
                        OS, Obs, det_mode
                    ):  # as long as the mission is not over
                        TK.advancetToStartOfNextOB()  # Advance To Start of Next OB
                elif waitTime is not None:
                    # CASE 1: Advance specific wait time
                    _ = TK.advanceToAbsTime(
                        TK.currentTimeAbs.copy() + waitTime,
                        self.defaultAddExoplanetObsTime,
                    )
                    self.vprint("waitTime is not None")
                else:
                    startTimes = (
                        TK.currentTimeAbs.copy() + np.zeros(TL.nStars) * u.d
                    )  # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(
                        TL,
                        np.arange(TL.nStars),
                        startTimes,
                        self.koMaps,
                        self.koTimes,
                        det_mode,
                    )[0]
                    # CASE 2 If There are no observable targets for the rest of the
                    # mission
                    if (
                        observableTimes[
                            (
                                TK.missionFinishAbs.copy().value * u.d
                                > observableTimes.value * u.d
                            )
                            * (
                                observableTimes.value * u.d
                                >= TK.currentTimeAbs.copy().value * u.d
                            )
                        ].shape[0]
                    ) == 0:
                        self.vprint(
                            (
                                "No Observable Targets for Remainder of mission at "
                                "currentTimeNorm = {}"
                            ).format(TK.currentTimeNorm)
                        )
                        # Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife
                        TK.currentTimeAbs = TK.missionFinishAbs
                    # CASE 3 nominal wait time if at least 1 target is still in list
                    # and observable
                    else:
                        # TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[
                            observableTimes.value * u.d
                            > TK.currentTimeAbs.copy().value * u.d
                        ]
                        # apply intTime filter
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1)
                        # apply revisit Filter #NOTE this means stars you added to the
                        # revisit list
                        inds3 = self.revisitFilter(
                            inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)
                        )
                        self.vprint(
                            "Filtering %d stars from advanceToAbsTime"
                            % (TL.nStars - len(inds3))
                        )
                        oTnowToEnd = observableTimes[inds3]
                        # there is at least one observableTime between now and the end
                        # of the mission
                        if not oTnowToEnd.value.shape[0] == 0:
                            # advance to that observable time
                            tAbs = np.min(oTnowToEnd)
                        else:
                            tAbs = (
                                TK.missionStart + TK.missionLife
                            )  # advance to end of mission
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        # Advance Time to this time OR start of next
                        # OB following this time
                        _ = TK.advanceToAbsTime(tAbs, self.defaultAddExoplanetObsTime)
                        self.vprint(
                            (
                                "No Observable Targets a currentTimeNorm= {:.2f}. "
                                "Advanced To {:.2f}"
                            ).format(
                                tmpcurrentTimeNorm.to("day"),
                                TK.currentTimeNorm.to("day"),
                            )
                        )
        else:  # TK.mission_is_over()
            dtsim = (time.time() - t0) * u.s
            log_end = (
                "Mission complete: no more time available.\n"
                + "Simulation duration: %s.\n" % dtsim.astype("int")
                + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            )
            self.logger.info(log_end)
            self.vprint(log_end)

    def arbitrary_time_advancement(self, dt):
        """Handles fully dynamically scheduled case where OBduration is infinite and
        missionPortion is less than 1.

        Args:
            dt (~astropy.units.quantity.Quantity):
                Total amount of time, including all overheads and extras used for the
                previous observation.

        Returns:
            None
        """

        self.TimeKeeping.allocate_time(
            dt
            * (1.0 - self.TimeKeeping.missionPortion)
            / self.TimeKeeping.missionPortion,
            addExoplanetObsTime=False,
        )

    def next_target(self, old_sInd, mode):
        """Finds index of next target star and calculates its integration time.

        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.

        Args:
            old_sInd (int):
                Index of the previous target star
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                DRM (dict):
                    Design Reference Mission, contains the results of one complete
                    observation (detection and characterization)
                sInd (int):
                    Index of next target star. Defaults to None.
                intTime (astropy.units.Quantity):
                    Selected star integration time for detection in units of day.
                    Defaults to None.
                waitTime (astropy.units.Quantity):
                    a strategically advantageous amount of time to wait in the case
                    of an occulter for slew times

        """
        OS = self.OpticalSystem
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        # create DRM
        DRM = {}

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # create appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        # fZs = np.zeros(TL.nStars) / u.arcsec**2.0
        dV = np.zeros(TL.nStars) * u.m / u.s
        intTimes = np.zeros(TL.nStars) * u.d
        obsTimes = np.zeros([2, TL.nStars]) * u.d
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        sd = None
        if OS.haveOcculter:
            sd = Obs.star_angularSep(TL, old_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs
            )

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # start times, including slew times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        # 2.5 Filter stars not observable at startTimes
        try:
            tmpIndsbool = list()
            for i in np.arange(len(sInds)):
                koTimeInd = np.where(
                    np.round(startTimes[sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    koMap[sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            sInds = sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722 # If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        # 3. filter out all previously (more-)visited targets, unless in
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # 4.1 calculate integration times for ALL preselected targets
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
        )  # Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            if OS.haveOcculter and (old_sInd is not None):
                (
                    sInds,
                    slewTimes[sInds],
                    intTimes[sInds],
                    dV[sInds],
                ) = self.refineOcculterSlews(
                    old_sInd, sInds, slewTimes, obsTimes, sd, mode
                )
                endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
            else:
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)
                sInds = sInds[
                    np.where(intTimes[sInds] <= maxIntTime)
                ]  # Filters targets exceeding end of OB
                endTimes = tmpCurrentTimeAbs.copy() + intTimes

                if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)

        # 5.1 TODO Add filter to filter out stars entering and exiting keepout
        # between startTimes and endTimes

        # 5.2 find spacecraft orbital END positions (for each candidate target),
        # and filter out unavailable targets
        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            # endTimes may exist past koTimes so we have an exception to hand this case
            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(
                        np.round(endTimes[sInds[i]].value) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        koMap[sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                sInds = np.asarray([], dtype=int)

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            sInd, waitTime = self.choose_next_target(
                old_sInd, sInds, slewTimes, intTimes[sInds]
            )

            # Should Choose Next Target decide there are no stars it wishes to
            # observe at this time.
            if sInd is None and (waitTime is not None):
                self.vprint(
                    "There are no stars available to observe. Waiting {}".format(
                        waitTime
                    )
                )
                return DRM, None, None, waitTime
            elif (sInd is None) and (waitTime is None):
                self.vprint(
                    "There are no stars available to observe and waitTime is None."
                )
                return DRM, None, None, waitTime
            # store selected star integration time
            intTime = intTimes[sInd]

        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint(
                "No Observable Targets at currentTimeNorm= "
                + str(TK.currentTimeNorm.copy())
            )
            return DRM, None, None, None

        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        # populate DRM with occulter related values
        if OS.haveOcculter:
            DRM = Obs.log_occulterResults(
                DRM, slewTimes[sInd], sInd, sd[sInd], dV[sInd]
            )
            return DRM, sInd, intTime, slewTimes[sInd]

        return DRM, sInd, intTime, waitTime

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Helper method for next_target to aid in overloading for alternative
        implementations.

        Given a subset of targets, calculate their integration times given the
        start of observation time.

        Prototype just calculates integration times for fixed contrast depth.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.
                must be of the same size as sInds
            mode (dict):
                Selected observing mode for detection

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Integration times for detection
                same dimension as sInds

        .. note::

            next_target filter will discard targets with zero integration times.

        """

        SU = self.SimulatedUniverse
        TL = self.TargetList

        # assumed values for detection
        fZ = self.ZodiacalLight.fZ(
            self.Observatory, self.TargetList, sInds, startTimes, mode
        )
        fEZ = self.ZodiacalLight.fEZ0
        fEZs = np.zeros(len(sInds)) / u.arcsec**2
        for i, sInd in enumerate(sInds):
            pInds = np.where(SU.plan2star == sInd)[0]
            pInds_earthlike = pInds[self.is_earthlike(pInds, sInd)]
            if len(pInds_earthlike) == 0:
                fEZs[i] = fEZ
            else:
                fEZs[i] = np.max(SU.fEZ[pInds_earthlike])
        dMag = TL.int_dMag[sInds]
        WA = TL.int_WA[sInds]

        # save out file containing photon count info
        if self.record_counts_path is not None and len(self.count_lines) == 0:
            C_p, C_b, C_sp, C_extra = self.OpticalSystem.Cp_Cb_Csp(
                self.TargetList, sInds, fZ, fEZs, dMag, WA, mode, returnExtra=True
            )
            import csv

            count_fpath = os.path.join(self.record_counts_path, "counts")

            if not os.path.exists(count_fpath):
                os.mkdir(count_fpath)

            outfile = os.path.join(count_fpath, str(self.seed) + ".csv")
            self.count_lines.append(
                [
                    "sInd",
                    "HIPs",
                    "C_F0",
                    "C_p0",
                    "C_sr",
                    "C_z",
                    "C_ez",
                    "C_dc",
                    "C_cc",
                    "C_rn",
                    "C_p",
                    "C_b",
                    "C_sp",
                ]
            )

            for i, sInd in enumerate(sInds):
                self.count_lines.append(
                    [
                        sInd,
                        self.TargetList.Name[sInd],
                        C_extra["C_F0"][0].value,
                        C_extra["C_sr"][i].value,
                        C_extra["C_z"][i].value,
                        C_extra["C_ez"][i].value,
                        C_extra["C_dc"][i].value,
                        C_extra["C_cc"][i].value,
                        C_extra["C_rn"][i].value,
                        C_p[i].value,
                        C_b[i].value,
                        C_sp[i].value,
                    ]
                )

            with open(outfile, "w") as csvfile:
                c = csv.writer(csvfile)
                c.writerows(self.count_lines)

        intTimes = self.OpticalSystem.calc_intTime(
            self.TargetList, sInds, fZ, fEZs, dMag, WA, mode
        )
        intTimes[~np.isfinite(intTimes)] = 0 * u.d

        return intTimes

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Helper method for method next_target to simplify alternative implementations.

        Given a subset of targets (pre-filtered by method next_target or some
        other means), select the best next one. The prototype uses completeness
        as the sole heuristic.

        Args:
            old_sInd (int):
                Index of the previous target star
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            slewTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                slew times to all stars (must be indexed by sInds)
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (int):
                    Index of next target star
                waitTime (:py:class:`~astropy.units.Quantity`):
                    Some strategic amount of time to wait in case an occulter slew is
                    desired (default is None)

        """

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        Obs = self.Observatory
        allModes = OS.observingModes

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm.copy() + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        # choose target with maximum completeness
        sInd = np.random.choice(sInds[comps == max(comps)])

        # Check if exoplanetObsTime would be exceeded
        mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
        )  # Maximum intTime allowed
        intTimes2 = self.calc_targ_intTime(
            np.array([sInd]), TK.currentTimeAbs.copy(), mode
        )
        if (
            intTimes2 > maxIntTime
        ):  # check if max allowed integration time would be exceeded
            self.vprint("max allowed integration time would be exceeded")
            sInd = None
            waitTime = 1.0 * u.d
            return sInd, waitTime

        return (
            sInd,
            slewTimes[sInd],
        )  # if coronagraph or first sInd, waitTime will be 0 days

    def refineOcculterSlews(self, old_sInd, sInds, slewTimes, obsTimes, sd, mode):
        """Refines/filters/chooses occulter slews based on time constraints

        Refines the selection of occulter slew times by filtering based on mission time
        constraints and selecting the best slew time for each star. This method calls on
        other occulter methods within SurveySimulation depending on how slew times were
        calculated prior to calling this function (i.e. depending on which
        implementation of the Observatory module is used).

        Args:
            old_sInd (int):
                Index of the previous target star
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            obsTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                A binary array with TargetList.nStars rows and
                (missionFinishAbs-missionStart)/dt columns
                where dt is 1 day by default. A value of 1 indicates the star is in
                keepout (and therefore cannot be observed). A value of 0 indicates the
                star is not in keepout and may be observed.
            sd (~astropy.units.Quantity(~numpy.ndarray(float))):
                Angular separation between stars in rad
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                sInds (int):
                    Indeces of next target star
                slewTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    slew times to all stars (must be indexed by sInds)
                intTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    Integration times for detection in units of day
                dV (astropy.units.Quantity(numpy.ndarray(float))):
                    Delta-V used to transfer to new star line of sight in unis of m/s
        """

        Obs = self.Observatory
        TL = self.TargetList

        # initializing arrays
        obsTimeArray = np.zeros([TL.nStars, 50]) * u.d
        intTimeArray = np.zeros([TL.nStars, 2]) * u.d

        for n in sInds:
            obsTimeArray[n, :] = (
                np.linspace(obsTimes[0, n].value, obsTimes[1, n].value, 50) * u.d
            )
        intTimeArray[sInds, 0] = self.calc_targ_intTime(
            sInds, Time(obsTimeArray[sInds, 0], format="mjd", scale="tai"), mode
        )
        intTimeArray[sInds, 1] = self.calc_targ_intTime(
            sInds, Time(obsTimeArray[sInds, -1], format="mjd", scale="tai"), mode
        )

        # determining which scheme to use to filter slews
        obsModName = Obs.__class__.__name__

        # slew times have not been calculated/decided yet (SotoStarshade)
        if obsModName == "SotoStarshade":
            sInds, intTimes, slewTimes, dV = self.findAllowableOcculterSlews(
                sInds,
                old_sInd,
                sd[sInds],
                slewTimes[sInds],
                obsTimeArray[sInds, :],
                intTimeArray[sInds, :],
                mode,
            )

        # slew times were calculated/decided beforehand (Observatory Prototype)
        else:
            sInds, intTimes, slewTimes = self.filterOcculterSlews(
                sInds,
                slewTimes[sInds],
                obsTimeArray[sInds, :],
                intTimeArray[sInds, :],
                mode,
            )
            dV = np.zeros(len(sInds)) * u.m / u.s

        return sInds, slewTimes, intTimes, dV

    def filterOcculterSlews(self, sInds, slewTimes, obsTimeArray, intTimeArray, mode):
        """Filters occulter slews that have already been calculated/selected.

        Used by the refineOcculterSlews method when slew times have been selected
        a priori. This method filters out slews that are not within desired observing
        blocks, the maximum allowed integration time, and are outside of future
        keepouts.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            slewTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                slew times to all stars (must be indexed by sInds)
            obsTimeArray (~astropy.units.Quantity(~numpy.ndarray(float))):
                Array of times during which a star is out of keepout, has shape
                nx50 where n is the number of stars in sInds. Unit of days
            intTimeArray (~astropy.units.Quantity(~numpy.ndarray(float))):
                Array of integration times for each time in obsTimeArray, has shape
                nx2 where n is the number of stars in sInds. Unit of days
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                sInds (int):
                    Indeces of next target star
                intTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    Integration times for detection in units of day
                slewTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    slew times to all stars (must be indexed by sInds)
        """

        TK = self.TimeKeeping
        Obs = self.Observatory

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # 0. lambda function that linearly interpolates Integration Time
        # between obsTimes
        linearInterp = lambda y, x, t: np.diff(y) / np.diff(x) * (  # noqa: E731
            t - np.array(x[:, 0]).reshape(len(t), 1)
        ) + np.array(y[:, 0]).reshape(len(t), 1)

        # 1. initializing arrays
        obsTimesRange = np.array(
            [obsTimeArray[:, 0], obsTimeArray[:, -1]]
        )  # nx2 array with start and end times of obsTimes for each star
        intTimesRange = np.array([intTimeArray[:, 0], intTimeArray[:, -1]])

        OBnumbers = np.zeros(
            [len(sInds), 1]
        )  # for each sInd, will show during which OB observations will take place
        maxIntTimes = np.zeros([len(sInds), 1]) * u.d

        intTimes = (
            linearInterp(
                intTimesRange.T,
                obsTimesRange.T,
                (tmpCurrentTimeAbs + slewTimes).reshape(len(sInds), 1).value,
            )
            * u.d
        )  # calculate intTimes for each slew time

        minObsTimeNorm = (obsTimesRange[0, :] - tmpCurrentTimeAbs.value).reshape(
            [len(sInds), 1]
        )
        maxObsTimeNorm = (obsTimesRange[1, :] - tmpCurrentTimeAbs.value).reshape(
            [len(sInds), 1]
        )
        ObsTimeRange = maxObsTimeNorm - minObsTimeNorm

        # 2. find OBnumber for each sInd's slew time
        if len(TK.OBendTimes) > 1:
            for i in range(len(sInds)):
                S = np.where(
                    TK.OBstartTimes.value - tmpCurrentTimeNorm.value
                    < slewTimes[i].value
                )[0][-1]
                F = np.where(
                    TK.OBendTimes.value - tmpCurrentTimeNorm.value < slewTimes[i].value
                )[0]

                # case when slews are in the first OB
                if F.shape[0] == 0:
                    F = -1
                else:
                    F = F[-1]

                # slew occurs within an OB (nth OB has started but hasn't ended)
                if S != F:
                    OBnumbers[i] = S
                    (
                        maxIntTimeOBendTime,
                        maxIntTimeExoplanetObsTime,
                        maxIntTimeMissionLife,
                    ) = TK.get_ObsDetectionMaxIntTime(Obs, mode, TK.OBstartTimes[S], S)
                    maxIntTimes[i] = min(
                        maxIntTimeOBendTime,
                        maxIntTimeExoplanetObsTime,
                        maxIntTimeMissionLife,
                    )  # Maximum intTime allowed

                # slew occurs between OBs, badbadnotgood
                else:
                    OBnumbers[i] = -1
                    maxIntTimes[i] = 0 * u.d
            OBstartTimeNorm = (
                TK.OBstartTimes[np.array(OBnumbers, dtype=int)].value
                - tmpCurrentTimeNorm.value
            )
        else:
            (
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
            ) = TK.get_ObsDetectionMaxIntTime(Obs, mode, tmpCurrentTimeNorm)
            maxIntTimes[:] = min(
                maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
            )  # Maximum intTime allowed
            OBstartTimeNorm = np.zeros(OBnumbers.shape)

        # finding the minimum possible slew time
        # (either OBstart or when star JUST leaves keepout)
        minAllowedSlewTime = np.max([OBstartTimeNorm, minObsTimeNorm], axis=0)

        # 3. start filtering process
        good_inds = np.where((OBnumbers >= 0) & (ObsTimeRange > intTimes.value))[0]
        # star slew within OB -AND- can finish observing
        # before it goes back into keepout
        if good_inds.shape[0] > 0:
            # the good ones
            sInds = sInds[good_inds]
            slewTimes = slewTimes[good_inds]
            intTimes = intTimes[good_inds]
            OBstartTimeNorm = OBstartTimeNorm[good_inds]
            minAllowedSlewTime = minAllowedSlewTime[good_inds]

            # maximum allowed slew time based on integration times
            maxAllowedSlewTime = maxIntTimes[good_inds].value - intTimes.value
            maxAllowedSlewTime[maxAllowedSlewTime < 0] = -np.inf
            maxAllowedSlewTime += OBstartTimeNorm  # calculated rel to currentTime norm

            # checking to see if slewTimes are allowed
            good_inds = np.where(
                (slewTimes.reshape([len(sInds), 1]).value > minAllowedSlewTime)
                & (slewTimes.reshape([len(sInds), 1]).value < maxAllowedSlewTime)
            )[0]

            slewTimes = slewTimes[good_inds]
        else:
            slewTimes = slewTimes[good_inds]

        return sInds[good_inds], intTimes[good_inds].flatten(), slewTimes

    def findAllowableOcculterSlews(
        self, sInds, old_sInd, sd, slewTimes, obsTimeArray, intTimeArray, mode
    ):
        """Finds an array of allowable slew times for each star

        Used by the refineOcculterSlews method when slew times have NOT been selected
        a priori. This method creates nx50 arrays (where the row corresponds to a
        specific star and the column corresponds to a future point in time relative to
        currentTime).

        These arrays are initially zero but are populated with the corresponding values
        (slews, intTimes, etc) if slewing to that time point (i.e. beginning an
        observation) would lead to a successful observation. A "successful observation"
        is defined by certain conditions relating to keepout and the komap, observing
        blocks, mission lifetime, and some constraints on the dVmap calculation in
        SotoStarshade. Each star will likely have a range of slewTimes that would lead
        to a successful observation -- another method is then called to select the best
        of these slewTimes.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            old_sInd (int):
                Index of the previous target star
            sd (~astropy.units.Quantity(~numpy.ndarray(float))):
                Angular separation between stars in rad
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            obsTimeArray (~astropy.units.Quantity(~numpy.ndarray(float))):
                Array of times during which a star is out of keepout, has shape
                nx50 where n is the number of stars in sInds
            intTimeArray (~astropy.units.Quantity(~numpy.ndarray(float))):
                Array of integration times for each time in obsTimeArray, has shape
                nx50 where n is the number of stars in sInds
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                sInds (numpy.ndarray(int)):
                    Indices of next target star
                slewTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    slew times to all stars (must be indexed by sInds)
                intTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    Integration times for detection in units of day
                dV (astropy.units.Quantity(numpy.ndarray(float))):
                    Delta-V used to transfer to new star line of sight in unis of m/s
        """
        TK = self.TimeKeeping
        Obs = self.Observatory
        TL = self.TargetList

        # 0. lambda function that linearly interpolates Integration
        # Time between obsTimes
        linearInterp = lambda y, x, t: np.diff(y) / np.diff(x) * (  # noqa: E731
            t - np.array(x[:, 0]).reshape(len(t), 1)
        ) + np.array(y[:, 0]).reshape(len(t), 1)

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # 1. initializing arrays
        obsTimes = np.array(
            [obsTimeArray[:, 0], obsTimeArray[:, -1]]
        )  # nx2 array with start and end times of obsTimes for each star
        intTimes_int = (
            np.zeros(obsTimeArray.shape) * u.d
        )  # initializing intTimes of shape nx50 then interpolating
        intTimes_int = (
            np.hstack(
                [
                    intTimeArray[:, 0].reshape(len(sInds), 1).value,
                    linearInterp(
                        intTimeArray.value, obsTimes.T, obsTimeArray[:, 1:].value
                    ),
                ]
            )
            * u.d
        )
        allowedSlewTimes = np.zeros(obsTimeArray.shape) * u.d
        allowedintTimes = np.zeros(obsTimeArray.shape) * u.d
        allowedCharTimes = np.zeros(obsTimeArray.shape) * u.d
        obsTimeArrayNorm = obsTimeArray.value - tmpCurrentTimeAbs.value

        # obsTimes -> relative to current Time
        try:
            minObsTimeNorm = np.array([np.min(v[v > 0]) for v in obsTimeArrayNorm])
        except:  # noqa: E722
            # an error pops up sometimes at the end of the mission, this fixes it
            # TODO: define the error type that occurs
            # rewrite to avoid a try/except if possible
            minObsTimeNorm = obsTimes[1, :].T - tmpCurrentTimeAbs.value

        maxObsTimeNorm = obsTimes[1, :].T - tmpCurrentTimeAbs.value
        ObsTimeRange = maxObsTimeNorm - minObsTimeNorm

        # getting max possible intTime
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, mode, tmpCurrentTimeNorm, TK.OBnumber)
        maxIntTime = min(
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
        )  # Maximum intTime allowed

        # 2. giant array of min and max slew times, starts at current time, ends when
        # stars enter keepout (all same size). Each entry either has a slew time value
        # if a slew is allowed at that date or 0 if slewing is not allowed

        # first filled in for the current OB
        minAllowedSlewTimes = np.array(
            [minObsTimeNorm.T] * len(intTimes_int.T)
        ).T  # just to make it nx50 so it plays nice with the other arrays
        maxAllowedSlewTimes = maxIntTime.value - intTimes_int.value
        maxAllowedSlewTimes[maxAllowedSlewTimes > Obs.occ_dtmax.value] = (
            Obs.occ_dtmax.value
        )

        # conditions that must be met to define an allowable slew time
        cond1 = (
            minAllowedSlewTimes >= Obs.occ_dtmin.value
        )  # minimum dt time in dV map interpolant
        cond2 = (
            maxAllowedSlewTimes <= Obs.occ_dtmax.value
        )  # maximum dt time in dV map interpolant
        cond3 = maxAllowedSlewTimes > minAllowedSlewTimes
        cond4 = intTimes_int.value < ObsTimeRange.reshape(len(sInds), 1)

        conds = cond1 & cond2 & cond3 & cond4
        minAllowedSlewTimes[np.invert(conds)] = (
            np.inf
        )  # these are filtered during the next filter
        maxAllowedSlewTimes[np.invert(conds)] = -np.inf

        # one last condition to meet
        map_i, map_j = np.where(
            (obsTimeArrayNorm > minAllowedSlewTimes)
            & (obsTimeArrayNorm < maxAllowedSlewTimes)
        )

        # 2.5 if any stars are slew-able to within this OB block, populate
        # "allowedSlewTimes", a running tally of possible slews
        # within the time range a star is observable (out of keepout)
        if map_i.shape[0] > 0 and map_j.shape[0] > 0:
            allowedSlewTimes[map_i, map_j] = obsTimeArrayNorm[map_i, map_j] * u.d
            allowedintTimes[map_i, map_j] = intTimes_int[map_i, map_j]
            allowedCharTimes[map_i, map_j] = maxIntTime - intTimes_int[map_i, map_j]

        # 3. search future OBs
        OB_withObsStars = (
            TK.OBstartTimes.value - np.min(obsTimeArrayNorm) - tmpCurrentTimeNorm.value
        )  # OBs within which any star is observable

        if any(OB_withObsStars > 0):
            nOBstart = np.argmin(np.abs(OB_withObsStars))
            nOBend = np.argmax(OB_withObsStars)

            # loop through the next 5 OBs (or until mission is over if there are less
            # than 5 OBs in the future)
            for i in np.arange(nOBstart, np.min([nOBend, nOBstart + 5])):

                # max int Times for the next OB
                (
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                ) = TK.get_ObsDetectionMaxIntTime(
                    Obs, mode, TK.OBstartTimes[i + 1], i + 1
                )
                maxIntTime_nOB = min(
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                )  # Maximum intTime allowed

                # min and max slew times rel to current Time (norm)
                nOBstartTimeNorm = np.array(
                    [TK.OBstartTimes[i + 1].value - tmpCurrentTimeNorm.value]
                    * len(sInds)
                )

                # min slew times for stars start either whenever the star first leaves
                # keepout or when next OB stars, whichever comes last
                minAllowedSlewTimes_nOB = np.array(
                    [np.max([minObsTimeNorm, nOBstartTimeNorm], axis=0).T]
                    * len(maxAllowedSlewTimes.T)
                ).T
                maxAllowedSlewTimes_nOB = (
                    nOBstartTimeNorm.reshape(len(sInds), 1)
                    + maxIntTime_nOB.value
                    - intTimes_int.value
                )
                maxAllowedSlewTimes_nOB[
                    maxAllowedSlewTimes_nOB > Obs.occ_dtmax.value
                ] = Obs.occ_dtmax.value

                # amount of time left when the stars are still out of keepout
                ObsTimeRange_nOB = (
                    maxObsTimeNorm
                    - np.max([minObsTimeNorm, nOBstartTimeNorm], axis=0).T
                )

                # condition to be met for an allowable slew time
                cond1 = minAllowedSlewTimes_nOB >= Obs.occ_dtmin.value
                cond2 = maxAllowedSlewTimes_nOB <= Obs.occ_dtmax.value
                cond3 = maxAllowedSlewTimes_nOB > minAllowedSlewTimes_nOB
                cond4 = intTimes_int.value < ObsTimeRange_nOB.reshape(len(sInds), 1)
                cond5 = intTimes_int.value < maxIntTime_nOB.value
                conds = cond1 & cond2 & cond3 & cond4 & cond5

                minAllowedSlewTimes_nOB[np.invert(conds)] = np.inf
                maxAllowedSlewTimes_nOB[np.invert(conds)] = -np.inf

                # one last condition
                map_i, map_j = np.where(
                    (obsTimeArrayNorm > minAllowedSlewTimes_nOB)
                    & (obsTimeArrayNorm < maxAllowedSlewTimes_nOB)
                )

                # 3.33 populate the running tally of allowable slew times if it meets
                # all conditions
                if map_i.shape[0] > 0 and map_j.shape[0] > 0:
                    allowedSlewTimes[map_i, map_j] = (
                        obsTimeArrayNorm[map_i, map_j] * u.d
                    )
                    allowedintTimes[map_i, map_j] = intTimes_int[map_i, map_j]
                    allowedCharTimes[map_i, map_j] = (
                        maxIntTime_nOB - intTimes_int[map_i, map_j]
                    )

        # 3.67 filter out any stars that are not observable at all
        filterDuds = np.sum(allowedSlewTimes, axis=1) > 0.0
        sInds = sInds[filterDuds]

        # 4. choose a slew time for each available star
        # calculate dVs for each possible slew time for each star
        allowed_dVs = Obs.calculate_dV(
            TL,
            old_sInd,
            sInds,
            sd[filterDuds],
            allowedSlewTimes[filterDuds, :],
            tmpCurrentTimeAbs,
        )

        if len(sInds.tolist()) > 0:
            # select slew time for each star
            dV_inds = np.arange(0, len(sInds))
            sInds, intTime, slewTime, dV = self.chooseOcculterSlewTimes(
                sInds,
                allowedSlewTimes[filterDuds, :],
                allowed_dVs[dV_inds, :],
                allowedintTimes[filterDuds, :],
                allowedCharTimes[filterDuds, :],
            )

            return sInds, intTime, slewTime, dV

        else:
            empty = np.asarray([], dtype=int)
            return empty, empty * u.d, empty * u.d, empty * u.m / u.s

    def chooseOcculterSlewTimes(self, sInds, slewTimes, dV, intTimes, charTimes):
        """Selects the best slew time for each star

        This method searches through an array of permissible slew times for
        each star and chooses the best slew time for the occulter based on
        maximizing possible characterization time for that particular star (as
        a default).

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            dV (~astropy.units.Quantity(~numpy.ndarray(float))):
                Delta-V used to transfer to new star line of sight in unis of m/s
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times for detection in units of day
            charTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Time left over after integration which could be used for
                characterization in units of day

        Returns:
            tuple:
                sInds (int):
                    Indeces of next target star
                slewTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    slew times to all stars (must be indexed by sInds)
                intTimes (astropy.units.Quantity(numpy.ndarray(float))):
                    Integration times for detection in units of day
                dV (astropy.units.Quantity(numpy.ndarray(float))):
                    Delta-V used to transfer to new star line of sight in unis of m/s
        """

        # selection criteria for each star slew
        good_j = np.argmax(
            charTimes, axis=1
        )  # maximum possible characterization time available
        good_i = np.arange(0, len(sInds))

        dV = dV[good_i, good_j]
        intTime = intTimes[good_i, good_j]
        slewTime = slewTimes[good_i, good_j]

        return sInds, intTime, slewTime, dV

    def observation_detection(self, sInd, intTime, mode):
        """Determines SNR and detection status for a given integration time
        for detection. Also updates the lastDetected and starRevisit lists.

        Args:
            sInd (int):
                Integer index of the star of interest
            intTime (~astropy.units.Quantity(~numpy.ndarray(float))):
                Selected star integration time for detection in units of day.
                Defaults to None.
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                detected (numpy.ndarray(int)):
                    Detection status for each planet orbiting the observed target star:
                    1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
                fZ (astropy.units.Quantity(numpy.ndarray(float))):
                    Surface brightness of local zodiacal light in units of 1/arcsec2
                systemParams (dict):
                    Dictionary of time-dependant planet properties averaged over the
                    duration of the integration
                SNR (numpy.darray(float)):
                    Detection signal-to-noise ratio of the observable planets
                FA (bool):
                    False alarm (false positive) boolean

        """

        PPop = self.PlanetPopulation
        ZL = self.ZodiacalLight
        PPro = self.PostProcessing
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # Save Current Time before attempting time allocation
        currentTimeNorm = TK.currentTimeNorm.copy()
        currentTimeAbs = TK.currentTimeAbs.copy()

        # Allocate Time
        extraTime = intTime * (mode["timeMultiplier"] - 1.0)  # calculates extraTime
        success = TK.allocate_time(
            intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], True
        )
        assert success, "Could not allocate observation detection time ({}).".format(
            intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        # calculates partial time to be added for every ntFlux
        dt = intTime / float(self.ntFlux)
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]

        # initialize outputs
        detected = np.array([], dtype=int)
        fZ = 0.0 / u.arcsec**2
        # write current system params by default
        systemParams = SU.dump_system_params(sInd)
        SNR = np.zeros(len(pInds))

        # if any planet, calculate SNR
        if len(pInds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros(self.ntFlux) / u.arcsec**2
            systemParamss = np.empty(self.ntFlux, dtype="object")
            Ss = np.zeros((self.ntFlux, len(pInds)))
            Ns = np.zeros((self.ntFlux, len(pInds)))
            # accounts for the time since the current time
            timePlus = Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
            # integrate the signal (planet flux) and noise
            for i in range(self.ntFlux):
                # allocate first half of dt
                timePlus += dt / 2.0
                # calculate current zodiacal light brightness
                fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                # propagate the system to match up with current time
                SU.propag_system(
                    sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                )
                self.propagTimes[sInd] = currentTimeNorm + timePlus
                # save planet parameters
                systemParamss[i] = SU.dump_system_params(sInd)
                # calculate signal and noise (electron count rates)
                Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                    sInd, pInds, dt, mode, fZ=fZs[i]
                )
                # allocate second half of dt
                timePlus += dt / 2.0

            # average output parameters
            fZ = np.mean(fZs)
            systemParams = {
                key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                / float(self.ntFlux)
                for key in sorted(systemParamss[0])
            }
            # calculate SNR
            S = Ss.sum(0)
            N = Ns.sum(0)
            SNR[N > 0] = S[N > 0] / N[N > 0]

        # if no planet, just save zodiacal brightness in the middle of the integration
        else:
            totTime = intTime * (mode["timeMultiplier"])
            fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs + totTime / 2.0, mode)[0]

        # find out if a false positive (false alarm) or any false negative
        # (missed detections) have occurred
        FA, MD = PPro.det_occur(SNR, mode, TL, sInd, intTime)

        # populate detection status array
        # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
        if len(pInds) > 0:
            detected = (~MD).astype(int)
            WA = (
                np.array(
                    [
                        systemParamss[x]["WA"].to("arcsec").value
                        for x in range(len(systemParamss))
                    ]
                )
                * u.arcsec
            )
            detected[np.all(WA < mode["IWA"], 0)] = -1
            detected[np.all(WA > mode["OWA"], 0)] = -2

        # if planets are detected, calculate the minimum apparent separation
        smin = None
        det = detected == 1  # If any of the planets around the star have been detected
        if np.any(det):
            smin = np.min(SU.s[pInds[det]])
            log_det = "   - Detected planet inds %s (%s/%s)" % (
                pInds[det],
                len(pInds[det]),
                len(pInds),
            )
            self.logger.info(log_det)
            self.vprint(log_det)

        # populate the lastDetected array by storing det, fEZ, dMag, and WA
        self.lastDetected[sInd, :] = [
            det,
            systemParams["fEZ"].to("1/arcsec2").value,
            systemParams["dMag"],
            systemParams["WA"].to("arcsec").value,
        ]

        # in case of a FA, generate a random delta mag (between PPro.FAdMag0 and
        # TL.intCutoff_dMag) and working angle (between IWA and min(OWA, a_max))
        if FA:
            WA = (
                np.random.uniform(
                    mode["IWA"].to("arcsec").value,
                    np.minimum(mode["OWA"], np.arctan(max(PPop.arange) / TL.dist[sInd]))
                    .to("arcsec")
                    .value,
                )
                * u.arcsec
            )
            dMag = np.random.uniform(PPro.FAdMag0(WA), TL.intCutoff_dMag)
            self.lastDetected[sInd, 0] = np.append(self.lastDetected[sInd, 0], True)
            self.lastDetected[sInd, 1] = np.append(
                self.lastDetected[sInd, 1], ZL.fEZ0.to("1/arcsec2").value
            )
            self.lastDetected[sInd, 2] = np.append(self.lastDetected[sInd, 2], dMag)
            self.lastDetected[sInd, 3] = np.append(
                self.lastDetected[sInd, 3], WA.to("arcsec").value
            )
            sminFA = np.tan(WA) * TL.dist[sInd].to("AU")
            smin = np.minimum(smin, sminFA) if smin is not None else sminFA
            log_FA = "   - False Alarm (WA=%s, dMag=%s)" % (
                np.round(WA, 3),
                np.round(dMag, 1),
            )
            self.logger.info(log_FA)
            self.vprint(log_FA)

        # Schedule Target Revisit
        self.scheduleRevisit(sInd, smin, det, pInds)

        if self.make_debug_bird_plots:
            from tools.obs_plot import obs_plot

            obs_plot(self, systemParams, mode, sInd, pInds, SNR, detected)

        return detected.astype(int), fZ, systemParams, SNR, FA

    def scheduleRevisit(self, sInd, smin, det, pInds):
        """A Helper Method for scheduling revisits after observation detection

        Updates self.starRevisit attribute only

        Args:
            sInd (int):
                sInd of the star just detected
            smin (~astropy.units.Quantity):
                minimum separation of the planet to star of planet just detected
            det (~np.ndarray(bool)):
                Detection status of all planets in target system
            pInds (~np.ndarray(int)):
                Indices of planets in the target system

        Returns:
            None

        """
        TK = self.TimeKeeping
        TL = self.TargetList
        SU = self.SimulatedUniverse

        # in both cases (detection or false alarm), schedule a revisit
        # based on minimum separation
        Ms = TL.MsTrue[sInd]
        if smin is not None:  # smin is None if no planet was detected
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G * (Mp + Ms)
            T = 2.0 * np.pi * np.sqrt(sp**3.0 / mu)
            t_rev = TK.currentTimeNorm.copy() + T / 2.0
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G * (Mp + Ms)
            T = 2.0 * np.pi * np.sqrt(sp**3.0 / mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75 * T

        if self.revisit_wait is not None:
            t_rev = TK.currentTimeNorm.copy() + self.revisit_wait
        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to("day").value])
        if self.starRevisit.size == 0:  # If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])  # initialize sterRevisit
        else:
            revInd = np.where(self.starRevisit[:, 0] == sInd)[
                0
            ]  # indices of the first column of the starRevisit list containing sInd
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd, 1] = revisit[1]

    def observation_characterization(self, sInd, mode):
        """Finds if characterizations are possible and relevant information

        Args:
            sInd (int):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization

        Returns:
            tuple:
                characterized (list(int)):
                    Characterization status for each planet orbiting the observed
                    target star including False Alarm if any, where 1 is full spectrum,
                    -1 partial spectrum, and 0 not characterized
                fZ (astropy.units.Quantity(numpy.ndarray(float))):
                    Surface brightness of local zodiacal light in units of 1/arcsec2
                systemParams (dict):
                    Dictionary of time-dependant planet properties averaged over the
                    duration of the integration
                SNR (numpy.ndarray(float)):
                    Characterization signal-to-noise ratio of the observable planets.
                    Defaults to None.
                intTime (astropy.units.Quantity(numpy.ndarray(float))):
                    Selected star characterization time in units of day.
                    Defaults to None.

        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # selecting appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]

        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd, 0]
        FA = len(det) == len(pInds) + 1
        if FA:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if there's anything (planet or FA)
        # to characterize
        characterized = np.zeros(len(det), dtype=int)
        fZ = 0.0 / u.arcsec**2.0
        # write current system params by default
        systemParams = SU.dump_system_params(sInd)
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0:  # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime

        # look for last detected planets that have not been fully characterized
        if not (FA):  # only true planets, no FA
            tochar = self.fullSpectra[pIndsDet] == 0
        else:  # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)

        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = (
                TK.currentTimeAbs.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )

            # if we're beyond mission end, bail out
            if startTime >= TK.missionFinishAbs:
                return characterized, fZ, systemParams, SNR, intTime

            startTimeNorm = (
                TK.currentTimeNorm.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[
                0
            ][
                0
            ]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times
        # at the detected fEZ, dMag, and WA
        if np.any(tochar):
            fZ = ZL.fZ(Obs, TL, [sInd], startTime, mode)
            fEZ = self.lastDetected[sInd, 1][det][tochar] / u.arcsec**2
            dMag = self.lastDetected[sInd, 2][det][tochar]
            WA = self.lastDetected[sInd, 3][det][tochar] * u.arcsec
            intTimes = np.zeros(len(tochar)) * u.day
            intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode, TK=TK)
            intTimes[~np.isfinite(intTimes)] = 0 * u.d
            # add a predetermined margin to the integration times
            intTimes = intTimes * (1.0 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes * (mode["timeMultiplier"])

            # Filter totTimes to make nan integration times correspond to the
            # maximum float value because Time cannot handle nan values
            totTimes[np.where(np.isnan(totTimes))[0]] = np.finfo(np.float64).max * u.d

            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = (
                (totTimes > 0)
                & (totTimes <= OS.intCutoff)
                & (endTimesNorm <= TK.OBendTimes[TK.OBnumber])
            )

        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            koTimeInds = np.zeros(len(endTimes.value[tochar]), dtype=int)
            # find index in koMap where each endTime is closest to koTimes
            for t, endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(
                        np.floor(endTime) - self.koTimes.value == 0
                    )[0]
                    koTimeInds[t] = (
                        endTimeInBounds[0] if endTimeInBounds.size != 0 else -1
                    )
                else:
                    koTimeInds[t] = np.where(
                        np.round(endTime) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]

        # 4/ if yes, allocate the overhead time, and perform the characterization
        # for the maximum char time
        if np.any(tochar):
            # Save Current Time before attempting time allocation
            currentTimeNorm = TK.currentTimeNorm.copy()
            currentTimeAbs = TK.currentTimeAbs.copy()

            # Allocate Time
            intTime = np.max(intTimes[tochar])
            extraTime = intTime * (mode["timeMultiplier"] - 1.0)
            success = TK.allocate_time(
                intTime + extraTime + mode["syst"]["ohTime"] + Obs.settlingTime, True
            )  # allocates time
            if not (success):  # Time was not successfully allocated
                char_intTime = None
                lenChar = len(pInds) + 1 if FA else len(pInds)
                characterized = np.zeros(lenChar, dtype=float)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0.0 / u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)
                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

            pIndsChar = pIndsDet[tochar]
            log_char = "   - Charact. planet inds %s (%s/%s detected)" % (
                pIndsChar,
                len(pIndsChar),
                len(pIndsDet),
            )
            self.logger.info(log_char)
            self.vprint(log_char)

            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux) / u.arcsec**2.0
                systemParamss = np.empty(self.ntFlux, dtype="object")
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime / float(self.ntFlux)
                timePlus = (
                    Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
                )  # accounts for the time since the current time
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    timePlus += dt / 2.0
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(
                        sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                    )
                    self.propagTimes[sInd] = currentTimeNorm + timePlus
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                        sInd, planinds, dt, mode, fZ=fZs[i]
                    )
                    # allocate second half of dt
                    timePlus += dt / 2.0

                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {
                    key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                    / float(self.ntFlux)
                    for key in sorted(systemParamss[0])
                }
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0] / N[N > 0]
                # allocate extra time for timeMultiplier

            # if only a FA, just save zodiacal brightness in the middle of the
            # integration
            else:
                totTime = intTime * (mode["timeMultiplier"])
                fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs + totTime / 2.0, mode)[0]

            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd, 1][-1] / u.arcsec**2.0
                dMag = self.lastDetected[sInd, 2][-1]
                WA = self.lastDetected[sInd, 3][-1] * u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode, TK=TK)
                S = (C_p * intTime).decompose().value
                N = np.sqrt((C_b * intTime + (C_sp * intTime) ** 2.0).decompose().value)
                SNRfa = S / N if N > 0.0 else 0.0

            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)

            # now, store characterization status: 1 for full spectrum,
            # -1 for partial spectrum, 0 for not characterized
            char = SNR >= mode["SNR"]
            # initialize with full spectra
            characterized = char.astype(int)
            WAchar = self.lastDetected[sInd, 3][char] * u.arcsec
            # find the current WAs of characterized planets
            WAs = systemParams["WA"]
            if FA:
                WAs = np.append(WAs, self.lastDetected[sInd, 3][-1] * u.arcsec)
            # check for partial spectra (for coronagraphs only)
            if not (mode["syst"]["occulter"]):
                IWA_max = mode["IWA"] * (1.0 + mode["BW"] / 2.0)
                OWA_min = mode["OWA"] * (1.0 - mode["BW"] / 2.0)
                char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
                characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        if self.make_debug_bird_plots:
            from tools.obs_plot import obs_plot

            obs_plot(self, systemParams, mode, sInd, pInds, SNR, characterized)

        return characterized.astype(int), fZ, systemParams, SNR, intTime

    def calc_signal_noise(
        self, sInd, pInds, t_int, mode, fZ=None, fEZ=None, dMag=None, WA=None
    ):
        """Calculates the signal and noise fluxes for a given time interval. Called
        by observation_detection and observation_characterization methods in the
        SurveySimulation module.

        Args:
            sInd (int):
                Integer index of the star of interest
            t_int (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration time interval in units of day
            pInds (int):
                Integer indices of the planets of interest
            mode (dict):
                Selected observing mode (from OpticalSystem)
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (~numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec

        Returns:
            tuple:
                Signal (float):
                    Counts of signal
                Noise (float):
                    Counts of background noise variance

        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # calculate optional parameters if not provided
        fZ = (
            fZ
            if (fZ is not None)
            else ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy(), mode)
        )
        fEZ = fEZ if (fEZ is not None) else SU.fEZ[pInds]

        # if lucky_planets, use lucky planet params for dMag and WA
        if SU.lucky_planets and mode in list(
            filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
        ):
            phi = (1 / np.pi) * np.ones(len(SU.d))
            dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[pInds]  # delta magnitude
            WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to("arcsec")[
                pInds
            ]  # working angle
        else:
            dMag = dMag if (dMag is not None) else SU.dMag[pInds]
            WA = WA if (WA is not None) else SU.WA[pInds]

        # initialize Signal and Noise arrays
        Signal = np.zeros(len(pInds))
        Noise = np.zeros(len(pInds))

        # find observable planets wrt IWA-OWA range
        obs = (WA > mode["IWA"]) & (WA < mode["OWA"])

        if np.any(obs):
            # find electron counts
            C_p, C_b, C_sp = OS.Cp_Cb_Csp(
                TL, sInd, fZ, fEZ[obs], dMag[obs], WA[obs], mode, TK=TK
            )
            # calculate signal and noise levels (based on Nemati14 formula)
            Signal[obs] = (C_p * t_int).decompose().value
            Noise[obs] = np.sqrt((C_b * t_int + (C_sp * t_int) ** 2).decompose().value)

        return Signal, Noise

    def update_occulter_mass(self, DRM, sInd, t_int, skMode):
        """Updates the occulter wet mass in the Observatory module, and stores all
        the occulter related values in the DRM array.

        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            sInd (int):
                Integer index of the star of interest
            t_int (~astropy.units.Quantity(~numpy.ndarray(float))):
                Selected star integration time (for detection or characterization)
                in units of day
            skMode (str):
                Station keeping observing mode type ('det' or 'char')

        Returns:
            dict:
                Design Reference Mission dictionary, contains the results of one
                complete observation

        """

        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        assert skMode in ("det", "char"), "Observing mode type must be 'det' or 'char'."

        # decrement mass for station-keeping
        dF_lateral, dF_axial, intMdot, mass_used, deltaV = Obs.mass_dec_sk(
            TL, sInd, TK.currentTimeAbs.copy(), t_int
        )

        DRM[skMode + "_dV"] = deltaV.to("m/s")
        DRM[skMode + "_mass_used"] = mass_used.to("kg")
        DRM[skMode + "_dF_lateral"] = dF_lateral.to("N")
        DRM[skMode + "_dF_axial"] = dF_axial.to("N")
        # update spacecraft mass
        Obs.scMass = Obs.scMass - mass_used
        DRM["scMass"] = Obs.scMass.to("kg")
        if Obs.twotanks:
            Obs.skMass = Obs.skMass - mass_used
            DRM["skMass"] = Obs.skMass.to("kg")

        return DRM

    def reset_sim(self, genNewPlanets=True, rewindPlanets=True, seed=None):
        """Performs a full reset of the current simulation.

        This will reinitialize the TimeKeeping, Observatory, and SurveySimulation
        objects with their own outspecs.

        Args:
            genNewPlanets (bool):
                Generate all new planets based on the original input specification.
                If False, then the original planets will remain. Setting to True forces
                ``rewindPlanets`` to be True as well. Defaults True.
            rewindPlanets (bool):
                Reset the current set of planets to their original orbital phases.
                If both genNewPlanets and rewindPlanet  are False, the original planets
                will be retained and will not be rewound to their initial starting
                locations (i.e., all systems will remain at the times they were at the
                end of the last run, thereby effectively randomizing planet phases.
                Defaults True.
            seed (int, optional):
                Random seed to use for all random number generation. If None (default)
                a new random seed will be generated when re-initializing the
                SurveySimulation.

        """

        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        TL = self.TargetList

        # re-initialize SurveySimulation arrays
        specs = self._outspec
        specs["modules"] = self.modules

        if seed is None:  # pop the seed so a new one is generated
            if "seed" in specs:
                specs.pop("seed")
        else:  # if seed is provided, replace seed with provided seed
            specs["seed"] = seed

        # reset mission time, re-init surveysim and observatory
        TK.__init__(**TK._outspec)
        self.__init__(**specs)
        self.Observatory.__init__(**self.Observatory._outspec)

        # generate new planets if requested (default)
        if genNewPlanets:
            TL.stellar_mass()
            TL.I = TL.gen_inclinations(TL.PlanetPopulation.Irange)  # noqa: E741
            SU.gen_physical_properties(**SU._outspec)
            rewindPlanets = True
        # re-initialize systems if requested (default)
        if rewindPlanets:
            SU.init_systems()

        # reset helper arrays
        self.initializeStorageArrays()

        self.vprint("Simulation reset.")

    def genOutSpec(
        self,
        tofile: Optional[str] = None,
        starting_outspec: Optional[Dict[str, Any]] = None,
        modnames: bool = False,
    ) -> Dict[str, Any]:
        """Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.

        Args:
            tofile (str, optional):
                Name of the file containing all output specifications (outspecs).
                Defaults to None.
            starting_outspec (dict, optional):
                Initial outspec (from MissionSim). Defaults to None.
            modnames (bool):
                If True, populate outspec dictionary with the module it originated from,
                instead of the actual value of the keyword. Defaults False.

        Returns:
            dict:
                Dictionary containing the full :ref:`sec:inputspec`, including all
                filled-in default values. Combination of all individual module _outspec
                attributes.

        """

        # start with our own outspec
        if modnames:
            out = copy.copy(self._outspec)
            for k in out:
                out[k] = self.__class__.__name__
        else:
            out = copy.deepcopy(self._outspec)

        # Add any provided other outspec
        if starting_outspec is not None:
            out.update(starting_outspec)

        # add in all modules _outspec's
        for module in self.modules.values():
            if modnames:
                tmp = copy.copy(module._outspec)
                for k in tmp:
                    tmp[k] = module.__class__.__name__
            else:
                tmp = module._outspec
            out.update(tmp)

        # add in the specific module names used
        out["modules"] = {}
        for mod_name, module in self.modules.items():
            # find the module file
            mod_name_full = module.__module__
            if mod_name_full.startswith("EXOSIMS"):
                # take just its short name if it is in EXOSIMS
                mod_name_short = mod_name_full.split(".")[-1]
            else:
                # take its full path if it is not in EXOSIMS - changing .pyc -> .py
                mod_name_short = re.sub(
                    r"\.pyc$", ".py", inspect.getfile(module.__class__)
                )
            out["modules"][mod_name] = mod_name_short
        # add catalog name
        if self.TargetList.keepStarCatalog:
            module = self.TargetList.StarCatalog
            mod_name_full = module.__module__
            if mod_name_full.startswith("EXOSIMS"):
                # take just its short name if it is in EXOSIMS
                mod_name_short = mod_name_full.split(".")[-1]
            else:
                # take its full path if it is not in EXOSIMS - changing .pyc -> .py
                mod_name_short = re.sub(
                    r"\.pyc$", ".py", inspect.getfile(module.__class__)
                )
            out["modules"][mod_name] = mod_name_short
        else:
            out["modules"][
                "StarCatalog"
            ] = self.TargetList.StarCatalog  # we just copy the StarCatalog string

        # if we don't know about the SurveyEnsemble, just write a blank to the output
        if "SurveyEnsemble" not in out["modules"]:
            out["modules"]["SurveyEnsemble"] = " "

        # add in the SVN/Git revision
        path = os.path.split(inspect.getfile(self.__class__))[0]
        path = os.path.split(os.path.split(path)[0])[0]
        # handle case where EXOSIMS was imported from the working directory
        if path == "":
            path = os.getcwd()
        # comm = "git -C " + path + " log -1"
        comm = "git --git-dir=%s --work-tree=%s log -1" % (
            os.path.join(path, ".git"),
            path,
        )
        rev = subprocess.Popen(
            comm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        (gitRev, err) = rev.communicate()
        gitRev = gitRev.decode("utf-8")
        if isinstance(gitRev, str) & (len(gitRev) > 0):
            tmp = re.compile(
                r"\S*(commit [0-9a-fA-F]+)\n[\s\S]*Date: ([\S ]*)\n"
            ).match(gitRev)
            if tmp:
                out["Revision"] = "Github " + tmp.groups()[0] + " " + tmp.groups()[1]
        else:
            rev = subprocess.Popen(
                "svn info " + path + "| grep \"Revision\" | awk '{print $2}'",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            (svnRev, err) = rev.communicate()
            if isinstance(svnRev, str) & (len(svnRev) > 0):
                out["Revision"] = "SVN revision is " + svnRev[:-1]
            else:
                out["Revision"] = "Not a valid Github or SVN revision."

        # dump to file
        if tofile is not None:
            with open(tofile, "w") as outfile:
                json.dump(
                    out,
                    outfile,
                    sort_keys=True,
                    indent=4,
                    ensure_ascii=False,
                    separators=(",", ": "),
                    default=array_encoder,
                )

        return out

    def generateHashfName(self, specs):
        """Generate cached file Hashname

        Requires a .XXX appended to end of hashname for each individual use case

        Args:
            specs (dict):
                :ref:`sec:inputspec`

        Returns:
            str:
                Unique indentifier string for cahce products from this set of modules
                and inputs
        """
        # Allows cachefname to be predefined
        if "cachefname" in specs:
            return specs["cachefname"]

        cachefname = ""  # declares cachefname
        mods = ["Completeness", "TargetList", "OpticalSystem"]  # modules to look at
        tmp = (
            self.Completeness.PlanetPopulation.__class__.__name__
            + self.Completeness.PlanetPhysicalModel.__class__.__name__
            + self.PlanetPopulation.__class__.__name__
            + self.SimulatedUniverse.__class__.__name__
            + self.PlanetPhysicalModel.__class__.__name__
        )

        if "selectionMetric" in specs:
            tmp += specs["selectionMetric"]
        if "Izod" in specs:
            tmp += specs["Izod"]
        if "maxiter" in specs:
            tmp += str(specs["maxiter"])
        if "ftol" in specs:
            tmp += str(specs["ftol"])
        if "missionLife" in specs:
            tmp += str(specs["missionLife"])
        if "missionPortion" in specs:
            tmp += str(specs["missionPortion"])
        if "smaknee" in specs:
            tmp += str(specs["smaknee"])
        if "koAngleMax" in specs:
            tmp += str(specs["koAngleMax"])
        tmp += str(np.sum(self.Completeness.PlanetPopulation.arange.value))
        tmp += str(np.sum(self.Completeness.PlanetPopulation.Rprange.value))
        tmp += str(np.sum(self.Completeness.PlanetPopulation.erange))
        tmp += str(
            self.Completeness.PlanetPopulation.PlanetPhysicalModel.whichPlanetPhaseFunction  # noqa: E501
        )
        tmp += str(np.sum(self.PlanetPopulation.arange.value))
        tmp += str(np.sum(self.PlanetPopulation.Rprange.value))
        tmp += str(np.sum(self.PlanetPopulation.erange))
        tmp += str(self.PlanetPopulation.PlanetPhysicalModel.whichPlanetPhaseFunction)

        for mod in mods:
            cachefname += self.modules[mod].__module__.split(".")[
                -1
            ]  # add module name to end of cachefname
        cachefname += hashlib.md5(
            (str(self.TargetList.Name) + tmp).encode("utf-8")
        ).hexdigest()  # turn cachefname into hashlib
        cachefname = os.path.join(
            self.cachedir, cachefname + os.extsep
        )  # join into filepath and fname
        # Needs file terminator (.starkt0, .t0, etc) appended done by each individual
        # use case.
        return cachefname

    def revisitFilter(self, sInds, tmpCurrentTimeNorm):
        """Helper method for Overloading Revisit Filtering

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of stars still in observation list
            tmpCurrentTimeNorm (astropy.units.Quantity):
                Normalized simulation time

        Returns:
            ~numpy.ndarray(int):
                indices of stars still in observation list
        """

        tovisit = np.zeros(self.TargetList.nStars, dtype=bool)
        if len(sInds) > 0:
            # Check that no star has exceeded the number of revisits and the indicies
            # of all considered stars have minimum number of observations
            # This should prevent revisits so long as all stars have not
            # been observed
            tovisit[sInds] = (self.starVisits[sInds] == min(self.starVisits[sInds])) & (
                self.starVisits[sInds] < self.nVisitsMax
            )
            if self.starRevisit.size != 0:
                ind_rev = self.revisit_inds(sInds, tmpCurrentTimeNorm)
                tovisit[ind_rev] = self.starVisits[ind_rev] < self.nVisitsMax
            sInds = np.where(tovisit)[0]
        return sInds

    def is_earthlike(self, plan_inds, sInd):
        """Is the planet earthlike? Returns boolean array that's True for Earth-like
        planets.


        Args:
            plan_inds(~numpy.ndarray(int)):
                Planet indices
            sInd (int):
                Star index

        Returns:
            ~numpy.ndarray(bool):
                Array of same dimension as plan_inds input that's True for Earthlike
                planets and false otherwise.
        """
        TL = self.TargetList
        SU = self.SimulatedUniverse
        PPop = self.PlanetPopulation

        # extract planet and star properties
        Rp_plan = SU.Rp[plan_inds].value
        L_star = TL.L[sInd]
        if PPop.scaleOrbits:
            a_plan = (SU.a[plan_inds] / np.sqrt(L_star)).value
        else:
            a_plan = (SU.a[plan_inds]).value
        # Definition: planet radius (in earth radii) and solar-equivalent luminosity
        # must be between the given bounds.
        Rp_plan_lo = 0.80 / np.sqrt(a_plan)
        # We use the numpy versions so that plan_ind can be a numpy vector.
        return np.logical_and(
            np.logical_and(Rp_plan >= Rp_plan_lo, Rp_plan <= 1.4),
            np.logical_and(a_plan >= 0.95, a_plan <= 1.67),
        )

    def find_known_plans(self):
        """
        Find and return list of known RV stars and list of stars with earthlike planets
        based on info from David Plavchan dated 12/24/2018
        """
        TL = self.TargetList
        SU = self.SimulatedUniverse
        PPop = self.PlanetPopulation
        L_star = TL.L[SU.plan2star]

        c = 28.4 * u.m / u.s
        Mj = 317.8 * u.earthMass
        Mpj = SU.Mp / Mj  # planet masses in jupiter mass units
        Ms = TL.MsTrue[SU.plan2star]
        Teff = TL.Teff[SU.plan2star]
        mu = const.G * (SU.Mp + Ms)
        T = (2.0 * np.pi * np.sqrt(SU.a**3 / mu)).to(u.yr)
        e = SU.e

        # pinds in correct temp range
        t_filt = np.where((Teff.value > 3000) & (Teff.value < 6800))[0]

        K = (
            (c / np.sqrt(1 - e[t_filt]))
            * Mpj[t_filt]
            * np.sin(SU.I[t_filt])
            * Ms[t_filt] ** (-2 / 3)
            * T[t_filt] ** (-1 / 3)
        )

        K_filter = (T[t_filt].to(u.d) / 10**4).value  # create period-filter
        # if period-filter value is lower than .03, set to .03
        K_filter[np.where(K_filter < 0.03)[0]] = 0.03
        k_filt = t_filt[np.where(K.value > K_filter)[0]]  # pinds in the correct K range

        if PPop.scaleOrbits:
            a_plan = (SU.a / np.sqrt(L_star)).value
        else:
            a_plan = SU.a.value

        Rp_plan_lo = 0.80 / np.sqrt(a_plan)

        # pinds in habitable zone
        a_filt = k_filt[np.where((a_plan[k_filt] > 0.95) & (a_plan[k_filt] < 1.67))[0]]
        # rocky planets
        r_filt = a_filt[
            np.where(
                (SU.Rp.value[a_filt] >= Rp_plan_lo[a_filt])
                & (SU.Rp.value[a_filt] < 1.4)
            )[0]
        ]
        self.known_earths = np.union1d(self.known_earths, r_filt).astype(int)

        # these are known_rv stars with earths around them
        known_stars = np.unique(SU.plan2star[k_filt])  # these are known_rv stars
        known_rocky = np.unique(SU.plan2star[r_filt])

        # if include_known_RV, then filter out all other sInds
        if self.include_known_RV is not None:
            HIP_sInds = np.where(np.in1d(TL.Name, self.include_known_RV))[0]
            known_stars = np.intersect1d(HIP_sInds, known_stars)
            known_rocky = np.intersect1d(HIP_sInds, known_rocky)
        return known_stars.astype(int), known_rocky.astype(int)

    def find_char_SNR(self, sInd, pIndsChar, startTime, intTime, mode):
        """Finds the SNR achieved by an observing mode after intTime days

        The observation window (which includes settling and overhead times)
        is a superset of the integration window (in which photons are collected).

        The observation window begins at startTime. The integration window
        begins at startTime + Obs.settlingTime + mode["ohTime"],
        and the integration itself has a duration of intTime.

        Args:
            sInd (int):
                Integer index of the star of interest
            pIndsChar (int numpy.ndarray):
                Observable planets to characterize. Place (-1) at end to put
                False Alarm parameters at end of returned tuples.
            startTime (astropy.units.Quantity):
                Beginning of observation window in units of day.
            intTime (astropy.units.Quantity):
                Selected star characterization integration time in units of day.
            mode (dict):
                Observing mode for the characterization

        Returns:
            tuple:
                SNR (float numpy.ndarray):
                    Characterization signal-to-noise ratio of the observable planets.
                    Defaults to None. [TBD]
                fZ (astropy.units.Quantity):
                    Surface brightness of local zodiacal light in units of 1/arcsec2
                systemParams (dict):
                    Dictionary of time-dependent planet properties averaged over the
                    duration of the integration.

        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # time at start of integration window
        currentTimeNorm = startTime
        currentTimeAbs = TK.missionStart + startTime

        # first, calculate SNR for observable planets (without false alarm)
        planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
        SNRplans = np.zeros(len(planinds))
        if len(planinds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros(self.ntFlux) / u.arcsec**2.0
            systemParamss = np.empty(self.ntFlux, dtype="object")
            Ss = np.zeros((self.ntFlux, len(planinds)))
            Ns = np.zeros((self.ntFlux, len(planinds)))
            # integrate the signal (planet flux) and noise
            dt = intTime / float(self.ntFlux)
            timePlus = (
                Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
            )  # accounts for the time since the current time
            for i in range(self.ntFlux):
                # calculate signal and noise (electron count rates)
                if SU.lucky_planets:
                    fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs, mode)[0]
                    Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                        sInd, planinds, dt, mode, fZ=fZs[i]
                    )
                # allocate first half of dt
                timePlus += dt / 2.0
                # calculate current zodiacal light brightness
                fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                # propagate the system to match up with current time
                SU.propag_system(
                    sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                )
                self.propagTimes[sInd] = currentTimeNorm + timePlus
                # time-varying planet params (WA, dMag, phi, fEZ, d)
                systemParamss[i] = SU.dump_system_params(sInd)
                # calculate signal and noise (electron count rates)
                if not SU.lucky_planets:
                    Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                        sInd, planinds, dt, mode, fZ=fZs[i]
                    )
                # allocate second half of dt
                timePlus += dt / 2.0

            # average output parameters
            fZ = np.mean(fZs)
            systemParams = {
                key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                / float(self.ntFlux)
                for key in sorted(systemParamss[0])
            }
            # calculate planets SNR
            S = Ss.sum(0)
            N = Ns.sum(0)
            SNRplans[N > 0] = S[N > 0] / N[N > 0]
            # allocate extra time for timeMultiplier

        # if only a FA, just save zodiacal brightness
        # in the middle of the integration
        else:
            totTime = intTime * (mode["timeMultiplier"])
            fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs.copy() + totTime / 2.0, mode)[0]

        # calculate the false alarm SNR (if any)
        SNRfa = []
        if pIndsChar[-1] == -1:
            # Note: these attributes may not exist for all schedulers
            fEZ = self.lastDetected[sInd, 1][-1] / u.arcsec**2.0
            dMag = self.lastDetected[sInd, 2][-1]
            WA = self.lastDetected[sInd, 3][-1] * u.arcsec
            C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
            S = (C_p * intTime).decompose().value
            N = np.sqrt((C_b * intTime + (C_sp * intTime) ** 2.0).decompose().value)
            SNRfa = S / N if N > 0.0 else 0.0

        # SNR vector is of length (#planets), plus 1 if FA
        # This routine leaves SNR = 0 if unknown or not found
        pInds = np.where(SU.plan2star == sInd)[0]
        FA_present = pIndsChar[-1] == -1
        # there will be one SNR for each entry of pInds_FA
        pInds_FA = np.append(pInds, [-1] if FA_present else np.zeros(0, dtype=int))

        # boolean index vector into SNR
        #   True iff we have computed a SNR for that planet
        #   False iff we didn't find an SNR (and will leave 0 there)
        #   if FA_present, SNR_plug_in[-1] = True
        SNR_plug_in = np.isin(pInds_FA, pIndsChar)

        # save all SNRs (planets and FA) to one array
        SNR = np.zeros(len(pInds_FA))
        # plug in the SNR's we computed (pIndsChar and optional FA)
        SNR[SNR_plug_in] = np.append(SNRplans, SNRfa)

        return SNR, fZ, systemParams

    def revisit_inds(self, sInds, tmpCurrentTimeNorm):
        """Return subset of star indices that are scheduled for revisit.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of stars to consider
            tmpCurrentTimeNorm (astropy.units.Quantity):
                Normalized simulation time

        Returns:
            ~numpy.ndarray(int):
                Indices of stars whose revisit is scheduled for within `self.dt_max` of
                the current time

        """
        dt_rev = np.abs(self.starRevisit[:, 1] * u.day - tmpCurrentTimeNorm)
        ind_rev = [
            int(x) for x in self.starRevisit[dt_rev < self.dt_max, 0] if x in sInds
        ]

        return ind_rev

    def keepout_filter(self, sInds, startTimes, endTimes, koMap):
        """Filters stars not observable due to keepout

        Args:
            sInds (~numpy.ndarray(int)):
                indices of stars still in observation list
            startTimes (~numpy.ndarray(float)):
                start times of observations
            endTimes (~numpy.ndarray(float)):
                end times of observations
            koMap (~numpy.ndarray(bool)):
                map where True is for a target unobstructed and observable,
                False is for a target obstructed and unobservable due to keepout zone

        Returns:
            ~numpy.ndarray(int):
                sInds - filtered indices of stars still in observation list

        """
        # find the indices of keepout times that pertain to the start and end times of
        # observation
        startInds = np.searchsorted(self.koTimes.value, startTimes)
        endInds = np.searchsorted(self.koTimes.value, endTimes)

        # boolean array of available targets (unobstructed between observation start and
        # end time) we include a -1 in the start and +1 in the end to include keepout
        # days enclosing start and end times
        availableTargets = np.array(
            [
                np.all(
                    koMap[
                        sInd,
                        max(startInds[sInd] - 1, 0) : min(
                            endInds[sInd] + 1, len(self.koTimes.value) + 1
                        ),
                    ]
                )
                for sInd in sInds
            ],
            dtype="bool",
        )

        return sInds[availableTargets]


def array_encoder(obj):
    r"""Encodes numpy arrays, astropy Times, and astropy Quantities, into JSON.

    Called from json.dump for types that it does not already know how to represent,
    like astropy Quantity's, numpy arrays, etc.  The json.dump() method encodes types
    like ints, strings, and lists itself, so this code does not see these types.
    Likewise, this routine can and does return such objects, which is OK as long as
    they unpack recursively into types for which encoding is known

    Args:
        obj (Any):
            Object to encode.

        Returns:
            Any:
                Encoded object

    """

    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    if isinstance(obj, Time):
        # astropy Time -> time string
        return obj.fits  # isot also makes sense here
    if isinstance(obj, u.quantity.Quantity):
        # note: it is possible to have a numpy ndarray wrapped in a Quantity.
        # NB: alternatively, can return (obj.value, obj.unit.name)
        return obj.value
    if isinstance(obj, SkyCoord):
        return dict(
            lon=obj.heliocentrictrueecliptic.lon.value,
            lat=obj.heliocentrictrueecliptic.lat.value,
            distance=obj.heliocentrictrueecliptic.distance.value,
        )
    if isinstance(obj, (np.ndarray, np.number)):
        # ndarray -> list of numbers
        return obj.tolist()
    if isinstance(obj, complex):
        # complex -> (real, imag) pair
        return [obj.real, obj.imag]
    if callable(obj):
        # this case occurs for interpolants like PSF and QE
        # We cannot simply "write" the function to JSON, so we make up a string
        # to keep from throwing an error.
        # The fix is simple: when generating the interpolant, add a _outspec attribute
        # to the function (or the lambda), containing (e.g.) the fits filename, or the
        # explicit number -- whatever string was used.  Then, here, check for that
        # attribute and write it out instead of this dummy string.  (Attributes can
        # be transparently attached to python functions, even lambda's.)
        return "interpolant_function"
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode()
    # an EXOSIMS object
    if hasattr(obj, "_modtype"):
        return obj.__dict__
    # an object for which no encoding is defined yet
    #   as noted above, ordinary types (lists, ints, floats) do not take this path
    raise ValueError("Could not JSON-encode an object of type %s" % type(obj))
