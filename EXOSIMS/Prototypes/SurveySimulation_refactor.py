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

        # defaultAddExoplanetObsTime Tells us time advanced when no targets available
        # counts agains exoplanetObsTime (when True)
        self.defaultAddExoplanetObsTime = defaultAddExoplanetObsTime
        self._outspec["defaultAddExoplanetObsTime"] = defaultAddExoplanetObsTime

        # initialize arrays updated in run_sim()
        self.initializeStorageArrays()

        # Generate File Hashnames and loction
        self.cachefname = self.generateHashfName(specs)

        # get keepout and fZ maps for entire mission
        self.gen_maps()

    def gen_maps(self):
        """Generate keepout and zodiacal light maps for whole mission"""

        startTime = self.TimeKeeping.missionStart.copy()
        endTime = self.TimeKeeping.missionFinishAbs.copy()

        nSystems = len(self.OpticalSystem.observingModes)
        systNames = np.unique(
            [
                self.OpticalSystem.observingModes[x]["syst"]["name"]
                for x in np.arange(nSystems)
            ]
        )
        systOrder = np.argsort(systNames)
        koStr = ["koAngles_Sun", "koAngles_Moon", "koAngles_Earth", "koAngles_Small"]
        koangles = np.zeros([len(systNames), 4, 2])

        for x in systOrder:
            rel_mode = list(
                filter(
                    lambda mode: mode["syst"]["name"] == systNames[x],
                    self.OpticalSystem.observingModes,
                )
            )[0]
            koangles[x] = np.asarray([rel_mode["syst"][k] for k in koStr])

        koMaps, self.koTimes = self.Observatory.generate_koMap(
            self.TargetList, startTime, endTime, koangles
        )
        self.koMaps = {}
        for x, n in zip(systOrder, systNames[systOrder]):
            print(n)
            self.koMaps[n] = koMaps[x, :, :]

        self.fZmins = {}
        self.fZtypes = {}
        for x, n in zip(systOrder, systNames[systOrder]):
            self.fZmins[n] = np.array([])
            self.fZtypes[n] = np.array([])

        # TODO: do we need to do this for all modes? doing det only breaks other
        # schedulers, but maybe there's a better approach here.
        sInds = np.arange(self.TargetList.nStars)  # Initialize some sInds array
        for mode in self.OpticalSystem.observingModes:
            # This instantiates fZMap arrays for every starlight suppresion system
            # that is actually used in a mode
            modeHashName = (
                f"{self.cachefname[0:-1]}_{self.TargetList.nStars}_"
                '{mode["syst"]["name"]}.'
            )

            if (np.size(self.fZmins[mode["syst"]["name"]]) == 0) or (
                np.size(self.fZtypes[mode["syst"]["name"]]) == 0
            ):
                self.ZodiacalLight.generate_fZ(
                    self.Observatory,
                    self.TargetList,
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
                    self.TargetList,
                    self.TimeKeeping,
                    mode,
                    modeHashName,
                    self.koMaps[mode["syst"]["name"]],
                    self.koTimes,
                )

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

    def filter_targets(self, obsStart):
        """Return indices of targets available at observation start"""
        pass

    def wait(self):
        """Wait until a target becomes available for observation"""
        pass

    def general_astrophysics(self):
        """Allocate time devoted to general astrophysics"""
        pass

    def choose_next_target(self, sInds):
        """ Return index of next target and type of observation.
        Observation types are: "det", "char", or "ga" (general astrophysics)
        "ga" can only be returned when TK.using_observing_blocks is False
        If using observing blocks and no target is available, return None for both
        """

    def observation_detection(self, sInd):
        """Simulate a detection observation

        """

    def observation_characterization(self, sInd):
        """Simulate a characterization observation

        """


    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s: survey beginning." % (TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        # this should work without picking a mode:
        while not TK.mission_is_over(OS, Obs, det_mode):
            old_sInd = sInd  # used to save sInd if returned sInd is None

            sInds = self.filter_targets()

            # if no targets are currently available, need to either wait until one
            # becomes available (if using observing blocks) or do general_astrophysics
            # (if not using blocks)
            if (sInds is None) or (len(sInds) == 0):
                if TK.using_observing_blocks:  # need TK to declare if using blocks
                    self.wait()
                else:
                    self.general_astrophysics()

                # regardless of how the time was used, you can now move to next
                # iteration of this loop
                continue

            # select the next target
            sInd, obstype = self.choose_next_target(sInds)

            # handle case where both outputs are None (not using observing blocks and no
            # target available)
            if (sInd is None) and (obstype is None):
                self.wait()
            # general astrophysics scheduled
            elif obstype == 'ga':
                self.general_astrophysics()
            elif obstype == 'det':
                self.observation_detection(sInd)
                # TBD logic here to proceed to self.observation_characterization
                # new characterization_conditions_met method called here
            elif obstype == 'char':
                self.observation_characterization(sInd)

