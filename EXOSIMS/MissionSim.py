from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.waypoint import waypoint
from EXOSIMS.util.CheckScript import CheckScript
from EXOSIMS.util.keyword_fun import get_all_mod_kws, check_opticalsystem_kws
import logging
import json
import os.path
import tempfile
import numpy as np
import astropy.units as u
import copy
import inspect
import warnings
from typing import Dict, Optional, Any


class MissionSim(object):
    """Mission Simulation (backbone) class

    This class is responsible for instantiating all objects required
    to carry out a mission simulation.

    Args:
        scriptfile (string):
            Full path to JSON script file.  If not set, assumes that dictionary has been
            passed through specs.
        nopar (bool):
            Ignore any provided ensemble module in the script or specs and force the
            prototype :py:class:`~EXOSIMS.Prototypes.SurveyEnsemble`. Defaults True
        verbose (bool):
            Input to :py:meth:`~EXOSIMS.util.vprint.vprint`, toggling verbosity of
            print statements. Defaults True.
        logfile (str of None):
            Path to the log file. If None, logging is turned off.
            If supplied but empty string (''), a temporary file is generated.
        loglevel (str):
            The level of log, defaults to 'INFO'. Valid levels are: CRITICAL,
            ERROR, WARNING, INFO, DEBUG (case sensitive).
        checkInputs (bool):
            Validate inputs against selected modules. Defaults True.
        **specs (dict):
            :ref:`sec:inputspec`

    Attributes:
        StarCatalog (StarCatalog module):
            StarCatalog class object (only retained if keepStarCatalog is True)
        PlanetPopulation (PlanetPopulation module):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            Background Source class object
        PostProcessing (PostProcessing module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        TargetList (TargetList module):
            TargetList class object
        SimulatedUniverse (SimulatedUniverse module):
            SimulatedUniverse class object
        Observatory (Observatory module):
            Observatory class object
        TimeKeeping (TimeKeeping module):
            TimeKeeping class object
        SurveySimulation (SurveySimulation module):
            SurveySimulation class object
        SurveyEnsemble (SurveyEnsemble module):
            SurveyEnsemble class object
        modules (dict):
            Dictionary of all modules, except StarCatalog
        verbose (bool):
            Boolean used to create the vprint function, equivalent to the
            python print function with an extra verbose toggle parameter
            (True by default). The vprint function can be accessed by all
            modules from EXOSIMS.util.vprint.
        seed (int):
            Number used to seed the NumPy generator. Generated randomly
            by default.
        logfile (str):
            Path to the log file. If None, logging is turned off.
            If supplied but empty string (''), a temporary file is generated.
        loglevel (str):
            The level of log, defaults to 'INFO'. Valid levels are: CRITICAL,
            ERROR, WARNING, INFO, DEBUG (case sensitive).

    """

    _modtype = "MissionSim"
    _outspec = {}

    def __init__(
        self,
        scriptfile=None,
        nopar=False,
        verbose=True,
        logfile=None,
        loglevel="INFO",
        checkInputs=True,
        **specs,
    ):
        """Initializes all modules from a given script file or specs dictionary."""

        # extend given specs with (JSON) script file
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file." % scriptfile
            try:
                with open(scriptfile, "r") as ff:
                    script = ff.read()
                specs_from_file = json.loads(script)
                specs_from_file.update(specs)
            except ValueError as err:
                print(
                    "Error: %s: Input file `%s' improperly formatted."
                    % (self._modtype, scriptfile)
                )
                print("Error: JSON error was: %s" % err)
                # re-raise here to suppress the rest of the backtrace.
                # it is only confusing details about the bowels of json.loads()
                raise ValueError(err)
        else:
            specs_from_file = {}
        specs.update(specs_from_file)

        if "modules" not in specs:
            raise ValueError("No modules field found in script.")

        # push all inputs into combined spec dict and save a copy before it gets
        # modified through module instantiations
        specs["verbose"] = bool(verbose)
        specs["logfile"] = logfile
        specs["loglevel"] = loglevel
        specs["nopar"] = bool(nopar)
        specs["checkInputs"] = bool(checkInputs)
        specs0 = copy.deepcopy(specs)

        # load the vprint function (same line in all prototype module constructors)
        self.verbose = specs["verbose"]
        self.vprint = vprint(self.verbose)

        # overwrite any ensemble setting if nopar is set
        self.nopar = specs["nopar"]
        if self.nopar:
            self.vprint("No-parallel: resetting SurveyEnsemble to Prototype")
            specs["modules"]["SurveyEnsemble"] = " "

        # start logging, with log file and logging level (default: INFO)
        self.logfile = specs.get("logfile", None)
        self.loglevel = specs.get("loglevel", "INFO").upper()
        specs["logger"] = self.get_logger(self.logfile, self.loglevel)
        specs["logger"].info(
            "Start Logging: loglevel = %s" % specs["logger"].level
            + " (%s)" % self.loglevel
        )

        # populate outspec
        self.checkInputs = specs["checkInputs"]
        for att in self.__dict__:
            if att not in ["vprint"]:
                self._outspec[att] = self.__dict__[att]

        # create a surveysimulation object (triggering init of everything else)
        self.SurveySimulation = get_module(
            specs["modules"]["SurveySimulation"], "SurveySimulation"
        )(**specs)

        # collect sub-initializations
        SS = self.SurveySimulation
        self.StarCatalog = SS.StarCatalog
        self.PlanetPopulation = SS.PlanetPopulation
        self.PlanetPhysicalModel = SS.PlanetPhysicalModel
        self.OpticalSystem = SS.OpticalSystem
        self.ZodiacalLight = SS.ZodiacalLight
        self.BackgroundSources = SS.BackgroundSources
        self.PostProcessing = SS.PostProcessing
        self.Completeness = SS.Completeness
        self.TargetList = SS.TargetList
        self.SimulatedUniverse = SS.SimulatedUniverse
        self.Observatory = SS.Observatory
        self.TimeKeeping = SS.TimeKeeping

        # now that everything has successfully built, you can create the ensemble
        self.SurveyEnsemble = get_module(
            specs["modules"]["SurveyEnsemble"], "SurveyEnsemble"
        )(**copy.deepcopy(specs0))

        # create a dictionary of all modules, except StarCatalog
        self.modules = SS.modules
        self.modules["SurveyEnsemble"] = self.SurveyEnsemble

        # alias SurveySimulation random seed to attribute for easier access
        self.seed = self.SurveySimulation.seed
        self.specs0 = specs0

        # run keywords check if requested
        if self.checkInputs:
            self.check_ioscripts()

    def check_ioscripts(self) -> None:
        """Collect all input and output scripts against selected module inits and
        report and discrepancies.

        """

        # get a list of all modules in use
        mods = {}
        for modname in self.modules:
            mods[modname] = self.modules[modname].__class__
        mods["MissionSim"] = self.__class__
        if self.TargetList.keepStarCatalog:
            mods["StarCatalog"] = self.TargetList.StarCatalog.__class__
        else:
            mods["StarCatalog"] = self.TargetList.StarCatalog

        # collect keywords
        allkws, allkwmods, ukws, ukwcounts = get_all_mod_kws(mods)

        self.vprint(
            (
                "\nThe following keywords are used in multiple inits (this is ok):"
                "\n\t{}"
            ).format("\n\t".join(ukws[ukwcounts > 1]))
        )

        # now let's compare against specs0
        unused = list(set(self.specs0.keys()) - set(ukws))
        if "modules" in unused:
            unused.remove("modules")
        if "seed" in unused:
            unused.remove("seed")
        if len(unused) > 0:
            warnstr = (
                "\nThe following input keywords were not used in any "
                "module init:\n\t{}".format("\n\t".join(unused))
            )
            warnings.warn(warnstr)
        self.vprint(
            "\n{} keywords were set to their default values.".format(
                len(list(set(ukws) - set(self.specs0.keys())))
            )
        )

        # check the optical system
        out = check_opticalsystem_kws(self.specs0, self.OpticalSystem)
        if out != "":
            warnings.warn(f"\n{out}")

        # and finally, let's look at the outspec
        outspec = self.genOutSpec(modnames=True)
        # these are extraneous things allowed to be in outspec:
        whitelist = ["modules", "Revision", "seed", "nStars"]
        for w in whitelist:
            _ = outspec.pop(w, None)

        extraouts = list(set(outspec.keys()) - set(ukws))
        if len(extraouts) > 0:
            warnstr = (
                "\nThe following outspec keywords were not used in any "
                "module init:\n"
            )
            for e in extraouts:
                warnstr += "\t{:>20} ({})\n".format(e, outspec[e])
            warnings.warn(warnstr)

        missingouts = list(set(ukws) - set(outspec.keys()))
        if len(missingouts) > 0:
            allkws = np.array(allkws)
            allkwmods = np.array(allkwmods)
            warnstr = "\nThe following init keywords were not found in any outspec:\n"
            for m in missingouts:
                warnstr += "\t{:>20} ({})\n".format(
                    m, ", ".join(allkwmods[allkws == m])
                )
            warnings.warn(warnstr)

    def get_logger(self, logfile, loglevel):
        r"""Set up logging object so other modules can use logging.info(),
        logging.warning, etc.

        Args:
            logfile (string):
                Path to the log file. If None, logging is turned off.
                If supplied but empty string (''), a temporary file is generated.
            loglevel (string):
                The level of log, defaults to 'INFO'. Valid levels are: CRITICAL,
                ERROR, WARNING, INFO, DEBUG (case sensitive).

        Returns:
            logger (logging object):
                Mission Simulation logger.

        """

        # this leaves the default logger in place, so logger.warn will appear on stderr
        if logfile is None:
            logger = logging.getLogger(__name__)
            return logger

        # if empty string, a temporary file is generated
        if logfile == "":
            (dummy, logfile) = tempfile.mkstemp(
                prefix="EXOSIMS.", suffix=".log", dir="/tmp", text=True
            )
        else:
            # ensure we can write it
            try:
                with open(logfile, "w") as ff:  # noqa: F841
                    pass
            except (IOError, OSError):
                print('%s: Failed to open logfile "%s"' % (__file__, logfile))
                return None
        self.vprint("Logging to '%s' at level '%s'" % (logfile, loglevel.upper()))

        # convert string to a logging.* level
        numeric_level = getattr(logging, loglevel.upper())
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % loglevel.upper())

        # set up the top-level logger
        logger = logging.getLogger(__name__.split(".")[0])
        logger.setLevel(numeric_level)
        # do not propagate EXOSIMS messages to higher loggers in this case
        logger.propagate = False
        # create a handler that outputs to the named file
        handler = logging.FileHandler(logfile, mode="w")
        handler.setLevel(numeric_level)
        # logging format
        formatter = logging.Formatter(
            "%(levelname)s: %(filename)s(%(lineno)s): " + "%(funcName)s: %(message)s"
        )
        handler.setFormatter(formatter)
        # add the handler to the logger
        logger.addHandler(handler)

        return logger

    def run_sim(self):
        """Convenience method that simply calls the SurveySimulation run_sim method."""

        res = self.SurveySimulation.run_sim()

        return res

    def reset_sim(self, genNewPlanets=True, rewindPlanets=True, seed=None):
        """
        Convenience method that simply calls the SurveySimulation reset_sim method.
        """

        res = self.SurveySimulation.reset_sim(
            genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets, seed=seed
        )
        self.modules = self.SurveySimulation.modules
        self.modules["SurveyEnsemble"] = self.SurveyEnsemble  # replace SurveyEnsemble

        return res

    def run_ensemble(
        self,
        nb_run_sim,
        run_one=None,
        genNewPlanets=True,
        rewindPlanets=True,
        kwargs={},
    ):
        """
        Convenience method that simply calls the SurveyEnsemble run_ensemble method.
        """

        res = self.SurveyEnsemble.run_ensemble(
            self,
            nb_run_sim,
            run_one=run_one,
            genNewPlanets=genNewPlanets,
            rewindPlanets=rewindPlanets,
            kwargs=kwargs,
        )

        return res

    def genOutSpec(
        self,
        tofile: Optional[str] = None,
        modnames: bool = False,
    ) -> Dict[str, Any]:
        """Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.

        Args:
            tofile (str):
                Name of the file containing all output specifications (outspecs).
                Defaults to None.
            modnames (bool):
                If True, populate outspec dictionary with the module it originated from,
                instead of the actual value of the keyword.  Defaults False.

        Returns:
            dict:
                Dictionary containing the full :ref:`sec:inputspec`, including all
                filled-in default values. Combination of all individual module _outspec
                attributes.
        """

        starting_outspec = copy.copy(self._outspec)
        if modnames:
            for k in starting_outspec:
                starting_outspec[k] = "MissionSim"

        out = self.SurveySimulation.genOutSpec(
            starting_outspec=starting_outspec, tofile=tofile, modnames=modnames
        )

        return out

    def genWaypoint(self, targetlist=None, duration=365, tofile=None, charmode=False):
        """generates a ballpark estimate of the expected number of star visits and
        the total completeness of these visits for a given mission duration

        Args:
            targetlist (list, optional):
                List of target indices
            duration (int):
                The length of time allowed for the waypoint calculation, defaults to 365
            tofile (str):
                Name of the file containing a plot of total completeness over mission
                time, by default genWaypoint does not create this plot
            charmode (bool):
                Run the waypoint calculation using either the char mode instead of the
                det mode

        Returns:
            dict:
                Output dictionary containing the number of stars visited, the total
                completeness achieved, and the amount of time spent integrating.

        """

        SS = self.SurveySimulation
        OS = SS.OpticalSystem
        ZL = SS.ZodiacalLight
        Comp = SS.Completeness
        TL = SS.TargetList
        Obs = SS.Observatory
        TK = SS.TimeKeeping

        # Only considering detections
        allModes = OS.observingModes
        if charmode:
            int_mode = list(
                filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
            )[0]
        else:
            int_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        mpath = os.path.split(inspect.getfile(self.__class__))[0]

        if targetlist is not None:
            num_stars = len(targetlist)
            sInds = np.array(targetlist)
        else:
            num_stars = TL.nStars
            sInds = np.arange(TL.nStars)

        startTimes = TK.currentTimeAbs + np.zeros(num_stars) * u.d
        fZ = ZL.fZ(Obs, TL, sInds, startTimes, int_mode)
        fEZ = np.ones(sInds.shape) * ZL.fEZ0
        dMag = TL.int_dMag[sInds]
        WA = TL.int_WA[sInds]

        # sort star indices by completeness diveded by integration time
        intTimes = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, int_mode)
        comps = Comp.comp_per_intTime(intTimes, TL, sInds, fZ, fEZ, WA[0], int_mode)
        wp = waypoint(comps, intTimes, duration, mpath, tofile)

        return wp

    def checkScript(self, scriptfile, prettyprint=False, tofile=None):
        """Calls CheckScript and checks the script file against the mission outspec.

        Args:
            scriptfile (str):
                The path to the scriptfile being used by the sim
            prettyprint (bool):
                Outputs the results of Checkscript in a readable format.
            tofile (str):
                Name of the file containing all output specifications (outspecs).
                Default to None.

        Returns:
            str:
                Output string containing the results of the check.

        """
        if scriptfile is not None:
            cs = CheckScript(scriptfile, self.genOutSpec())
            out = cs.recurse(cs.specs_from_file, cs.outspec, pretty_print=prettyprint)
            if tofile is not None:
                mpath = os.path.split(inspect.getfile(self.__class__))[0]
                cs.write_file(os.path.join(mpath, tofile))
        else:
            out = None

        return out

    def DRM2array(self, key, DRM=None):
        """Creates an array corresponding to one element of the DRM dictionary.

        Args:
            key (str):
                Name of an element of the DRM dictionary
            DRM (list(dict)):
                Design Reference Mission, contains the results of a survey simulation

        Returns:
            ~numpy.ndarray or ~astropy.units.Quantity(~numpy.ndarray):
                Array containing all the DRM values of the selected element

        """

        # if the DRM was not specified, get it from the current SurveySimulation
        if DRM is None:
            DRM = self.SurveySimulation.DRM
        assert DRM != [], "DRM is empty. Use MissionSim.run_sim() to start simulation."

        # lists of relevant DRM elements
        keysStar = [
            "star_ind",
            "star_name",
            "arrival_time",
            "OB_nb",
            "det_time",
            "det_fZ",
            "char_time",
            "char_fZ",
        ]
        keysPlans = ["plan_inds", "det_status", "det_SNR", "char_status", "char_SNR"]
        keysParams = [
            "det_fEZ",
            "det_dMag",
            "det_WA",
            "det_d",
            "char_fEZ",
            "char_dMag",
            "char_WA",
            "char_d",
        ]
        keysFA = [
            "FA_det_status",
            "FA_char_status",
            "FA_char_SNR",
            "FA_char_fEZ",
            "FA_char_dMag",
            "FA_char_WA",
        ]
        keysOcculter = [
            "slew_time",
            "slew_dV",
            "det_dF_lateral",
            "scMass",
            "slewMass",
            "skMass",
            "char_dF_axial",
            "det_mass_used",
            "slew_mass_used",
            "det_dF_axial",
            "det_dV",
            "slew_angle",
            "char_dF_lateral",
        ]

        assert key in (
            keysStar + keysPlans + keysParams + keysFA + keysOcculter
        ), "'%s' is not a relevant DRM keyword."

        # extract arrays for each relevant keyword in the DRM
        if key in keysParams:
            if "det_" in key:
                elem = [DRM[x]["det_params"][key[4:]] for x in range(len(DRM))]

            elif "char_" in key:
                elem = [DRM[x]["char_params"][key[5:]] for x in range(len(DRM))]

        elif isinstance(DRM[0][key], u.Quantity):
            elem = ([DRM[x][key].value for x in range(len(DRM))]) * DRM[0][key].unit

        else:
            elem = [DRM[x][key] for x in range(len(DRM))]

        try:
            elem = np.array(elem)
        except ValueError:
            elem = np.array(elem, dtype=object)

        return elem

    def filter_status(self, key, status, DRM=None, obsMode=None):
        """Finds the values of one DRM element, corresponding to a status value,
        for detection or characterization.

        Args:
            key (string):
                Name of an element of the DRM dictionary
            status (integer):
                Status value for detection or characterization
            DRM (list of dicts):
                Design Reference Mission, contains the results of a survey simulation
            obsMode (string):
                Observing mode type ('det' or 'char')

        Returns:
            elemStat (ndarray / astropy Quantity array):
                Array containing all the DRM values of the selected element,
                and filtered by the value of the corresponding status array

        """

        # get DRM detection status array
        det = (
            self.DRM2array("FA_det_status", DRM=DRM)
            if "FA_" in key
            else self.DRM2array("det_status", DRM=DRM)
        )
        # get DRM characterization status array
        char = (
            self.DRM2array("FA_char_status", DRM=DRM)
            if "FA_" in key
            else self.DRM2array("char_status", DRM=DRM)
        )
        # get DRM key element array
        elem = self.DRM2array(key, DRM=DRM)

        # reshape elem array, for keys with 1 value per observation
        if elem[0].shape == () and "FA_" not in key:
            if isinstance(elem[0], u.Quantity):
                elem = np.array(
                    [
                        np.array([elem[x].value] * len(det[x])) * elem[0].unit
                        for x in range(len(elem))
                    ]
                )
            else:
                elem = np.array(
                    [np.array([elem[x]] * len(det[x])) for x in range(len(elem))]
                )

        # assign a default observing mode type ('det' or 'char')
        if obsMode is None:
            obsMode = "char" if "char_" in key else "det"
        assert obsMode in (
            "det",
            "char",
        ), "Observing mode type must be 'det' or 'char'."

        # now, find the values of elem corresponding to the specified status value
        if obsMode == "det":
            if isinstance(elem[0], u.Quantity):
                elemStat = (
                    np.concatenate(
                        [elem[x][det[x] == status].value for x in range(len(elem))]
                    )
                    * elem[0].unit
                )
            else:
                elemStat = np.concatenate(
                    [elem[x][det[x] == status] for x in range(len(elem))]
                )
        else:  # if obsMode is 'char'
            if isinstance(elem[0], u.Quantity):
                elemDet = (
                    np.concatenate(
                        [elem[x][det[x] == 1].value for x in range(len(elem))]
                    )
                    * elem[0].unit
                )
            else:
                elemDet = np.concatenate(
                    [elem[x][det[x] == 1] for x in range(len(elem))]
                )
            charDet = np.concatenate([char[x][det[x] == 1] for x in range(len(elem))])
            elemStat = elemDet[charDet == status]

        return elemStat
