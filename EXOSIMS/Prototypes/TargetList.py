from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.io
import re
import scipy.interpolate
import os.path
import inspect
import json
import hashlib
from pathlib import Path
from scipy.optimize import root_scalar
from tqdm import tqdm
import pickle
import urllib
import pkg_resources


class TargetList(object):
    r""":ref:`TargetList` Prototype

    Instantiation of an object of this class requires the instantiation of the
    following class objects:

    * StarCatalog
    * OpticalSystem
    * ZodiacalLight
    * Completeness
    * PostProcessing

    Args:
        missionStart (float):
            Mission start time (MJD)
        staticStars (bool):
            Do not apply proper motions to stars during simulation. Defaults True.
        keepStarCatalog (bool):
            Retain the StarCatalog object as an attribute after the target
            list is populated (defaults False)
        fillPhotometry (bool):
            Attempt to fill in missing photometric data for targets from
            available values (primarily spectral type and luminosity). Defaults False.
        explainFiltering (bool):
            Print informational messages at each target list filtering step.
            Defaults False.
        filterBinaries (bool):
            Remove all binary stars or stars with known close companions from target
            list. Defaults True.
        filterSubM (bool):
            Remove stars of spectral type L, T, Y.  Defaults False.
        cachedir (str or None):
            Full path to cache directory.  If None, use default.
        filter_for_char (bool):
            Use spectroscopy observation mode (rather than the default detection mode)
            for all calculations. Defaults False.
        earths_only (bool):
            Used in upstream modules.  Alias for filter_for_char. Defaults False.
        getKnownPlanets (bool):
            Retrieve information about which stars have known planets. Defaults False.
        int_WA (float or numpy.ndarray or None):
            Working angle (arcsec) at which to caluclate integration times for default
            observations.  If input is scalar, all targets will get the same value.  If
            input is an array, it must match the size of the input catalog.  If None,
            a default value of halway between the inner and outer working angle of the
            default observing mode will be used.  If the OWA is infinite, twice the IWA
            is used.
        int_dMag (float or numpy.ndarray):
            :math:`\Delta\textrm{mag}` to assume when calculating integration time for
            default observations. If input is scalar, all targets will get the same
            value.  If input is an array, it must match the size of the input catalog.
            Defaults to 25
        scaleWAdMag (bool):
            If True, rescale int_dMag and int_WA for all stars based on luminosity and
            to ensure that WA is within the IWA/OWA. Defaults False.
        int_dMag_offset (float):
            Offset applied to int_dMag when scaleWAdMag is True.
        **specs:
            Keyword inputs passed to all sub-object instantiations.

    Attributes:
        (StarCatalog values)
            Mission specific filtered star catalog values from StarCatalog class object
            Typically these include:
            Name, Spec, Umag, Bmag, Vmag, Rmag, Imag, Jmag, Hmag, Kmag, BV, MV,
            BC, L, Binary_Cut, dist, parx, coords, pmra, pmdec, rv
        catalog_atts (list):
            Names of all catalog attributes
        StarCatalog (StarCatalog module):
            StarCatalog class object (only retained if keepStarCatalog is True).
            If keepStarCatalog is False, retain the class type only.
        PlanetPopulation (PlanetPopulation module)
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        PostProcessing (PostProcessing module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        base_filename (str):
            Base of filename to use for all cached data products
        cachedir (str):
            Full path to cache directory
        int_comp (numpy.ndarray):
            Completeness value for each target star for default observation WA and
            :math:`\Delta{\\textrm{mag}}`.
        int_dMag (numpy.ndarray):
             :math:`\Delta{\\textrm{mag}}` used for default observation integration time
             calculation
        int_dMag_offset (float):
            Offset applied to int_dMag when scaleWAdMag is True.
        earths_only (bool):
            Used in upstream modules.  Alias for filter_for_char.
        explainFiltering (bool):
            Print informational messages at each target list filtering step.
        extstr (str):
            String whose hash generates the base of cache products filename.
        F0dict (dict):
            Internal storage of pre-computed zero-mag flux values that is populated
            each time an F0 is requested for a particular target.
        filterBinaries (bool):
            Remove all binary stars or stars with known close companions from target
            list.
        filter_for_char (bool):
            Use spectroscopy observation mode (rather than the default detection mode)
            for all calculations.
        fillPhotometry (bool):
            Attempt to fill in missing target photometric  values using interpolants of
            tabulated values for the stellar type.
        filterSubM (bool):
            Remove all sub-M spectral types (L,T,Y).  Note that fillPhotometry will
            typically fail for any stars of this type, so this should be set to True
            when fillPhotometry is True.
        getKnownPlanets (bool):
            a boolean indicating whether to grab the list of known planets from IPAC
            and read the alias pkl file
        I (astropy.units.quantity.Quantity):
            array of star system inclinations (angle)
        intCutoff_comp (numpy.ndarray):
            Completeness values of all targets corresponding to the cutoff integration
            time set in the optical system.
        intCutoff_dMag (numpy.ndarray):
            :math:`\Delta{\\textrm{mag}}` of all targets corresponding to the cutoff
            integration time set in the optical system.
        keepStarCatalog (bool):
            Do not delete StarCatalog after TargetList is built
        MsEst (numpy.ndarray):
            'approximate' stellar mass in units of solar mass
        MsTrue (numpy.ndarray):
            'true' stellar mass in units of solar mass
        nStars (int):
            Number of target stars.  Length of all arrays listed in catalog_atts
        saturation_comp (numpy.ndarray):
            Maximum possible completeness values of all targets.
        saturation_dMag (numpy.ndarray):
            :math:`\Delta{\\textrm{mag}}` at which completness stops increasing for all
            targets.
        scaleWAdMag (bool):
            Rescale int_dMag and int_WA for all stars based on luminosity and to ensure
            that WA is within the IWA/OWA.
        specdatapath (str):
            Full path to spectral data folder
        specdict (dict):
            Dictionary of spectral types
        staticStars (bool):
            Do not apply proper motions to stars.  Stars always at mission start time
            positions.
        int_WA (astropy.units.quantity.Quantity):
            Working angle used for integration time calculation (angle)
        _outspec (dict):
            Output specification.  All input items with their input or default values.

    """

    _modtype = "TargetList"

    def __init__(
        self,
        missionStart=60634,
        staticStars=True,
        keepStarCatalog=False,
        fillPhotometry=False,
        explainFiltering=False,
        filterBinaries=True,
        filterSubM=False,
        cachedir=None,
        filter_for_char=False,
        earths_only=False,
        getKnownPlanets=False,
        int_WA=None,
        int_dMag=25,
        scaleWAdMag=False,
        int_dMag_offset=1,
        **specs,
    ):

        # start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # validate TargetList inputs
        assert isinstance(staticStars, bool), "staticStars must be a boolean."
        assert isinstance(keepStarCatalog, bool), "keepStarCatalog must be a boolean."
        assert isinstance(fillPhotometry, bool), "fillPhotometry must be a boolean."
        assert isinstance(explainFiltering, bool), "explainFiltering must be a boolean."
        assert isinstance(filterBinaries, bool), "filterBinaries must be a boolean."
        assert isinstance(filterSubM, bool), "filterSubM must be a boolean."
        self.staticStars = bool(staticStars)
        self.keepStarCatalog = bool(keepStarCatalog)
        self.fillPhotometry = bool(fillPhotometry)
        self.explainFiltering = bool(explainFiltering)
        self.filterBinaries = bool(filterBinaries)
        self.filterSubM = bool(filterSubM)
        self.filter_for_char = bool(filter_for_char)
        self.earths_only = bool(earths_only)

        # check if KnownRVPlanetsTargetList is using KnownRVPlanets
        if specs["modules"]["TargetList"] == "KnownRVPlanetsTargetList":
            assert (
                specs["modules"]["PlanetPopulation"] == "KnownRVPlanets"
            ), "KnownRVPlanetsTargetList must use KnownRVPlanets"
        else:
            assert (
                specs["modules"]["PlanetPopulation"] != "KnownRVPlanets"
            ), "This TargetList cannot use KnownRVPlanets"

        # check if KnownRVPlanetsTargetList is using KnownRVPlanets
        if specs["modules"]["TargetList"] == "KnownRVPlanetsTargetList":
            assert (
                specs["modules"]["PlanetPopulation"] == "KnownRVPlanets"
            ), "KnownRVPlanetsTargetList must use KnownRVPlanets"
        else:
            assert (
                specs["modules"]["PlanetPopulation"] != "KnownRVPlanets"
            ), "This TargetList cannot use KnownRVPlanets"

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec"]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat
        # set up stuff for spectral type conversion
        # Paths
        indexf = pkg_resources.resource_filename(
            "EXOSIMS.TargetList", "pickles_index.pkl"
        )
        assert os.path.exists(
            indexf
        ), "Pickles catalog index file not found in TargetList directory."

        datapath = pkg_resources.resource_filename("EXOSIMS.TargetList", "dat_uvk")
        assert os.path.isdir(
            datapath
        ), "Could not locate %s in TargetList directory." % (datapath)

        # grab Pickles Atlas index
        with open(indexf, "rb") as handle:
            self.specindex = pickle.load(handle)

        self.speclist = sorted(self.specindex.keys())
        self.specdatapath = datapath

        # spectral type decomposition
        # default string: Letter|number|roman numeral
        # number is either x, x.x, x/x
        # roman numeral is either
        # either number of numeral can be wrapped in ()
        self.specregex1 = re.compile(
            r"([OBAFGKMLTY])\s*\(*(\d*\.\d+|\d+|\d+\/\d+)\)*\s*\(*([IV]+\/{0,1}[IV]*)"
        )
        # next option is that you have something like 'G8/K0IV'
        self.specregex2 = re.compile(
            r"([OBAFGKMLTY])\s*(\d+)\/[OBAFGKMLTY]\s*\d+\s*\(*([IV]+\/{0,1}[IV]*)"
        )
        # next down the list, just try to match leading vals and assume it's a dwarf
        self.specregex3 = re.compile(r"([OBAFGKMLTY])\s*(\d*\.\d+|\d+|\d+\/\d+)")
        # last resort is just match spec type
        self.specregex4 = re.compile(r"([OBAFGKMLTY])")

        self.romandict = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
        self.specdict = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

        # everything in speclist is correct, so only need first regexp
        specliste = []
        for spec in self.speclist:
            specliste.append(self.specregex1.match(spec).groups())
        self.specliste = np.vstack(specliste)
        self.spectypenum = np.array(
            [self.specdict[ll] for ll in self.specliste[:, 0]]
        ) * 10 + np.array(self.specliste[:, 1]).astype(float)

        # Create F0 dictionary for storing mode-associated F0s
        self.F0dict = {}
        # get desired module names (specific or prototype) and instantiate objects
        self.StarCatalog = get_module(specs["modules"]["StarCatalog"], "StarCatalog")(
            **specs
        )
        self.OpticalSystem = get_module(
            specs["modules"]["OpticalSystem"], "OpticalSystem"
        )(**specs)
        self.ZodiacalLight = get_module(
            specs["modules"]["ZodiacalLight"], "ZodiacalLight"
        )(**specs)
        self.PostProcessing = get_module(
            specs["modules"]["PostProcessing"], "PostProcessing"
        )(**specs)
        self.Completeness = get_module(
            specs["modules"]["Completeness"], "Completeness"
        )(**specs)

        # bring inherited class objects to top level of Simulated Universe
        self.BackgroundSources = self.PostProcessing.BackgroundSources

        # if specs contains a completeness_spec then we are going to generate separate
        # instances of planet population and planet physical model for completeness
        # and for the rest of the sim
        if "completeness_specs" in specs:
            self.PlanetPopulation = get_module(
                specs["modules"]["PlanetPopulation"], "PlanetPopulation"
            )(**specs)
            self.PlanetPhysicalModel = self.PlanetPopulation.PlanetPhysicalModel
        else:
            self.PlanetPopulation = self.Completeness.PlanetPopulation
            self.PlanetPhysicalModel = self.Completeness.PlanetPhysicalModel

        # list of possible Star Catalog attributes
        self.catalog_atts = [
            "Name",
            "Spec",
            "parx",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "dist",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
            "closesep",
            "closedm",
            "brightsep",
            "brightdm",
        ]

        detmode = list(
            filter(
                lambda mode: mode["detectionMode"], self.OpticalSystem.observingModes
            )
        )[0]

        # Define int_dMag and int_WA
        if int_WA is None:
            int_WA = (
                2.0 * detmode["IWA"]
                if np.isinf(detmode["OWA"])
                else (detmode["IWA"] + detmode["OWA"]) / 2.0
            )
            int_WA = int_WA.to("arcsec")

        # Save the dMag and WA values used to calculate integration time
        self.int_dMag = np.array(int_dMag, dtype=float, ndmin=1)
        self.int_WA = np.array(int_WA, dtype=float, ndmin=1) * u.arcsec
        # This parameter is used to modify the dMag value used to calculate
        # integration time
        self.int_dMag_offset = int_dMag_offset
        # Flag for whether to do luminosity scaling
        self.scaleWAdMag = scaleWAdMag

        module_list = [
            "OpticalSystem",
            "StarCatalog",
            "ZodiacalLight",
            "PostProcessing",
            "BackgroundSources",
            "Completeness",
            "PlanetPopulation",
            "PlanetPhysicalModel",
            "TargetList",
        ]
        self.base_filename = "TargetList" + "".join(
            getattr(self, obj).__class__.__name__ for obj in module_list[:-1]
        )
        atts = list(self.__dict__)
        self.extstr = ""
        for att in sorted(atts, key=str.lower):
            if not callable(getattr(self, att)) and (att not in module_list):
                att_str = str(getattr(self, att))
                self.extstr += f"{att+att_str+' '}"
        self.extstr += (
            f"{len(self.StarCatalog.Name)} {self.fillPhotometry} "
            f"{self.filterBinaries} {self.filterSubM} "
            f"{self.filter_for_char} {self.earths_only} {self.int_dMag} "
            f"{self.int_WA} {self.scaleWAdMag}"
        )
        for mode in self.OpticalSystem.observingModes:
            for key, item in mode.items():
                if not callable(item) and key != "hex":
                    if type(item) is dict:
                        for subkey, subitem in item.items():
                            if not callable(subitem):
                                self.extstr += f"{subkey}-{subitem}"
                    # self.extstr += str(item)
                    else:
                        self.extstr += f"{key}-{item}"
        # self.extstr += str(detmode.items())
        ext = hashlib.md5(self.extstr.encode("utf-8")).hexdigest()
        self.base_filename += ext
        # Remove spaces from string (in the case of prototype use)
        self.base_filename.replace(" ", "")

        # now populate and filter the list
        self.populate_target_list(**specs)
        # generate any completeness update data needed
        self.Completeness.gen_update(self)
        self.filter_target_list(**specs)

        # have target list, no need for catalog now (unless asked to retain)
        # instead, just keep the class of the star catalog for bookkeeping
        if not self.keepStarCatalog:
            self.StarCatalog = self.StarCatalog.__class__

        # add nStars to outspec
        self._outspec["nStars"] = self.nStars

        # if staticStars is True, the star coordinates are taken at mission start,
        # and are not propagated during the mission
        # TODO: This is barely legible. Replace with class method.
        self.starprop_static = None
        if self.staticStars is True:
            allInds = np.arange(self.nStars, dtype=int)
            missionStart = Time(float(missionStart), format="mjd", scale="tai")
            self.starprop_static = (
                lambda sInds, currentTime, eclip=False, c1=self.starprop(
                    allInds, missionStart, eclip=False
                ), c2=self.starprop(allInds, missionStart, eclip=True): c1[sInds]
                if not (eclip)  # noqa: E275
                else c2[sInds]
            )

        # Find Known Planets
        self.getKnownPlanets = getKnownPlanets
        self._outspec["getKnownPlanets"] = self.getKnownPlanets
        if self.getKnownPlanets:
            alias = self.loadAliasFile()
            data = self.constructIPACurl()
            starsWithPlanets = self.setOfStarsWithKnownPlanets(data)
            knownPlanetBoolean = self.createKnownPlanetBoolean(  # noqa: F841
                alias, starsWithPlanets
            )

    def __str__(self):
        """String representation of the Target List object

        When the command 'print' is used on the Target List object, this method
        will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Target List class object attributes"

    def populate_target_list(self, popStars=None, **specs):
        """This function is actually responsible for populating values from the star
        catalog (or any other source) into the target list attributes.

        The prototype implementation does the following:

        Copy directly from star catalog and remove stars with any NaN attributes
        Calculate completeness and max integration time, and generates stellar masses.

        Args:
            popStars (list, optional):
                Remove given stars (by exact name matching) from target list.
                Defaults None.
            **specs:
                :ref:`sec:inputspec`

        """
        SC = self.StarCatalog
        Comp = self.Completeness

        # bring Star Catalog values to top level of Target List
        missingatts = []
        for att in self.catalog_atts:
            if not hasattr(SC, att):
                missingatts.append(att)
            else:
                if type(getattr(SC, att)) == np.ma.core.MaskedArray:
                    setattr(self, att, getattr(SC, att).filled(fill_value=float("nan")))
                else:
                    setattr(self, att, getattr(SC, att))
        for att in missingatts:
            self.catalog_atts.remove(att)

        # number of target stars
        self.nStars = len(SC.Name)
        if self.explainFiltering:
            print("%d targets imported from star catalog." % self.nStars)

        if popStars is not None:
            tmp = np.arange(self.nStars)
            for n in popStars:
                tmp = tmp[self.Name != n]

            self.revise_lists(tmp)

            if self.explainFiltering:
                print(
                    "%d targets remain after removing requested targets." % self.nStars
                )

        if self.filterSubM:
            self.subM_filter()

        if self.fillPhotometry:
            self.fillPhotometryVals()

        # filter out nan attribute values from Star Catalog
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering." % self.nStars)

        # filter out target stars with 0 luminosity
        self.zero_lum_filter()
        if self.explainFiltering:
            print("%d targets remain after removing requested targets." % self.nStars)

        self.calc_saturation_and_intCutoff_vals()

        # populate completeness values
        self.int_comp = Comp.target_completeness(self)

        # calculate 'true' and 'approximate' stellar masses
        self.vprint("Calculating target stellar masses.")
        self.stellar_mass()

        # Calculate Star System Inclinations
        self.I = self.gen_inclinations(self.PlanetPopulation.Irange)  # noqa: E741

        # include new attributes to the target list catalog attributes
        self.catalog_atts.append("int_comp")
        self.catalog_atts.append("int_dMag")
        self.catalog_atts.append("int_WA")

    def calc_saturation_and_intCutoff_vals(self):
        """
        Calculates the saturation and integration cutoff time dMag and
        completeness values, saves them as attributes, refines the dMag used to
        calculate integration times so it does not exceed the integration
        cutoff time dMag, and handles any orbit scaling necessary
        """

        if len(self.int_WA) == 1:
            self.int_WA = np.repeat(self.int_WA, self.nStars)
        if len(self.int_dMag) == 1:
            self.int_dMag = np.repeat(self.int_dMag, self.nStars)
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPop = self.PlanetPopulation
        Comp = self.Completeness
        detmode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[0]
        if self.filter_for_char or self.earths_only:
            mode = list(
                filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
            )[0]
            self.calc_char_int_comp = True
        else:
            mode = detmode
            self.calc_char_int_comp = False

        # Calculate the saturation dMag
        self.saturation_dMag = self.calc_saturation_dMag(mode)

        # Calculate the completeness value if the star is integrated for an
        # infinite time by using the saturation dMag
        if PPop.scaleOrbits:
            tmp_smin = np.tan(mode["IWA"]) * self.dist / np.sqrt(self.L)
            if np.isinf(mode["OWA"]):
                tmp_smax = np.inf * self.dist
            else:
                tmp_smax = np.tan(mode["OWA"]) * self.dist / np.sqrt(self.L)
            tmp_dMag = self.saturation_dMag - 2.5 * np.log10(self.L)
        else:
            tmp_smin = np.tan(mode["IWA"]) * self.dist
            if np.isinf(mode["OWA"]):
                tmp_smax = np.inf * self.dist
            else:
                tmp_smax = np.tan(mode["OWA"]) * self.dist
            tmp_dMag = self.saturation_dMag
        stars_hash = hashlib.md5(str(self.StarCatalog.Name).encode("utf-8")).hexdigest()
        saturation_comp_path = Path(
            self.cachedir, self.base_filename + stars_hash + ".sat_comp"
        )
        if saturation_comp_path.exists():
            self.vprint(f"Loaded saturation_comp values from {saturation_comp_path}")
            with open(saturation_comp_path, "rb") as f:
                self.saturation_comp = pickle.load(f)
        else:
            self.vprint("Calculating the saturation time completeness")
            self.saturation_comp = Comp.comp_calc(
                tmp_smin.to(u.AU).value, tmp_smax.to(u.AU).value, tmp_dMag
            )
            with open(saturation_comp_path, "wb") as f:
                pickle.dump(self.saturation_comp, f)

        # Set limiting dMag for intCutoff time
        self.intCutoff_dMag = self.calc_intCutoff_dMag(mode)

        # Calculate intCutoff completeness
        intCutoff_comp_path = Path(
            self.cachedir, self.base_filename + stars_hash + ".intCutoff_comp"
        )
        if intCutoff_comp_path.exists():
            self.vprint(f"Loaded intCutoff_comp values from {intCutoff_comp_path}")
            with open(intCutoff_comp_path, "rb") as f:
                self.intCutoff_comp = pickle.load(f)
        else:
            self.vprint("Calculating the integration cutoff time completeness")
            self.intCutoff_comp = Comp.comp_per_intTime(
                OS.intCutoff,
                self,
                np.arange(self.nStars),
                ZL.fZ0,
                ZL.fEZ0,
                self.int_WA,
                mode,
            )
            with open(intCutoff_comp_path, "wb") as f:
                pickle.dump(self.intCutoff_comp, f)

        # Refine int_dMag
        if len(self.int_dMag) == 1:
            self._outspec["int_dMag"] = self.int_dMag[0]
            self.int_dMag = np.array([self.int_dMag[0]] * self.nStars)
        else:
            assert (
                len(self.int_dMag) == self.nStars
            ), "Input int_dMag array doesn't match number of target stars."
            self._outspec["int_dMag"] = self.int_dMag

        if len(self.int_WA) == 1:
            self._outspec["int_WA"] = self.int_WA[0].to("arcsec").value
            self.int_WA = (
                np.array([self.int_WA[0].value] * self.nStars) * self.int_WA.unit
            )
        else:
            assert (
                len(self.int_WA) == self.nStars
            ), "Input int_WA array doesn't match number of target stars."
            self._outspec["int_WA"] = self.int_WA.to("arcsec").value

        if self.scaleWAdMag:
            # the goal of this is to make these values match the earthlike pdf
            # used to calculate completness, which scales with luminosity
            self.int_WA = ((np.sqrt(self.L) * u.AU / self.dist).decompose() * u.rad).to(
                u.arcsec
            )
            self.int_WA[np.where(self.int_WA > detmode["OWA"])[0]] = detmode["OWA"] * (
                1.0 - 1e-14
            )
            self.int_WA[np.where(self.int_WA < detmode["IWA"])[0]] = detmode["IWA"] * (
                1.0 + 1e-14
            )
            self.int_dMag = (
                self.intCutoff_dMag - self.int_dMag_offset + 2.5 * np.log10(self.L)
            )

        # if requested, rescale based on luminosities and mode limits
        # Commented out until a better understanding of where this came from is
        # available. Block above is a simplified version of this logic
        # if self.scaleWAdMag:
        # for i,Lstar in enumerate(self.L):
        # if (Lstar < 6.85) and (Lstar > 0.):
        # self.int_dMag[i] = self.intCutoff_dMag[i] - self.int_dMag_offset + 2.5 * np.log10(Lstar)   # noqa: E501
        # else:
        # self.int_dMag[i] = self.intCutoff_dMag[i]

        # EEID = ((np.sqrt(Lstar)*u.AU/self.dist[i]).decompose()*u.rad).to(u.arcsec)
        # if EEID < detmode['IWA']:
        # EEID = detmode['IWA']*(1.+1e-14)
        # elif EEID > detmode['OWA']:
        # EEID = detmode['OWA']*(1.-1e-14)

        # self.int_WA[i] = EEID
        # self._outspec['scaleWAdMag'] = self.scaleWAdMag

        # Go through the int_dMag values and replace with limiting dMag where
        # int_dMag is higher. Since the int_dMag will never be reached if
        # intCutoff_dMag is below it
        for i, int_dMag_val in enumerate(self.int_dMag):
            if int_dMag_val > self.intCutoff_dMag[i]:
                self.int_dMag[i] = self.intCutoff_dMag[i]

        self.catalog_atts.append("intCutoff_dMag")
        self.catalog_atts.append("intCutoff_comp")
        self.catalog_atts.append("saturation_dMag")
        self.catalog_atts.append("saturation_comp")

    def F0(self, BW, lam, spec=None):
        """
        This function calculates the spectral flux density for a given
        spectral type. Assumes the Pickles Atlas is saved to TargetList:
        ftp://ftp.stsci.edu/cdbs/grid/pickles/dat_uvk/

        If spectral type is provided, tries to match based on luminosity class,
        then spectral type. If no type, or not match, defaults to fit based on
        Traub et al. 2016 (JATIS), which gives spectral flux density of
        ~9.5e7 [ph/s/m2/nm] @ 500nm


        Args:
            BW (float):
                Bandwidth fraction
            lam (~astropy.units.Quantity):
                Central wavelength in units of nm
            Spec (str):
                Spectral type. Should be something like G0V

        Returns:
            ~astropy.units.Quantity:
                Spectral flux density in units of ph/m**2/s/nm.
        """

        if spec is not None:
            # Try to decmompose the input spectral type
            tmp = self.specregex1.match(spec)
            if not (tmp):
                tmp = self.specregex2.match(spec)
            if tmp:
                spece = [
                    tmp.groups()[0],
                    float(tmp.groups()[1].split("/")[0]),
                    tmp.groups()[2].split("/")[0],
                ]
            else:
                tmp = self.specregex3.match(spec)
                if tmp:
                    spece = [tmp.groups()[0], float(tmp.groups()[1].split("/")[0]), "V"]
                else:
                    tmp = self.specregex4.match(spec)
                    if tmp:
                        spece = [tmp.groups()[0], 0, "V"]
                    else:
                        spece = None

            # now match to the atlas
            if spece is not None:
                lumclass = self.specliste[:, 2] == spece[2]
                ind = np.argmin(
                    np.abs(
                        self.spectypenum[lumclass]
                        - (self.specdict[spece[0]] * 10 + spece[1])
                    )
                )
                specmatch = "".join(self.specliste[lumclass][ind])
            else:
                specmatch = None
        else:
            specmatch = None

        if specmatch is None:
            F0 = (
                1e4
                * 10 ** (4.01 - (lam / u.nm - 550) / 770)
                * u.ph
                / u.s
                / u.m**2
                / u.nm
            )
        else:
            # Open corresponding spectrum
            with fits.open(
                os.path.join(self.specdatapath, self.specindex[specmatch])
            ) as hdulist:
                sdat = hdulist[1].data

            # Reimann integration of spectrum within bandwidth, converted from
            # erg/s/cm**2/angstrom to ph/s/m**2/nm, where dlam in nm is the
            # variable of integration.
            lmin = lam * (1 - BW / 2)
            lmax = lam * (1 + BW / 2)

            # midpoint Reimann sum
            band = (sdat.WAVELENGTH >= lmin.to(u.Angstrom).value) & (
                sdat.WAVELENGTH <= lmax.to(u.Angstrom).value
            )
            ls = sdat.WAVELENGTH[band] * u.Angstrom
            Fs = (sdat.FLUX[band] * u.erg / u.s / u.cm**2 / u.AA) * (
                ls / const.h / const.c
            )
            F0 = (
                np.sum((Fs[1:] + Fs[:-1]) * np.diff(ls) / 2.0) / (lmax - lmin) * u.ph
            ).to(u.ph / u.s / u.m**2 / u.nm)

        return F0

    def fillPhotometryVals(self):
        """
        This routine attempts to fill in missing photometric values, including
        the luminosity, absolute magnitude, V band bolometric correction, and the
        apparent VBHJK magnitudes by interpolating values from a table of standard
        stars by spectral type.

        The data is from:
        "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester)
        Version 2017.09.06

        """

        # Looking for file EEM_dwarf_UBVIJHK_colors_Teff.txt in the TargetList folder
        filename = "EEM_dwarf_UBVIJHK_colors_Teff.txt"
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        classpath = os.path.normpath(os.path.join(classpath, "..", "TargetList"))
        datapath = os.path.join(classpath, filename)
        assert os.path.isfile(datapath), (
            "Could not locate %s in TargetList directory." % filename
        )

        data = astropy.io.ascii.read(
            datapath, fill_values=[("...", np.nan), ("....", np.nan), (".....", np.nan)]
        )

        specregex = re.compile(r"([OBAFGKMLTY])(\d*\.\d+|\d+)V")
        specregex2 = re.compile(r"([OBAFGKMLTY])(\d*\.\d+|\d+).*")

        MK = []
        MKn = []
        for s in data["SpT"].data:
            m = specregex.match(s)
            MK.append(m.groups()[0])
            MKn.append(m.groups()[1])
        MK = np.array(MK)
        MKn = np.array(MKn)

        # create dicts of interpolants
        Mvi = {}
        BmVi = {}
        logLi = {}
        BCi = {}
        VmKi = {}
        VmIi = {}
        HmKi = {}
        JmHi = {}
        VmRi = {}
        UmBi = {}

        for l in "OBAFGKM":  # noqa: E741
            Mvi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["Mv"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            BmVi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["B-V"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            logLi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["logL"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            VmKi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["V-Ks"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            VmIi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["V-Ic"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            VmRi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["V-Rc"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            HmKi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["H-K"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            JmHi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["J-H"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            BCi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["BCv"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )
            UmBi[l] = scipy.interpolate.interp1d(
                MKn[MK == l].astype(float),
                data["U-B"][MK == l].data.astype(float),
                bounds_error=False,
                fill_value="extrapolate",
            )

        # first try to fill in missing Vmags
        if np.all(self.Vmag == 0):
            self.Vmag *= np.nan
        if np.any(np.isnan(self.Vmag)):
            inds = np.where(np.isnan(self.Vmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.Vmag[i] = Mvi[m.groups()[0]](m.groups()[1])
                    self.MV[i] = self.Vmag[i] - 5 * (
                        np.log10(self.dist[i].to("pc").value) - 1
                    )

        # next, try to fill in any missing B mags
        if np.all(self.Bmag == 0):
            self.Bmag *= np.nan
        if np.any(np.isnan(self.Bmag)):
            inds = np.where(np.isnan(self.Bmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.BV[i] = BmVi[m.groups()[0]](m.groups()[1])
                    self.Bmag[i] = self.BV[i] + self.Vmag[i]

        # next fix any missing luminosities
        if np.all(self.L == 0):
            self.L *= np.nan
        if np.any(np.isnan(self.L)):
            inds = np.where(np.isnan(self.L))[0]
            for i in inds:
                m = self.specregex2.match(self.Spec[i])
                if not (m):
                    m = specregex2.match(self.Spec[i])
                if m:
                    self.L[i] = 10.0 ** logLi[m.groups()[0]](m.groups()[1])  # *u.L_sun

        # and bolometric corrections
        if np.all(self.BC == 0):
            self.BC *= np.nan
        if np.any(np.isnan(self.BC)):
            inds = np.where(np.isnan(self.BC))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.BC[i] = BCi[m.groups()[0]](m.groups()[1])

        # next fill in K mags
        if np.all(self.Kmag == 0):
            self.Kmag *= np.nan
        if np.any(np.isnan(self.Kmag)):
            inds = np.where(np.isnan(self.Kmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    VmK = VmKi[m.groups()[0]](m.groups()[1])
                    self.Kmag[i] = self.Vmag[i] - VmK

        # next fill in H mags
        if np.all(self.Hmag == 0):
            self.Hmag *= np.nan
        if np.any(np.isnan(self.Hmag)):
            inds = np.where(np.isnan(self.Hmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    HmK = HmKi[m.groups()[0]](m.groups()[1])
                    self.Hmag[i] = self.Kmag[i] + HmK

        # next fill in J mags
        if np.all(self.Jmag == 0):
            self.Jmag *= np.nan
        if np.any(np.isnan(self.Jmag)):
            inds = np.where(np.isnan(self.Jmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    JmH = JmHi[m.groups()[0]](m.groups()[1])
                    self.Jmag[i] = self.Hmag[i] + JmH

        # next fill in I mags
        if np.all(self.Imag == 0):
            self.Imag *= np.nan
        if np.any(np.isnan(self.Imag)):
            inds = np.where(np.isnan(self.Imag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    VmI = VmIi[m.groups()[0]](m.groups()[1])
                    self.Imag[i] = self.Vmag[i] - VmI

        # next fill in U mags
        if np.all(self.Umag == 0):
            self.Umag *= np.nan
        if np.any(np.isnan(self.Umag)):
            inds = np.where(np.isnan(self.Umag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    UmB = UmBi[m.groups()[0]](m.groups()[1])
                    self.Umag[i] = self.Bmag[i] + UmB

        # next fill in R mags
        if np.all(self.Rmag == 0):
            self.Rmag *= np.nan
        if np.any(np.isnan(self.Rmag)):
            inds = np.where(np.isnan(self.Rmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    VmR = VmRi[m.groups()[0]](m.groups()[1])
                    self.Rmag[i] = self.Vmag[i] - VmR

    def filter_target_list(self, **specs):
        """This function is responsible for filtering by any required metrics.

        The prototype implementation removes the following stars:
            * Stars with NAN values in their parameters
            * Binary stars
            * Systems with planets inside the OpticalSystem fundamental IWA
            * Systems where minimum integration time is longer than OpticalSystem cutoff
            * Systems not meeting the Completeness threshold

        Additional filters can be provided in specific TargetList implementations.

        """
        # filter out binary stars
        if self.filterBinaries:
            self.binary_filter()
            if self.explainFiltering:
                print("%d targets remain after binary filter." % self.nStars)

        # filter out systems with planets within the IWA
        self.outside_IWA_filter()
        if self.explainFiltering:
            print("%d targets remain after IWA filter." % self.nStars)

        # filter out systems which do not reach the completeness threshold
        self.completeness_filter()
        if self.explainFiltering:
            print("%d targets remain after completeness filter." % self.nStars)

    def nan_filter(self):
        """Populates Target List and filters out values which are nan"""

        # filter out nan values in numerical attributes
        for att in self.catalog_atts:
            if ("close" in att) or ("bright" in att):
                continue
            if getattr(self, att).shape[0] == 0:
                pass
            elif (type(getattr(self, att)[0]) == str) or (
                type(getattr(self, att)[0]) == bytes
            ):
                # FIXME: intent here unclear:
                # note float('nan') is an IEEE NaN, getattr(.) is a str,
                # and != on NaNs is special
                i = np.where(getattr(self, att) != float("nan"))[0]
                self.revise_lists(i)
            # exclude non-numerical types
            elif type(getattr(self, att)[0]) not in (
                np.unicode_,
                np.string_,
                np.bool_,
                bytes,
            ):
                if att == "coords":
                    i1 = np.where(~np.isnan(self.coords.ra.to("deg").value))[0]
                    i2 = np.where(~np.isnan(self.coords.dec.to("deg").value))[0]
                    i = np.intersect1d(i1, i2)
                else:
                    i = np.where(~np.isnan(getattr(self, att)))[0]
                self.revise_lists(i)

    def binary_filter(self):
        """Removes stars which have attribute Binary_Cut == True"""

        i = np.where(~self.Binary_Cut)[0]
        self.revise_lists(i)

    def life_expectancy_filter(self):
        """Removes stars from Target List which have BV < 0.3"""

        i = np.where(self.BV > 0.3)[0]
        self.revise_lists(i)

    def subM_filter(self):
        """
        Filter out any targets of spectral type L, T, Y
        """
        specregex = re.compile(r"([OBAFGKMLTY])*")
        spect = np.full(self.Spec.size, "")
        for j, s in enumerate(self.Spec):
            m = specregex.match(s)
            if m:
                spect[j] = m.groups()[0]

        i = np.where((spect != "L") & (spect != "T") & (spect != "Y"))[0]
        self.revise_lists(i)

    def main_sequence_filter(self):
        """Removes stars from Target List which are not main sequence"""

        # indices from Target List to keep
        i1 = np.where((self.BV < 0.74) & (self.MV < 6 * self.BV + 1.8))[0]
        i2 = np.where(
            (self.BV >= 0.74) & (self.BV < 1.37) & (self.MV < 4.3 * self.BV + 3.05)
        )[0]
        i3 = np.where((self.BV >= 1.37) & (self.MV < 18 * self.BV - 15.7))[0]
        i4 = np.where((self.BV < 0.87) & (self.MV > -8 * (self.BV - 1.35) ** 2 + 7.01))[
            0
        ]
        i5 = np.where(
            (self.BV >= 0.87) & (self.BV < 1.45) & (self.MV < 5 * self.BV + 0.81)
        )[0]
        i6 = np.where((self.BV >= 1.45) & (self.MV > 18 * self.BV - 18.04))[0]
        ia = np.append(np.append(i1, i2), i3)
        ib = np.append(np.append(i4, i5), i6)
        i = np.intersect1d(np.unique(ia), np.unique(ib))
        self.revise_lists(i)

    def fgk_filter(self):
        """Includes only F, G, K spectral type stars in Target List"""

        spec = np.array(list(map(str, self.Spec)))
        iF = np.where(np.core.defchararray.startswith(spec, "F"))[0]
        iG = np.where(np.core.defchararray.startswith(spec, "G"))[0]
        iK = np.where(np.core.defchararray.startswith(spec, "K"))[0]
        i = np.append(np.append(iF, iG), iK)
        i = np.unique(i)
        self.revise_lists(i)

    def vis_mag_filter(self, Vmagcrit):
        """Includes stars which are below the maximum apparent visual magnitude

        Args:
            Vmagcrit (float):
                maximum apparent visual magnitude

        """

        i = np.where(self.Vmag < Vmagcrit)[0]
        self.revise_lists(i)

    def outside_IWA_filter(self):
        """Includes stars with planets with orbits outside of the IWA"""

        PPop = self.PlanetPopulation
        OS = self.OpticalSystem

        s = np.tan(OS.IWA) * self.dist
        L = np.sqrt(self.L) if PPop.scaleOrbits else 1.0
        i = np.where(s < L * np.max(PPop.rrange))[0]
        self.revise_lists(i)

    def zero_lum_filter(self):
        """Filter Target Stars with 0 luminosity"""
        i = np.where(self.L != 0.0)[0]
        self.revise_lists(i)

    def max_dmag_filter(self):
        """Includes stars if maximum delta mag is in the allowed orbital range

        Removed from prototype filters. Prototype is already calling the
        completeness_filter with self.intCutoff_dMag

        """

        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel

        # s and beta arrays
        s = np.tan(self.int_WA) * self.dist
        if PPop.scaleOrbits:
            s /= np.sqrt(self.L)
        beta = np.array([1.10472881476178] * len(s)) * u.rad

        # fix out of range values
        below = np.where(s < np.min(PPop.rrange) * np.sin(beta))[0]
        above = np.where(s > np.max(PPop.rrange) * np.sin(beta))[0]
        s[below] = np.sin(beta[below]) * np.min(PPop.rrange)
        beta[above] = np.arcsin(s[above] / np.max(PPop.rrange))

        # calculate delta mag
        p = np.max(PPop.prange)
        Rp = np.max(PPop.Rprange)
        d = s / np.sin(beta)
        Phi = PPMod.calc_Phi(beta)
        i = np.where(deltaMag(p, Rp, d, Phi) < self.intCutoff_dMag)[0]
        self.revise_lists(i)

    def completeness_filter(self):
        """Includes stars if completeness is larger than the minimum value"""

        i = np.where(self.intCutoff_comp >= self.Completeness.minComp)[0]
        self.revise_lists(i)

    def revise_lists(self, sInds):
        """Replaces Target List catalog attributes with filtered values,
        and updates the number of target stars.

        Args:
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        if len(sInds) == 0:
            raise IndexError("Target list filtered to empty.")

        for att in self.catalog_atts:
            if att == "coords":
                ra = self.coords.ra[sInds].to("deg")
                dec = self.coords.dec[sInds].to("deg")
                self.coords = SkyCoord(ra, dec, self.dist.to("pc"))
            else:
                if getattr(self, att).size != 0:
                    setattr(self, att, getattr(self, att)[sInds])
        for key in self.F0dict:
            self.F0dict[key] = self.F0dict[key][sInds]
        try:
            self.Completeness.revise_updates(sInds)
        except AttributeError:
            pass
        self.nStars = len(sInds)
        assert self.nStars, "Target list is empty: nStars = %r" % self.nStars

    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses

        This method calculates stellar mass via the formula relating absolute V
        magnitude and stellar mass.  The values are in units of solar mass.

        Function called by reset sim

        """

        # 'approximate' stellar mass
        self.MsEst = (
            10.0 ** (0.002456 * self.MV**2 - 0.09711 * self.MV + 0.4365)
        ) * u.solMass
        # normally distributed 'error'
        err = (np.random.random(len(self.MV)) * 2.0 - 1.0) * 0.07
        self.MsTrue = (1.0 + err) * self.MsEst

        # if additional filters are desired, need self.catalog_atts fully populated
        if not hasattr(self.catalog_atts, "MsEst"):
            self.catalog_atts.append("MsEst")
        if not hasattr(self.catalog_atts, "MsTrue"):
            self.catalog_atts.append("MsTrue")

    def starprop(self, sInds, currentTime, eclip=False):
        """Finds target star positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method uses ICRS coordinates which is approximately the same as
        equatorial coordinates.

        Args:
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Target star positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of pc. Will return an m x n x 3 array
                where m is size of currentTime, n is size of sInds. If either m or
                n is 1, will return n x 3 or m x 3.

        Note: Use eclip=True to get ecliptic coordinates.

        """

        # if multiple time values, check they are different otherwise reduce to scalar
        if currentTime.size > 1:
            if np.all(currentTime == currentTime[0]):
                currentTime = currentTime[0]

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        # get all array sizes
        nStars = sInds.size
        nTimes = currentTime.size

        # if the starprop_static method was created (staticStars is True), then use it
        if self.starprop_static is not None:
            r_targ = self.starprop_static(sInds, currentTime, eclip)
            if nTimes == 1 or nStars == 1 or nTimes == nStars:
                return r_targ
            else:
                return np.tile(r_targ, (nTimes, 1, 1))

        # target star ICRS coordinates
        coord_old = self.coords[sInds]
        # right ascension and declination
        ra = coord_old.ra
        dec = coord_old.dec
        # directions
        p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(sInds.size)])
        q0 = np.array(
            [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
        )
        r0 = coord_old.cartesian.xyz / coord_old.distance
        # proper motion vector
        mu0 = p0 * self.pmra[sInds] + q0 * self.pmdec[sInds]
        # space velocity vector
        v = mu0 / self.parx[sInds] * u.AU + r0 * self.rv[sInds]
        # set J2000 epoch
        j2000 = Time(2000.0, format="jyear")

        # if only 1 time in currentTime
        if nTimes == 1 or nStars == 1 or nTimes == nStars:
            # target star positions vector in heliocentric equatorial frame
            dr = v * (currentTime.mjd - j2000.mjd) * u.day
            r_targ = (coord_old.cartesian.xyz + dr).T.to("pc")

            if eclip:
                # transform to heliocentric true ecliptic frame
                coord_new = SkyCoord(
                    r_targ[:, 0],
                    r_targ[:, 1],
                    r_targ[:, 2],
                    representation_type="cartesian",
                )
                r_targ = coord_new.heliocentrictrueecliptic.cartesian.xyz.T.to("pc")
            return r_targ

        # create multi-dimensional array for r_targ
        else:
            # target star positions vector in heliocentric equatorial frame
            r_targ = np.zeros([nTimes, nStars, 3]) * u.pc
            for i, m in enumerate(currentTime):
                dr = v * (m.mjd - j2000.mjd) * u.day
                r_targ[i, :, :] = (coord_old.cartesian.xyz + dr).T.to("pc")

            if eclip:
                # transform to heliocentric true ecliptic frame
                coord_new = SkyCoord(
                    r_targ[i, :, 0],
                    r_targ[i, :, 1],
                    r_targ[i, :, 2],
                    representation_type="cartesian",
                )
                r_targ[i, :, :] = coord_new.heliocentrictrueecliptic.cartesian.xyz.T.to(
                    "pc"
                )
            return r_targ

    def starF0(self, sInds, mode):
        """Return the spectral flux density of the requested stars for the
        given observing mode.  Caches results internally for faster access in
        subsequent calls.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest
            mode (dict):
                Observing mode dictionary (see OpticalSystem)

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Spectral flux densities in units of ph/m**2/s/nm.

        """

        if mode["hex"] in self.F0dict:
            tmp = np.isnan(self.F0dict[mode["hex"]][sInds])
            if np.any(tmp):
                inds = np.where(tmp)[0]
                for j in inds:
                    self.F0dict[mode["hex"]][sInds[j]] = self.F0(
                        mode["BW"], mode["lam"], spec=self.Spec[sInds[j]]
                    )
        else:
            self.F0dict[mode["hex"]] = np.full(self.nStars, np.nan) * (
                u.ph / u.s / u.m**2 / u.nm
            )
            for j in sInds:
                self.F0dict[mode["hex"]][j] = self.F0(
                    mode["BW"], mode["lam"], spec=self.Spec[j]
                )

        return self.F0dict[mode["hex"]][sInds]

    def starMag(self, sInds, lam):
        r"""Calculates star visual magnitudes with B-V color using empirical fit
        to data from Pecaut and Mamajek (2013, Appendix C).
        The expression for flux is accurate to about 7%, in the range of validity
        400 nm < :math:`\lambda` < 1000 nm (Traub et al. 2016).

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest
            lam (astropy Quantity):
                Wavelength in units of nm

        Returns:
            ~numpy.ndarray(float):
                Star magnitudes at wavelength from B-V color

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        Vmag = self.Vmag[sInds]
        BV = self.BV[sInds]

        lam_um = lam.to("um").value
        if lam_um < 0.550:
            b = 2.20
        else:
            b = 1.54
        mV = Vmag + b * BV * (1.0 / lam_um - 1.818)

        return mV

    def stellarTeff(self, sInds):
        """Calculate the effective stellar temperature based on B-V color.

        This method uses the empirical fit from Ballesteros (2012)
        doi:10.1209/0295-5075/97/34008

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Stellar effective temperatures in degrees K

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        Teff = (
            4600.0
            * u.K
            * (
                1.0 / (0.92 * self.BV[sInds] + 1.7)
                + 1.0 / (0.92 * self.BV[sInds] + 0.62)
            )
        )

        return Teff

    def radiusFromMass(self, sInds):
        """Estimates the star radius based on its mass
        Table 2, ZAMS models pg321
        STELLAR MASS-LUMINOSITY AND MASS-RADIUS RELATIONS OSMAN DEMIRCAN and GOKSEL
        KAHRAMAN 1991

        Args:
            sInds (~numpy.ndarray(int)):
                star indices

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Star radius estimates
        """

        M = self.MsTrue[sInds].value  # Always use this??
        a = -0.073
        b = 0.668
        starRadius = 10 ** (a + b * np.log(M))

        return starRadius * u.R_sun

    def gen_inclinations(self, Irange):
        """Randomly Generate Inclination of Target System Orbital Plane

        Args:
            Irange (~numpy.ndarray(float)):
                the range to generate inclinations over

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                System inclinations
        """
        C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
        return (
            np.arccos(np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=self.nStars))
        ).to("deg")

    def calc_HZ_inner(
        self,
        sInds,
        S_inner=1.7665,
        A_inner=1.3351e-4,
        B_inner=3.1515e-9,
        C_inner=-3.3488e-12,
        **kwargs,
    ):
        """
        Convenience function to find the inner edge of the habitable zone using the
        emperical approach in calc_HZ().

        Default contstants: Recent Venus limit Inner edge" , Kaltinegger et al 2018,
        Table 1.

        """
        return self.calc_HZ(sInds, S_inner, A_inner, B_inner, C_inner, **kwargs)

    def calc_HZ_outer(
        self,
        sInds,
        S_outer=0.324,
        A_outer=5.3221e-5,
        B_outer=1.4288e-9,
        C_outer=-1.1049e-12,
        **kwargs,
    ):
        """
        Convenience function to find the inner edge of the habitable zone using the
        emperical approach in calc_HZ().

        The default outer limit constants are the Early Mars outer limit,
        Kaltinegger et al (2018) Table 1.

        """
        return self.calc_HZ(sInds, S_outer, A_outer, B_outer, C_outer, **kwargs)

    def calc_IWA_AU(self, sInds, **kwargs):
        """

        Convenience function to find the separation from the star of the IWA

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest

        Returns:
            Quantity array:
                separation from the star of the IWA in AU
        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        return (
            self.dist[sInds].to(u.parsec).value
            * self.OpticalSystem.IWA.to(u.arcsec).value
            * u.AU
        )

    def calc_HZ(self, sInds, S, A, B, C, arcsec=False):
        """finds the inner or outer edge of the habitable zone

        This method uses the empirical fit from Kaltinegger et al  (2018) and
        references therein, https://arxiv.org/pdf/1903.11539.pdf

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest
            S (float):
                Constant
            A (float):
                Constant
            B (float):
                Constant
            C (float):
                Constant
            arcsec (bool):
                If True returns result arcseconds instead of AU
        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
               limit of HZ in AU or arcseconds

        """
        # Fill in photometry so we have all the luminousities
        self.fillPhotometryVals()

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        T_eff = self.stellarTeff(sInds)

        T_star = (5780 * u.K - T_eff).to(u.K).value

        Seff = S + A * T_star + B * T_star**2 + C * T_star**3

        d_HZ = np.sqrt((self.L[sInds]) / Seff) * u.AU

        if arcsec:
            return (
                d_HZ.to(u.AU).value / self.dist[sInds].to(u.parsec).value
            ) * u.arcsecond
        else:
            return d_HZ

    def calc_EEID(self, sInds, arcsec=False):
        """Finds the earth equivalent insolation distance (EEID)


        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest
        arcsec (bool):
            If True returns result arcseconds instead of AU

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                limit of HZ in AU or arcseconds

        """
        # Fill in photometry so we have all the luminousities
        self.fillPhotometryVals()

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        d_EEID = (1 / ((1 * u.AU) ** 2 * (self.L[sInds]))) ** (-0.5)
        # ((L_sun/(1*AU^2)/(0.25*L_sun)))^(-0.5)
        if arcsec:
            return (
                d_EEID.to(u.AU).value / self.dist[sInds].to(u.parsec).value
            ) * u.arcsecond
        else:
            return d_EEID

    def dump_catalog(self):
        """Creates a dictionary of stellar properties for archiving use.

        Args:
            None

        Returns:
            dict:
                Dictionary of star catalog properties

        """
        atts = [
            "Name",
            "Spec",
            "parx",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "dist",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
            "MsEst",
            "MsTrue",
            "int_comp",
            "I",
        ]
        # Not sure if MsTrue and others can be dumped properly...

        catalog = {atts[i]: getattr(self, atts[i]) for i in np.arange(len(atts))}

        return catalog

    def constructIPACurl(
        self,
        tableInput="exoplanets",
        columnsInputList=[
            "pl_hostname",
            "ra",
            "dec",
            "pl_discmethod",
            "pl_pnum",
            "pl_orbper",
            "pl_orbsmax",
            "pl_orbeccen",
            "pl_orbincl",
            "pl_bmassj",
            "pl_radj",
            "st_dist",
            "pl_tranflag",
            "pl_rvflag",
            "pl_imgflag",
            "pl_astflag",
            "pl_omflag",
            "pl_ttvflag",
            "st_mass",
            "pl_discmethod",
        ],
        formatInput="json",
    ):
        """
        Extracts Data from IPAC
        Instructions for to interface with ipac using API
        https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/
            index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01

        Args:
            tableInput (str):
                describes which table to query
            columnsInputList (list):
                List of strings from https://exoplanetarchive.ipac.caltech.edu/docs/
                                            API_exoplanet_columns.html
            formatInput (string):
                string describing output type. Only support JSON at this time
        Returns:
            data (dict):
                a dictionary of IPAC data
        """
        baseURL = (
            "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"
        )
        tablebaseURL = "table="
        # tableInput = "exoplanets" # exoplanets to query exoplanet table
        columnsbaseURL = "&select="  # Each table input must be separated by a comma
        # columnsInputList = ['pl_hostname','ra','dec','pl_discmethod','pl_pnum',
        #                     'pl_orbper','pl_orbsmax','pl_orbeccen',
        #                     'pl_orbincl','pl_bmassj','pl_radj','st_dist',
        #                     'pl_tranflag','pl_rvflag','pl_imgflag',
        #                     'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass',
        #                     'pl_discmethod']
        # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
        # for explanations

        """
        pl_hostname - Stellar name most commonly used in the literature.
        ra - Right Ascension of the planetary system in decimal degrees.
        dec - Declination of the planetary system in decimal degrees.
        pl_discmethod - Method by which the planet was first identified.
        pl_pnum - Number of planets in the planetary system.
        pl_orbper - Time the planet takes to make a complete orbit around the host star
                    or system.
        pl_orbsmax - The longest radius of an elliptic orbit, or, for exoplanets
                    detected via gravitational microlensing or direct imaging,
                    the projected separation in the plane of the sky. (AU)
        pl_orbeccen - Amount by which the orbit of the planet deviates from a
                    perfect circle.
        pl_orbincl - Angular distance of the orbital plane from the line of sight.
        pl_bmassj - Best planet mass estimate available, in order of preference: Mass,
                    M*sin(i)/sin(i), or M*sin(i), depending on availability,
                    and measured in Jupiter masses. See Planet Mass M*sin(i) Provenance
                    (pl_bmassprov) to determine which measure applies.
        pl_radj - Length of a line segment from the center of the planet to its surface,
                  measured in units of radius of Jupiter.
        st_dist - Distance to the planetary system in units of parsecs.
        pl_tranflag - Flag indicating if the planet transits its host star (1=yes, 0=no)
        pl_rvflag - Flag indicating if the planet host star exhibits radial velocity
                    variations due to the planet (1=yes, 0=no)
        pl_imgflag - Flag indicating if the planet has been observed via imaging
                     techniques (1=yes, 0=no)
        pl_astflag - Flag indicating if the planet host star exhibits astrometrical
                     variations due to the planet (1=yes, 0=no)
        pl_omflag - Flag indicating whether the planet exhibits orbital modulations
                    on the phase curve (1=yes, 0=no)
        pl_ttvflag - Flag indicating if the planet orbit exhibits transit timing
                     variations from another planet in the system (1=yes, 0=no).
                     Note: Non-transiting planets discovered via the transit timing
                     variations of another planet in the system will not hav
                     their TTV flag set, since they do not themselves demonstrate TTVs.
        st_mass - Amount of matter contained in the star, measured in units of masses
                  of the Sun.
        pl_discmethod - Method by which the planet was first identified.
        """

        columnsInput = ",".join(columnsInputList)
        formatbaseURL = "&format="
        # formatInput = 'json'
        # https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#format

        # Different acceptable "Inputs" listed at
        # https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/
        #       index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01

        myURL = (
            baseURL
            + tablebaseURL
            + tableInput
            + columnsbaseURL
            + columnsInput
            + formatbaseURL
            + formatInput
        )
        response = urllib.request.urlopen(myURL)
        data = json.load(response)

        return data

    def setOfStarsWithKnownPlanets(self, data):
        """From the data dict created in this script, this method extracts the set
        of unique star names

        Args:
            data (dict):
                dict containing the pl_hostname of each star
        Returns:
            list:
                list of star names with a known planet

        """
        starNames = list()
        for i in np.arange(len(data)):
            starNames.append(data[i]["pl_hostname"])
        return list(set(starNames))

    def loadAliasFile(self):
        """
        Args:
            None:

        Returns:
            list:
                List of aliases

        """
        # OLD aliasname = 'alias_4_11_2019.pkl'
        aliasname = "alias_10_07_2019.pkl"
        tmp1 = inspect.getfile(self.__class__).split("/")[:-2]
        tmp1.append("util")
        self.classpath = "/".join(tmp1)
        # self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        # vprint(inspect.getfile(self.__class__))
        self.alias_datapath = os.path.join(self.classpath, aliasname)
        # Load pkl and outspec files
        try:
            with open(self.alias_datapath, "rb") as f:  # load from cache
                alias = pickle.load(f, encoding="latin1")
        except (AttributeError, FileNotFoundError):
            self.vprint("Failed to open fullPathPKL %s" % self.alias_datapath)
            pass
        return alias

    def createKnownPlanetBoolean(self, alias, starsWithPlanets):
        """
        Args:
            alias ():

            starsWithPlanets ():

        Returns:
            ~numpy.ndarray:
                boolean numpy array indicating whether the star has a planet (true)
                or does not have a planet (false)

        """
        # Create List of Stars with Known Planets
        knownPlanetBoolean = np.zeros(self.nStars, dtype=bool)
        for i in np.arange(self.nStars):
            # Does the Star Have a Known Planet
            starName = self.Name[i]  # Get name of the current star
            if starName in alias[:, 1]:
                indWhereStarName = np.where(alias[:, 1] == starName)[0][
                    0
                ]  # there should be only 1
                starNum = alias[
                    indWhereStarName, 3
                ]  # this number is identical for all names of a target
                aliases = [
                    alias[j, 1] for j in np.arange(len(alias)) if alias[j, 3] == starNum
                ]  # creates a list of the known aliases
                if np.any(
                    [
                        True if aliases[j] in starsWithPlanets else False
                        for j in np.arange(len(aliases))
                    ]
                ):
                    knownPlanetBoolean[i] = 1
        return knownPlanetBoolean

    def calc_intCutoff_dMag(self, mode):
        """
        This calculates the delta magnitude for each target star that
        corresponds to the cutoff integration time. Uses the working
        angle, int_WA, used to calculate integration times.

        Args:
            mode (dict):
                Observing mode dictionary (see OpticalSystem)

        Returns:
            ~numpy.ndarray(float):
                Array with dMag values if exposed for the integration cutoff time
                for each target star
        """
        stars_hash = hashlib.md5(str(self.StarCatalog.Name).encode("utf-8")).hexdigest()
        intCutoff_dMag_path = Path(
            self.cachedir, self.base_filename + stars_hash + ".intCutoff_dMag"
        )
        if intCutoff_dMag_path.exists():
            self.vprint(f"Loaded intCutoff_dMag values from {intCutoff_dMag_path}")
            with open(intCutoff_dMag_path, "rb") as f:
                intCutoff_dMag = pickle.load(f)
        else:
            self.vprint("Calculating intCutoff_dMag")
            OS = self.OpticalSystem
            ZL = self.ZodiacalLight
            sInds = np.arange(self.nStars)

            # Getting the inputs into the right formats
            intTime = OS.intCutoff
            intTimes = np.repeat(intTime.value, len(sInds)) * intTime.unit

            # Use minimum global fZ value
            # set the fZminglobal value in ZL for the desired mode
            fZminglobal = ZL.global_zodi_min(mode)
            fZ = np.repeat(fZminglobal, len(sInds))
            fEZ = np.repeat(ZL.fEZ0, len(sInds))

            intCutoff_dMag = OS.calc_dMag_per_intTime(
                intTimes, self, sInds, fZ, fEZ, self.int_WA, mode
            ).reshape((len(intTimes),))
            with open(intCutoff_dMag_path, "wb") as f:
                pickle.dump(intCutoff_dMag, f)
            self.vprint(f"intCutoff_dMag values stored in {intCutoff_dMag_path}")
        return intCutoff_dMag

    def calc_saturation_dMag(self, mode):
        """
        This calculates the delta magnitude for each target star that
        corresponds to an infinite integration time. Uses the working
        angle, int_WA, used to calculate integration times.

        Args:
            mode (dict):
                Observing mode dictionary (see OpticalSystem)

        Returns:
            ~numpy.ndarray(float):
                Array with dMag values if exposed for the integration cutoff time for
                each target star
        """
        stars_hash = hashlib.md5(str(self.StarCatalog.Name).encode("utf-8")).hexdigest()
        saturation_dMag_path = Path(
            self.cachedir, self.base_filename + stars_hash + ".sat_dMag"
        )
        if saturation_dMag_path.exists():
            self.vprint(f"Loaded saturation_dMag values from {saturation_dMag_path}")
            with open(saturation_dMag_path, "rb") as f:
                saturation_dMag = pickle.load(f)
        else:
            OS = self.OpticalSystem
            ZL = self.ZodiacalLight
            sInds = np.arange(self.nStars)

            fZminglobal = ZL.global_zodi_min(mode)
            fZ = np.repeat(fZminglobal, len(sInds))
            fEZ = np.repeat(ZL.fEZ0, len(sInds))

            saturation_dMag = np.zeros(len(sInds))
            for i, sInd in enumerate(tqdm(sInds, desc="Calculating saturation_dMag")):
                args = (
                    self,
                    [sInd],
                    [fZ[i].value] * fZ.unit,
                    [fEZ[i].value] * fEZ.unit,
                    [self.int_WA[i].value] * self.int_WA.unit,
                    mode,
                    None,
                )
                singularity_res = root_scalar(
                    OS.int_time_denom_obj, args=args, method="brentq", bracket=[10, 40]
                )
                singularity_dMag = singularity_res.root
                saturation_dMag[i] = singularity_dMag

            # This block is not relevant w/ current implementation, but this
            # will create an interpolant of the saturation dMag as a function
            # of WA
            # initial_WA_range = np.linspace(mode['IWA'], mode['OWA'], 5)
            # next_WA_range = np.linspace(initial_WA_range[0], initial_WA_range[1], 4)
            # WA_range = np.unique(np.concatenate((next_WA_range, initial_WA_range)))
            # saturation_dMags = []
            # for WA in WA_range:
            # args = (self, 0, fZminglobal, ZL.fEZ0, WA, mode, None)
            # singularity_res = root_scalar(OS.int_time_denom_obj,
            # args=args, method='brentq',
            # bracket=[10, 40])
            # singularity_dMag = singularity_res.root
            # saturation_dMags.append(singularity_dMag)
            # saturation_dMag_curve = scipy.interpolate.interp1d(WA_range,
            #                                                   saturation_dMags)
            with open(saturation_dMag_path, "wb") as f:
                pickle.dump(saturation_dMag, f)
            self.vprint(f"saturation_dMag values stored in {saturation_dMag_path}")
        return saturation_dMag
