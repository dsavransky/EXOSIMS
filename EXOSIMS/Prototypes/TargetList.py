from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.getExoplanetArchive import getExoplanetArchiveAliases
from EXOSIMS.util.utils import genHexStr
from MeanStars import MeanStars
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits
import re
import os.path
import json
from pathlib import Path
from scipy.optimize import root_scalar
from tqdm import tqdm
import pickle
import pkg_resources
import warnings
import gzip


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
        cachedir (str or None):
            Full path to cache directory.  If None, use default.
        filter_for_char (bool):
            Use spectroscopy observation mode (rather than the default detection mode)
            for all calculations. Defaults False.
        earths_only (bool):
            Used in upstream modules.  Alias for filter_for_char. Defaults False.
        getKnownPlanets (bool):
            Retrieve information about which stars have known planets along with all
            known (to the NASA Exoplanet Archive) target aliases. Defaults False.

            .. warning::

                This can take a *very* long time for large star catalogs if starting
                from scratch. For this reason, the cache comes pre-loaded with all
                entries coresponding to EXOCAT1.

        int_WA (float or numpy.ndarray or None):
            Working angle (arcsec) at which to caluclate integration times for default
            observations.  If input is scalar, all targets will get the same value.  If
            input is an array, it must match the size of the input catalog.  If None,
            a default value of halway between the inner and outer working angle of the
            default observing mode will be used.  If the OWA is infinite, twice the IWA
            is used.
        int_dMag (float or numpy.ndarray):
            :math:`\\Delta\\textrm{mag}` to assume when calculating integration time for
            default observations. If input is scalar, all targets will get the same
            value.  If input is an array, it must match the size of the input catalog.
            Defaults to 25
        scaleWAdMag (bool):
            If True, rescale int_dMag and int_WA for all stars based on luminosity and
            to ensure that WA is within the IWA/OWA. Defaults False.
        int_dMag_offset (float):
            Offset applied to int_dMag when scaleWAdMag is True.
        popStars (list, optional):
            Remove given stars (by exact name matching) from target list.
            Defaults None.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        BackgroundSources (:ref:`BackgroundSources`):
            :ref:`BackgroundSources` object
        BC (numpy.ndarray):
            Bolometric correction (V band)
        Binary_Cut (numpy.ndarray):
            Boolean - True is target has close companion.
        Bmag (numpy.ndarray):
            B band magniutde
        BV (numpy.ndarray):
            B-V color
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        calc_char_int_comp (bool):
            Boolean flagged by ``filter_for_char`` or ``earths_only``
        catalog_atts (list):
            All star catalog attributes that were copied in
        Completeness (:ref:`Completeness`):
            :ref:`Completeness` object
        coords (astropy.coordinates.sky_coordinate.SkyCoord):
            Target coordinates
        dist (astropy.units.quantity.Quantity):
            Target distances
        earths_only (bool):
            Used in upstream modules.  Alias for filter_for_char.
        explainFiltering (bool):
            Print informational messages at each target list filtering step.
        F0dict (dict):
            Internal storage of pre-computed zero-mag flux values that is populated
            each time an F0 is requested for a particular target.
        fillPhotometry (bool):
            Attempt to fill in missing target photometric  values using interpolants of
            tabulated values for the stellar type. See MeanStars documentation for
            more details.
        filter_for_char (bool):
            Use spectroscopy observation mode (rather than the default detection mode)
            for all calculations.
        filterBinaries (bool):
            Remove all binary stars or stars with known close companions from target
            list.
        getKnownPlanets (bool):
            Grab the list of known planets and target aliases from the NASA Exoplanet
            Archive
        hasKnownPlanet (numpy.ndarray):
            bool array indicating whether a target has known planets.  Only populated
            if attribute ``getKnownPlanets`` is True. Otherwise all entries are False.
        Hmag (numpy.ndarray):
            H band magnitudes
        I (astropy.units.quantity.Quantity):
            Inclinations of target system orbital planes
        Imag (numpy.ndarray):
            I band magnitudes
        int_comp (numpy.ndarray):
            Completeness value for each target star for default observation WA and
            :math:`\Delta{\\textrm{mag}}`.
        int_dMag (numpy.ndarray):
             :math:`\Delta{\\textrm{mag}}` used for default observation integration time
             calculation
        int_dMag_offset (int):
            Offset applied to int_dMag when scaleWAdMag is True.
        int_WA (astropy.units.quantity.Quantity):
            Working angle used for integration time calculation (angle)
        intCutoff_comp (numpy.ndarray):
            Completeness values of all targets corresponding to the cutoff integration
            time set in the optical system.
        intCutoff_dMag (numpy.ndarray):
            :math:`\Delta{\\textrm{mag}}` of all targets corresponding to the cutoff
            integration time set in the optical system.
        Jmag (numpy.ndarray):
            J band magnitudes
        keepStarCatalog (bool):
            Keep star catalog object as attribute after TargetList is built.
        Kmag (numpy.ndarray):
            K band mangitudes
        L (numpy.ndarray):
            Luminosities in solar luminosities (linear scale!)
        ms (MeanStars.MeanStars.MeanStars):
            MeanStars object
        MsEst (astropy.units.quantity.Quantity):
            'approximate' stellar masses
        MsTrue (astropy.units.quantity.Quantity):
            'true' stellar masses
        MV (numpy.ndarray):
            Absolute V band magnitudes
        Name (numpy.ndarray):
            Target names (str array)
        nStars (int):
            Number of stars currently in target list
        OpticalSystem (:ref:`OpticalSystem`):
            :ref:`OpticalSystem` object
        parx (astropy.units.quantity.Quantity):
            Parallaxes
        PlanetPhysicalModel (:ref:`PlanetPhysicalModel`):
            :ref:`PlanetPhysicalModel` object
        PlanetPopulation (:ref:`PlanetPopulation`):
            :ref:`PlanetPopulation` object
        pmdec (astropy.units.quantity.Quantity):
            Proper motion in declination
        pmra (astropy.units.quantity.Quantity):
            Proper motion in right ascension
        popStars (list, optional):
            List of target names that were removed from target list
        PostProcessing (:ref:`PostProcessing`):
            :ref:`PostProcessing` object
        required_catalog_atts (list):
            Catalog attributes that may not be missing or nan
        Rmag (numpy.ndarray):
            R band magnitudes
        rv (astropy.units.quantity.Quantity):
            Radial velocities
        saturation_comp (numpy.ndarray):
            Maximum possible completeness values of all targets.
        saturation_dMag (numpy.ndarray):
            :math:`\Delta{\\textrm{mag}}` at which completness stops increasing for all
            targets.
        scaleWAdMag (bool):
            Rescale int_dMag and int_WA for all stars based on luminosity and to ensure
            that WA is within the IWA/OWA.
        Spec (numpy.ndarray):
            Spectral type strings. Will be strictly in G0V format.
        specdatapath (str):
            Full path to spectral data folder
        specdict (dict):
            Dictionary of spectral types
        specindex (dict):
            Index of spectral types
        speclist (list):
            List of spectral types available
        specliste (numpy.ndarray):
            Available spectral types split in class, subclass, and luminosity class
        spectral_class (numpy.ndarray):
            nStars x 3. First column is spectral class, second is spectral subclass and
            third is luminosity class.
        spectypenum (numpy.ndarray):
            Numerical value of spectral type for matching
        staticStars (bool):
            Do not apply proper motions to stars.  Stars always at mission start time
            positions.
        Umag (numpy.ndarray):
            U band magnitudes
        Vmag (numpy.ndarray):
            V band magnitudes
        ZodiacalLight (:ref:`ZodiacalLight`):
            :ref:`ZodiacalLight` object

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
        cachedir=None,
        filter_for_char=False,
        earths_only=False,
        getKnownPlanets=False,
        int_WA=None,
        int_dMag=25,
        scaleWAdMag=False,
        int_dMag_offset=1,
        popStars=None,
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

        # validate TargetList boolean flags
        assert isinstance(staticStars, bool), "staticStars must be a boolean."
        assert isinstance(keepStarCatalog, bool), "keepStarCatalog must be a boolean."
        assert isinstance(fillPhotometry, bool), "fillPhotometry must be a boolean."
        assert isinstance(explainFiltering, bool), "explainFiltering must be a boolean."
        assert isinstance(filterBinaries, bool), "filterBinaries must be a boolean."
        assert isinstance(getKnownPlanets, bool), "getKnownPlanets must be a boolean."

        self.getKnownPlanets = bool(getKnownPlanets)
        self.staticStars = bool(staticStars)
        self.keepStarCatalog = bool(keepStarCatalog)
        self.fillPhotometry = bool(fillPhotometry)
        self.explainFiltering = bool(explainFiltering)
        self.filterBinaries = bool(filterBinaries)
        self.filter_for_char = bool(filter_for_char)
        self.earths_only = bool(earths_only)

        # list of target names to remove from targetlist
        if popStars:
            assert isinstance(popStars, list), "popStars must be a list."
        self.popStars = popStars

        # This parameter is used to modify the dMag value used to calculate
        # integration time
        self.int_dMag_offset = int_dMag_offset
        # Flag for whether to do luminosity scaling
        self.scaleWAdMag = scaleWAdMag

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec"]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # set up stuff for spectral type conversion
        # Craete a MeanStars object for future use:
        self.ms = MeanStars()

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

        # identify default detection mode
        detmode = list(
            filter(
                lambda mode: mode["detectionMode"], self.OpticalSystem.observingModes
            )
        )[0]

        # Define int_WA if None provided
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

        # set Star Catalog attributes
        self.set_catalog_attributes()

        # bring in StarCatalog attributes into TargetList
        self.populate_target_list(**specs)
        if self.explainFiltering:
            print("%d targets imported from star catalog." % self.nStars)

        # remove any requested stars from TargetList
        if self.popStars is not None:
            tmp = np.arange(self.nStars)
            for n in self.popStars:
                tmp = tmp[self.Name != n]

            self.revise_lists(tmp)

            if self.explainFiltering:
                print(
                    "%d targets remain after removing requested targets." % self.nStars
                )

        # populate spectral types and, if requested, attempt to fill in any crucial
        # missing bits of information
        self.fillPhotometryVals()

        # filter out nan attribute values from Star Catalog
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering." % self.nStars)

        # filter out target stars with 0 luminosity
        self.zero_lum_filter()
        if self.explainFiltering:
            print(
                "%d targets remain after removing zero luminosity targets."
                % self.nStars
            )

        # Calculate saturation and intCutoff delta mags and completeness values
        self.calc_saturation_and_intCutoff_vals()

        # populate completeness values
        self.int_comp = self.Completeness.target_completeness(self)
        self.catalog_atts.append("int_comp")

        # calculate 'true' and 'approximate' stellar masses
        self.vprint("Calculating target stellar masses.")
        self.stellar_mass()

        # Calculate Star System Inclinations
        self.I = self.gen_inclinations(self.PlanetPopulation.Irange)

        # generate any completeness update data needed
        self.Completeness.gen_update(self)

        # apply any requeted additional filters
        self.filter_target_list(**specs)

        # get target system information from exopoanet archive if requested
        if self.getKnownPlanets:
            self.queryNEAsystems()
        else:
            self.hasKnownPlanet = np.zeros(self.nStars, dtype=bool)
            self.catalog_atts.append("hasKnownPlanet")

        # have target list, no need for catalog now (unless asked to retain)
        # instead, just keep the class of the star catalog for bookkeeping
        if not self.keepStarCatalog:
            self.StarCatalog = self.StarCatalog.__class__

        # add nStars to outspec (this is a rare exception to not allowing extraneous
        # information into the outspec).
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

    def __str__(self):
        """String representation of the Target List object

        When the command 'print' is used on the Target List object, this method
        will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Target List class object attributes"

    def set_catalog_attributes(self):
        """Hepler method that sets possible and required catalog attributes.

        Sets attributes:
            catalog_atts (list):
                Attributes to try to copy from star catalog.  Missing ones will be
                ignored and removed from this list.
            required_catalog_atts(list):
                Attributes that cannot be missing or nan.

        .. note::

            This is a separate method primarily for downstream implementations that wish
            to modify the catalog attributes.  Overloaded methods can first call this
            method to get the base values, or overwrite them entirely.

        """

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

        # required catalog attributes
        self.required_catalog_atts = [
            "Name",
            "Vmag",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "dist",
        ]

    def populate_target_list(self, **specs):
        """This function is responsible for populating values from the star
        catalog into the target list attributes and enforcing attribute requirements.


        Args:
            **specs:
                :ref:`sec:inputspec`

        """
        SC = self.StarCatalog

        # bring Star Catalog values to top level of Target List
        missingatts = []
        for att in self.catalog_atts:
            if not hasattr(SC, att):
                assert (
                    att not in self.required_catalog_atts
                ), f"Star catalog attribute {att} is missing but listed as required."
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

        # add catalog _outspec to our own
        self._outspec.update(SC._outspec)

    def calc_saturation_and_intCutoff_vals(self):
        """
        Calculates the saturation and integration cutoff time dMag and
        completeness values, saves them as attributes, refines the dMag used to
        calculate integration times so it does not exceed the integration
        cutoff time dMag, and handles any orbit scaling necessary
        """

        # pad out int_WA and int_dMag to size of targetlist, as needed
        if len(self.int_WA) == 1:
            self.int_WA = np.repeat(self.int_WA, self.nStars)
        if len(self.int_dMag) == 1:
            self.int_dMag = np.repeat(self.int_dMag, self.nStars)
        # add these to the target list catalog attributes
        self.catalog_atts.append("int_dMag")
        self.catalog_atts.append("int_WA")

        # grab required modules and determine which observing mode to use
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

        # 1. Calculate the saturation dMag. This is stricly a function of
        # fZminglobal, ZL.fEZ0, self.int_WA, mode, and the current targetlist
        zodi_vals_str = f"{str(ZL.global_zodi_min(mode))} {str(ZL.fEZ0)}"
        stars_str = f"fillPhotometry:{self.fillPhotometry}" + ",".join(self.Name)
        int_WA_str = ",".join(self.int_WA.value.astype(str)) + str(self.int_WA.unit)

        # cache filename is the three class names, the vals hash, and the mode hash
        vals_hash = genHexStr(zodi_vals_str + stars_str + int_WA_str)
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"{OS.__class__.__name__}_{ZL.__class__.__name__}_"
            f"vals_{vals_hash}_mode_{mode['hex']}"
        )

        self.saturation_dMag = self.calc_saturation_dMag(mode, fname)

        # 2. Calculate the completeness value if the star is integrated for an
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

        # cache filename is the two class names and the vals hash
        satcomp_valstr = (
            ",".join(tmp_smin.to(u.AU).value.astype(str))
            + ",".join(tmp_smax.to(u.AU).value.astype(str))
            + ",".join(tmp_dMag.astype(str))
        )

        vals_hash = genHexStr(stars_str + satcomp_valstr)
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"{Comp.__class__.__name__}_vals_{vals_hash}"
        )

        # calculate or load from disk if cache exists
        saturation_comp_path = Path(self.cachedir, f"{fname}.sat_comp")
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
            self.vprint(f"saturation_comp values stored in {saturation_comp_path}")

        # 3. Find limiting dMag for intCutoff time. This is stricly a function of
        # OS.intCutoff, fZminglobal, ZL.fEZ0, self.int_WA, mode, and the current
        # targetlist
        vals_hash = genHexStr(
            f"{OS.intCutoff} " + zodi_vals_str + stars_str + int_WA_str
        )
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"{OS.__class__.__name__}_{ZL.__class__.__name__}_"
            f"vals_{vals_hash}_mode_{mode['hex']}"
        )

        self.intCutoff_dMag = self.calc_intCutoff_dMag(mode, fname)

        # 4. Calculate intCutoff completeness. This is a function of the exact same
        # things as the previous calculation, so we can recycle the filename
        intCutoff_comp_path = Path(self.cachedir, f"{fname}.intCutoff_comp")
        if intCutoff_comp_path.exists():
            self.vprint(f"Loaded intCutoff_comp values from {intCutoff_comp_path}")
            with open(intCutoff_comp_path, "rb") as f:
                self.intCutoff_comp = pickle.load(f)
        else:
            self.vprint("Calculating the integration cutoff time completeness")
            self.intCutoff_comp = Comp.comp_calc(
                tmp_smin.to(u.AU).value, tmp_smax.to(u.AU).value, self.intCutoff_dMag
            )
            with open(intCutoff_comp_path, "wb") as f:
                pickle.dump(self.intCutoff_comp, f)
            self.vprint(f"intCutoff_comp values stored in {intCutoff_comp_path}")

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

        # update catalog attributes for any future filtering
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
                if not np.any(lumclass):
                    specmatch = None
                else:
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
        Attempts to determine the spectral class and luminosity class of each
        target star. If ``self.fillPhotometry`` is True, attempts to reconstruct any
        missing spectral types from other avaialable information, and then fill
        missing L, B, V and BC information from spectral type.

        Uses the MeanStars object for regexps and data filling. This explicitly treats
        all stars as dwarfs. TODO: Update to use spectra for other luminosity classes.

        The data is from:
        "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester)

        See MeanStars documentation for futher details.
        """

        # first let's try to establish the spectral type
        self.spectral_class = np.ndarray((self.nStars, 3), dtype=object)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for j, s in enumerate(self.Spec):
                tmp = self.ms.matchSpecType(s)
                if tmp:
                    self.spectral_class[j] = tmp
                elif self.fillPhotometry:
                    # if we have a luminosity, try to reconstruct from that
                    if not (np.isnan(self.L[j])) and (self.L[j] != 0):
                        ind = self.ms.tableLookup("logL", np.log10(self.L[j]))
                        self.spectral_class[j] = self.ms.MK[ind], self.ms.MKn[ind], "V"
                    elif not (np.isnan(self.BV[j])) and (self.BV[j] != 0):
                        ind = self.ms.tableLookup("B-V", self.BV[j])
                        self.spectral_class[j] = self.ms.MK[ind], self.ms.MKn[ind], "V"
                    else:
                        self.spectral_class[j] = "", np.nan, ""
                else:
                    self.spectral_class[j] = "", np.nan, ""

        self.catalog_atts.append("spectral_class")
        self.revise_lists(np.where(self.spectral_class[:, 0] != "")[0])
        if self.explainFiltering:
            print(
                (
                    "{} targets remain after removing those where spectral class "
                    "cannot be established."
                ).format(self.nStars)
            )

        # remove all subdwarfs and white-dwarfs
        sInds = np.array(
            [j for j in range(self.nStars) if self.spectral_class[j, 0] in "OBAFGKM"]
        )
        self.revise_lists(sInds)
        if self.explainFiltering:
            print(
                ("{} targets remain after removing white dwarfs and subdwarfs").format(
                    self.nStars
                )
            )

        # Update all spectral strings to their normalized values
        self.Spec = np.array(
            [f"{s[0]}{int(np.round(s[1]))}{s[2]}" for s in self.spectral_class]
        )

        # if we don't need to fill photometry values, we're done here
        if not (self.fillPhotometry):
            return

        # first check on absolute V mags
        if np.all(self.MV == 0):
            self.MV *= np.nan
        if np.all(self.Vmag == 0):
            self.Vmag *= np.nan
        if np.any(np.isnan(self.MV)):
            inds = np.where(np.isnan(self.MV))[0]
            for i in inds:
                # try to get from apparent mag
                if not (np.isnan(self.Vmag[i])):
                    self.MV[i] = self.Vmag[i] - 5 * (
                        np.log10(self.dist[i].to("pc").value) - 1
                    )
                # if not, get from MeanStars
                else:
                    self.MV[i] = self.ms.SpTOther(
                        "Mv", self.spectral_class[i][0], self.spectral_class[i][1]
                    )

        # We should now have all the absolute V mags and so should be able to just
        # fill in missing apparent V mags from those.
        if np.any(np.isnan(self.Vmag)):
            inds = np.isnan(self.Vmag)
            self.Vmag[inds] = self.MV[inds] + 5 * (
                np.log10(self.dist[inds].to("pc").value) - 1
            )

        # next, try to fill in any missing BV colors
        if np.all(self.BV == 0):
            self.BV *= np.nan
        if np.all(self.Bmag == 0):
            self.Bmag *= np.nan
        if np.any(np.isnan(self.BV)):
            inds = np.where(np.isnan(self.BV))[0]
            for i in inds:
                if not (np.isnan(self.Bmag[i])):
                    self.BV[i] = self.Bmag[i] - self.Vmag[i]
                else:
                    self.BV[i] = self.ms.SpTColor(
                        "B", "V", self.spectral_class[i][0], self.spectral_class[i][1]
                    )

        # we should now have all BV colors, so fill in missing Bmags from those
        if np.any(np.isnan(self.Bmag)):
            inds = np.isnan(self.Bmag)
            self.Bmag[inds] = self.BV[inds] + self.Vmag[inds]

        # next fix any missing luminosities
        if np.all(self.L == 0):
            self.L *= np.nan
        if np.any(np.isnan(self.L)) or np.any(self.L == 0):
            inds = np.where(np.isnan(self.L) | (self.L == 0))[0]
            for i in inds:
                self.L[i] = 10 ** self.ms.SpTOther(
                    "logL", self.spectral_class[i][0], self.spectral_class[i][1]
                )

        # and bolometric corrections
        if np.all(self.BC == 0):
            self.BC *= np.nan
        if np.any(np.isnan(self.BC)):
            inds = np.where(np.isnan(self.BC))[0]
            for i in inds:
                self.BC[i] = self.ms.SpTOther(
                    "BCv", self.spectral_class[i][0], self.spectral_class[i][1]
                )

        # and finally, get as many other bands as we can from table colors
        mag_atts = ["Kmag", "Hmag", "Jmag", "Imag", "Umag", "Rmag"]
        # these start-end colors to calculate for each band
        colors_to_add = [
            ["Ks", "V"],  # K = (K-V) + V
            ["H", "Ks"],  # H = (H-K) + K
            ["J", "H"],  # J = (J-H) + H
            ["Ic", "V"],  # I = (I-V) + V
            ["U", "B"],  # U = (U-B) + B
            ["Rc", "V"],  # R = (R-V) + V
        ]
        # and these are the known band values to add the colors to
        mags_to_add = [self.Vmag, self.Kmag, self.Hmag, self.Vmag, self.Bmag, self.Vmag]

        for mag_att, color_to_add, mag_to_add in zip(
            mag_atts, colors_to_add, mags_to_add
        ):
            mag = getattr(self, mag_att)
            if np.all(mag == 0):
                mag *= np.nan
            if np.any(np.isnan(mag)):
                inds = np.where(np.isnan(mag))[0]
                for i in inds:
                    mag[i] = (
                        self.ms.SpTColor(
                            color_to_add[0],
                            color_to_add[1],
                            self.spectral_class[i][0],
                            self.spectral_class[i][1],
                        )
                        + mag_to_add[i]
                    )

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
        """Filters out targets where required values are nan"""

        for att_name in self.required_catalog_atts:
            # treat sky coordinates differently form the other arrays
            if att_name == "coords":
                inds = (
                    ~np.isnan(self.coords.data.lon)
                    & ~np.isnan(self.coords.data.lat)
                    & ~np.isnan(self.coords.data.distance)
                )
            else:
                # all other attributes should be ndarrays or quantity ndarrays
                # in either case, they should have a dtype
                att = getattr(self, att_name)
                if np.issubdtype(att.dtype, np.number):
                    inds = ~np.isnan(att)
                elif np.issubdtype(att.dtype, str):
                    inds = att != ""
                else:
                    warnings.warn(
                        f"Cannot filter attribute {att_name} of type {att.dtype}"
                    )

            # only need to do something if there are any False in inds:
            if not (np.all(inds)):
                self.revise_lists(np.where(inds)[0])

    def binary_filter(self):
        """Removes stars which have attribute Binary_Cut == True"""

        i = np.where(~self.Binary_Cut)[0]
        self.revise_lists(i)

    def life_expectancy_filter(self):
        """Removes stars from Target List which have BV < 0.3"""

        i = np.where(self.BV > 0.3)[0]
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
                Integer indices of the stars to retain

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        if len(sInds) == 0:
            raise IndexError("Requested target revision would leave 0 stars.")

        for att in self.catalog_atts:
            if getattr(self, att).size != 0:
                setattr(self, att, getattr(self, att)[sInds])
        for key in self.F0dict:
            self.F0dict[key] = self.F0dict[key][sInds]
        try:
            self.Completeness.revise_updates(sInds)
        except AttributeError:
            pass
        self.nStars = len(sInds)

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

    def queryNEAsystems(self):
        """Queries NASA Exoplanet Archive system alias service to check for stars
        in the target list that have known planets.

        .. note::

            These queries take a *long* time, so rather than caching individual
            target lists, we'll keep everything we query and initially seed from a
            starting list that's part of the repository.
        """

        # define toggle for writing to disk
        systems_updated = False

        # grab cache from disk
        nea_file = Path(self.cachedir, "NASA_EXOPLANET_ARCHIVE_SYSTEMS.json")
        if not (nea_file.exists()):
            self.vprint("NASA Exoplanet Archive cache not found. Copying from default.")

            neacache = pkg_resources.resource_filename(
                "EXOSIMS.TargetList", "NASA_EXOPLANET_ARCHIVE_SYSTEMS.json.gz"
            )
            assert os.path.exists(neacache), (
                "NASA Exoplanet Archive default cache file not found in " f"{neacache}"
            )

            with gzip.open(neacache, "rb") as f:
                systems = json.loads(f.read())
                systems_updated = True
        else:
            self.vprint(f"Loading exoplanet archive cached systems from {nea_file}")
            with open(nea_file, "r") as ff:
                systems = json.loads(ff.read())

        # parse every name and alias in the list
        allaliases = []
        allaliaskeys = []
        for name in systems:
            for s in systems[name]["objects"]["stellar_set"]["stars"]:
                tmp = systems[name]["objects"]["stellar_set"]["stars"][s]["alias_set"][
                    "aliases"
                ]
                allaliases += tmp
                allaliaskeys += [name] * len(tmp)
        allaliases = np.array(allaliases)
        allaliaskeys = np.array(allaliaskeys)

        # find any missing ones to downloads
        missing = list(set(self.Name) - set(allaliases))
        if len(missing) > 0:
            systems_updated = True
            for n in tqdm(missing, desc="Querying NASA Exoplanet Archive Lookup"):
                dat = getExoplanetArchiveAliases(n)
                if dat:
                    systems[n] = dat

        # write results to disk if updated
        if systems_updated:
            with open(nea_file, "w") as outfile:
                json.dump(
                    systems,
                    outfile,
                    separators=(",", ":"),
                )
            self.vprint(f"Caching exoplanet archive systems to {nea_file}")

        # loop through systems data and create known planets boolean
        self.hasKnownPlanet = np.zeros(self.nStars, dtype=bool)
        self.targetAliases = np.ndarray((self.nStars), dtype=object)
        self.catalog_atts += ["hasKnownPlanet", "targetAliases"]
        for j, n in enumerate(self.Name):
            if (n in systems) or (n in allaliases):
                if n in systems:
                    name = n
                else:
                    name = allaliaskeys[allaliases == n][0]

                key = None
                for s in systems[name]["objects"]["stellar_set"]["stars"]:
                    if (
                        "requested_object"
                        in systems[name]["objects"]["stellar_set"]["stars"][s]
                    ):
                        key = s
                        break
                self.targetAliases[j] = systems[name]["objects"]["stellar_set"][
                    "stars"
                ][key]["alias_set"]["aliases"]
                self.hasKnownPlanet[j] = (
                    systems[name]["objects"]["planet_set"]["item_count"] > 0
                )

            else:
                self.targetAliases[j] = []

    def calc_intCutoff_dMag(self, mode, fname):
        """
        This calculates the delta magnitude for each target star that
        corresponds to the cutoff integration time. Uses the working
        angle, int_WA, used to calculate integration times.

        Args:
            mode (dict):
                Observing mode dictionary (see OpticalSystem)
            fname (str):
                Filename for caching results. Note that this should be just the base of
                the filename.  The full path to the cache directory (including the
                appropriate extension) is determined in this method.

        Returns:
            ~numpy.ndarray(float):
                Array with dMag values if exposed for the integration cutoff time
                for each target star
        """

        intCutoff_dMag_path = Path(self.cachedir, f"{fname}.intCutoff_dMag")
        if intCutoff_dMag_path.exists():
            self.vprint(f"Loaded intCutoff_dMag values from {intCutoff_dMag_path}")
            with open(intCutoff_dMag_path, "rb") as f:
                intCutoff_dMag = pickle.load(f)
        else:
            self.vprint("Calculating intCutoff_dMag")

            OS = self.OpticalSystem
            ZL = self.ZodiacalLight
            intTime = OS.intCutoff
            # Get the fZminglobal value in ZL for the desired mode
            fZminglobal = ZL.global_zodi_min(mode)

            # format inputs
            sInds = np.arange(self.nStars)
            intTimes = np.repeat(intTime.value, len(sInds)) * intTime.unit
            fZ = np.repeat(fZminglobal, len(sInds))
            fEZ = np.repeat(ZL.fEZ0, len(sInds))

            intCutoff_dMag = OS.calc_dMag_per_intTime(
                intTimes, self, sInds, fZ, fEZ, self.int_WA, mode
            ).reshape((len(intTimes),))
            with open(intCutoff_dMag_path, "wb") as f:
                pickle.dump(intCutoff_dMag, f)
            self.vprint(f"intCutoff_dMag values stored in {intCutoff_dMag_path}")
        return intCutoff_dMag

    def calc_saturation_dMag(self, mode, fname):
        """
        This calculates the delta magnitude for each target star that
        corresponds to an infinite integration time. Uses the working
        angle, int_WA, used to calculate integration times.

        Args:
            mode (dict):
                Observing mode dictionary (see :ref:`OpticalSystem`)
            fname (str):
                Filename for caching results. Note that this should be just the base of
                the filename.  The full path to the cache directory (including the
                appropriate extension) is determined in this method.

        Returns:
            ~numpy.ndarray(float):
                Array with dMag values if exposed for the integration cutoff time for
                each target star
        """
        saturation_dMag_path = Path(self.cachedir, f"{fname}.sat_dMag")
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
            if mode["syst"].get("occulter"):
                saturation_dMag = np.full(shape=len(sInds), fill_value=np.inf)
            else:
                for i, sInd in enumerate(
                    tqdm(sInds, desc="Calculating saturation_dMag")
                ):
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
                        OS.int_time_denom_obj,
                        args=args,
                        method="brentq",
                        bracket=[10, 40],
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
