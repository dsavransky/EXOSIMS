import copy
import gzip
import json
import os.path
import pickle
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import importlib.resources
from astropy.coordinates import SkyCoord
from astropy.time import Time
from MeanStars import MeanStars
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.exceptions import DisjointError, SynphotError
from synphot.models import BlackBodyNorm1D
from synphot.units import VEGAMAG
from tqdm import tqdm

from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.getExoplanetArchive import getExoplanetArchiveAliases
from EXOSIMS.util.utils import genHexStr
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util._numpy_compat import copy_if_needed


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
        fillMissingBandMags (bool):
            If ``fillPhotometry`` is True, also fill in missing band magnitudes.
            Ignored if ``fillPhotometry`` is False.  Defaults False.

            .. warning::

                This can be used for generating more complete target data for other uses
                but is *not* recommended for use in integration time calculations.

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
        popStars (list, optional):
            Remove given stars (by exact name matching) from target list.
            Defaults None.
        cherryPickStars (list, optional):
            Before doing any other filtering, filter out all stars from input star
            catalog *excep* for the ones in this list (by exact name matching).
            Defaults None (do not initially filter out any stars from the star catalog).
        skipSaturationCalcs (bool):
            If True, do not perform any new calculations for saturation dMag and
            saturation completeness (cached values will still be loaded if found on
            disk).  The saturation_dMag and saturation_comp will all be set to NaN if
            this keyword is set True and no cached values are found.  No cache will
            be written in that case. Defaults False.
        massLuminosityRelationship(str):
            String describing the mass-luminsoity relaitonship to use to populate
            stellar masses when not provided by the star catalog.
            Defaults to Henry1993.
            Allowable values: [Henry1993, Fernandes2021, Henry1993+1999, Fang2010]
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
        blackbody_spectra (numpy.ndarray):
            Storage array for blackbody spectra (populated as needed)
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
        cherryPickStars (list):
            List of star names to keep from input star catalog (all others are filtered
            out prior to any other filtering).
        Completeness (:ref:`Completeness`):
            :ref:`Completeness` object
        coords (astropy.coordinates.sky_coordinate.SkyCoord):
            Target coordinates
        diameter (astropy.units.quantity.Quantity):
            Stellar diameter in angular units.
        dist (astropy.units.quantity.Quantity):
            Target distances
        earths_only (bool):
            Used in upstream modules.  Alias for filter_for_char.
        explainFiltering (bool):
            Print informational messages at each target list filtering step.
        fillPhotometry (bool):
            Attempt to fill in missing target photometric  values using interpolants of
            tabulated values for the stellar type. See MeanStars documentation for
            more details.
        fillMissingBandMags (bool):
            If ``self.fillPhotometry`` is True, also fill in missing band magnitudes.
            Ignored if ``self.fillPhotometry`` is False.
        filter_for_char (bool):
            Use spectroscopy observation mode (rather than the default detection mode)
            for all calculations.
        filterBinaries (bool):
            Remove all binary stars or stars with known close companions from target
            list.
        filter_mode (dict):
            :ref:`OpticalSystem` observingMode dictionary. The observingMode used for
            target filtering.  Either the detection mode (default) or first
            characterization mode (if ``filter_for_char`` is True).
        getKnownPlanets (bool):
            Grab the list of known planets and target aliases from the NASA Exoplanet
            Archive
        hasKnownPlanet (numpy.ndarray):
            bool array indicating whether a target has known planets.  Only populated
            if attribute ``getKnownPlanets`` is True. Otherwise all entries are False.
        Hmag (numpy.ndarray):
            H band magnitudes
        Imag (numpy.ndarray):
            I band magnitudes
        int_comp (numpy.ndarray):
            Completeness value for each target star for default observation WA and
            :math:`\Delta{\\textrm{mag}}`.
        int_dMag (numpy.ndarray):
             :math:`\Delta{\\textrm{mag}}` used for default observation integration time
             calculation
        int_tmin (astropy.units.quantity.Quantity):
            Integration times corresponding to `int_dMag` with global minimum local zodi
            contribution.
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
        massLuminosityRealtionship (str):
            String describing the mass-luminosity relationship used to populate
            the stellar masses when not provided by the star catalog.
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
        systemOmega (astropy.units.quantity.Quantity):
            Base longitude of the ascending node for target system orbital planes
        OpticalSystem (:ref:`OpticalSystem`):
            :ref:`OpticalSystem` object
        optional_filters (dict):
            Dictionary of optional filters to apply to the target list.  Keys are the
            filter's name and values are the values necessary to apply the filter.
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
        skipSaturationCalcs (bool):
            If True (default), saturation dMag and saturation completeness are not
            computed. If cached values exist, they will be loaded, otherwise
            saturation_dMag and saturation_comp will all be set to NaN.  No new cache
            files will be written for these values.
        Spec (numpy.ndarray):
            Spectral type strings. Will be strictly in G0V format.
        specdict (dict):
            Dictionary of numerical mappings for spectral classes (O = 0, M = 6).
        spectral_catalog_index (dict):
            Dictionary mapping spectral type strings (keys) to template spectra files
            on disk (values).
        spectral_catalog_types (numpy.ndarray):
            nx4 ndarray (n is the number of template spectra avaiable). First three
            columns are spectral class (str), subclass (int), and luinosity class (str).
            The fourth column is a spectral class numeric representation, equaling
            ``specdict[specclass]*10 + subclass``.
        spectral_class (numpy.ndarray):
            nStars x 4 array.  Same column definitions as ``spectral_catalog_types`` but
            evaluated for the target stars rather than the template spectra.
        standard_bands (dict):
            Dictionary mapping UVBRIJHK (keys are single characters) to
            :py:class:`synphot.spectrum.SpectralElement` objects of the filter profiles.
        standard_bands_deltaLam (astropy.units.quantity.Quantity):
            Effective bandpasses of the profiles in `standard_bands`.
        standard_bands_lam (astropy.units.quantity.Quantity):
            Effective central wavelengths of the profiles in `standard_bands`.
        standard_bands_letters (str):
            Concatenation of the keys of  `standard_bands`.
        star_fluxes (dict):
            Internal storage of pre-computed star flux values that is populated
            each time a flux is requested for a particular target. Keyed by observing
            mode hex attribute.
        staticStars (bool):
            Do not apply proper motions to stars.  Stars always at mission start time
            positions.
        systemInclination (astropy.units.quantity.Quantity):
            Inclinations of target system orbital planes
        Teff (astropy.units.Quantity):
            Stellar effective temperature.
        template_spectra (dict):
            Dictionary of template spectra objects (populated as needed).
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
        fillMissingBandMags=False,
        explainFiltering=False,
        filterBinaries=True,
        cachedir=None,
        filter_for_char=False,
        earths_only=False,
        getKnownPlanets=False,
        int_WA=None,
        int_dMag=25,
        scaleWAdMag=False,
        popStars=None,
        cherryPickStars=None,
        skipSaturationCalcs=True,
        massLuminosityRelationship="Henry1993",
        optional_filters={},
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

        # assign inputs to attributes
        self.getKnownPlanets = bool(getKnownPlanets)
        self.staticStars = bool(staticStars)
        self.keepStarCatalog = bool(keepStarCatalog)
        self.fillPhotometry = bool(fillPhotometry)
        self.fillMissingBandMags = bool(fillMissingBandMags)
        self.explainFiltering = bool(explainFiltering)
        self.filterBinaries = bool(filterBinaries)
        self.filter_for_char = bool(filter_for_char)
        self.earths_only = bool(earths_only)
        self.scaleWAdMag = bool(scaleWAdMag)
        self.skipSaturationCalcs = bool(skipSaturationCalcs)
        self.massLuminosityRelationship = str(massLuminosityRelationship)
        allowable_massLuminosityRelationships = [
            "Henry1993",
            "Fernandes2021",
            "Henry1993+1999",
            "Fang2010",
        ]

        # Set up optional filters
        default_filters = {
            "outside_IWA_filter": {"enabled": True},
            "completeness_filter": {"enabled": True},
        }
        # Add the binary filter to the default optional filters
        if optional_filters.get("binary_filter"):
            # if binary_filter is provided in the optional_filters dict, we use that
            # value but raise a warning if it conflicts with the set filter value.
            if self.filterBinaries != optional_filters.get("enabled"):
                warnings.warn(
                    f"binary_filter is {optional_filters.get('enabled')} "
                    f"but filterBinaries is {self.filterBinaries}. "
                    f"Using binary_filter value of {optional_filters.get('enabled')}."
                )
        else:
            default_filters["binary_filter"] = {"enabled": self.filterBinaries}
        # Add the provided optional filters to the default filters, overriding
        # the defaults if necessary, and then save the combined dictionary as a
        # class attribute
        default_filters.update(optional_filters)
        self.optional_filters = default_filters

        assert (
            self.massLuminosityRelationship in allowable_massLuminosityRelationships
        ), (
            "massLuminosityRelationship must be one of: "
            f"{','.join(allowable_massLuminosityRelationships)}"
        )

        # list of target names to remove from targetlist
        if popStars is not None:
            assert isinstance(popStars, list), "popStars must be a list."
        self.popStars = popStars

        # list of target names to keep in targetlist
        if cherryPickStars is not None:
            assert isinstance(cherryPickStars, list), "cherryPickStars must be a list."
        self.cherryPickStars = cherryPickStars

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec"]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # set up stuff for spectral type conversion
        # Create a MeanStars object for future use
        self.ms = MeanStars()
        # Define a helper dictionary for spectral classes
        self.specdict = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}
        # figure out what template spectra we have access to
        self.load_spectral_catalog()
        # set up standard photometric bands
        self.load_standard_bands()
        # Create internal storage for speeding up spectral flux  calculations.
        # This dictionary is for storing SourceSpectrum objects (values) by spectral
        # types (keys). This will be populated as spectra are loaded.
        self.template_spectra = {}
        # This dictionary is for storing target-specific fluxes for observing mode
        # bands. keys are mod['hex'] values. values are arrays equal in size to the
        # current targetlist. This will be populated as calculations are performed.
        self.star_fluxes = {}

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

        # identify the observingMode to use for target filtering
        detmode = list(
            filter(
                lambda mode: mode["detectionMode"], self.OpticalSystem.observingModes
            )
        )[0]
        if self.filter_for_char or self.earths_only:
            mode = list(
                filter(
                    lambda mode: "spec" in mode["inst"]["name"],
                    self.OpticalSystem.observingModes,
                )
            )[0]
            self.calc_char_int_comp = True
        else:
            mode = detmode
            self.calc_char_int_comp = False
        self.filter_mode = mode

        # Define int_WA if None provided
        if int_WA is None:
            int_WA = (
                2.0 * self.filter_mode["IWA"]
                if np.isinf(self.filter_mode["OWA"])
                else (self.filter_mode["IWA"] + self.filter_mode["OWA"]) / 2.0
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
            keep = np.ones(self.nStars, dtype=bool)
            for n in self.popStars:
                keep[self.Name == n] = False

            self.revise_lists(np.where(keep)[0])

            if self.explainFiltering:
                print(
                    "%d targets remain after removing requested targets." % self.nStars
                )

        # if cherry-picking stars, filter out all the rest
        if self.cherryPickStars is not None:
            keep = np.zeros(self.nStars, dtype=bool)

            for n in self.cherryPickStars:
                keep[self.Name == n] = True

            self.revise_lists(np.where(keep)[0])

            if self.explainFiltering:
                print("%d targets remain after cherry-picking targets." % self.nStars)

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

        # compute stellar effective temperatures as needed
        self.stellar_Teff()
        # calculate 'true' and 'approximate' stellar masses and radii
        self.stellar_mass()
        self.stellar_diameter()
        # Calculate Star System Inclinations
        self.systemInclination = self.gen_inclinations(self.PlanetPopulation.Irange)

        # Calculate common Star System longitude of the ascending node
        self.systemOmega = self.gen_Omegas(self.PlanetPopulation.Orange)

        # create placeholder array black-body spectra
        # (only filled if any modes require it)
        self.blackbody_spectra = np.ndarray((self.nStars), dtype=object)
        self.catalog_atts.append("blackbody_spectra")

        # Apply optional filters that don't depend on completeness
        secondary_filter_names = ["completeness_filter"]
        first_filter_set = {
            key: self.optional_filters.get(key, {})
            for key in self.optional_filters.keys()
            if key not in secondary_filter_names
        }
        self.filter_target_list(first_filter_set)

        # Calculate saturation and intCutoff delta mags and completeness values
        self.calc_saturation_and_intCutoff_vals()

        # populate completeness values
        self.int_comp = self.Completeness.target_completeness(self)
        self.catalog_atts.append("int_comp")

        # generate any completeness update data needed
        self.Completeness.gen_update(self)

        # apply any requested additional filters that depend on completeness
        second_filter_set = {
            key: self.optional_filters.get(key, {}) for key in secondary_filter_names
        }
        self.filter_target_list(second_filter_set)

        # if we're doing filter_for_char, then that means that we haven't computed the
        # star fluxes for the detection mode.  Let's do that now (post-filtering to
        # limit the number of calculations
        if self.filter_for_char or self.earths_only:
            fname = (
                f"TargetList_{self.StarCatalog.__class__.__name__}_"
                f"nStars_{self.nStars}_mode_{detmode['hex']}.star_fluxes"
            )
            star_flux_path = Path(self.cachedir, fname)
            if star_flux_path.exists():
                with open(star_flux_path, "rb") as f:
                    self.star_fluxes[detmode["hex"]] = pickle.load(f)
                self.vprint(f"Loaded star fluxes values from {star_flux_path}")
            else:
                _ = self.starFlux(np.arange(self.nStars), detmode)
                with open(star_flux_path, "wb") as f:
                    pickle.dump(self.star_fluxes[detmode["hex"]], f)
                    self.vprint(f"Star fluxes stored in {star_flux_path}")

            # remove any zero-flux vals
            if np.any(self.star_fluxes[detmode["hex"]].value == 0):
                keepinds = np.where(self.star_fluxes[detmode["hex"]].value != 0)[0]
                self.revise_lists(keepinds)
                if self.explainFiltering:
                    print(
                        (
                            "{} targets remain after removing those with zero flux. "
                        ).format(self.nStars)
                    )

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
                ), c2=self.starprop(allInds, missionStart, eclip=True): (
                    c1[sInds] if not (eclip) else c2[sInds]  # noqa: E275
                )
            )

    def __str__(self):
        """String representation of the Target List object

        When the command 'print' is used on the Target List object, this method
        will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Target List class object attributes"

    def load_spectral_catalog(self):
        """Helper method for generating a cache of available template spectra and
        loading them as attributes

        Creates the following attributes:

        #. ``spectral_catalog_index``: A dictionary of spectral types (keys) and the
           associated spectra files on disk (values)
        #. ``spectral_catalog_types``: An nx4 ndarray (n is the number of teplate
           spectra avaiable). First three columns are spectral class (str),
           subclass (int), and luinosity class (str). The fourth column is a spectral
           class numeric representation, equaling specdict[specclass]*10 + subclass.

        """
        spectral_catalog_cache = Path(self.cachedir, "spectral_catalog.pkl")
        if spectral_catalog_cache.exists():
            with open(spectral_catalog_cache, "rb") as f:
                tmp = pickle.load(f)
                self.spectral_catalog_index = tmp["spectral_catalog_index"]
                self.spectral_catalog_types = tmp["spectral_catalog_types"]
        else:
            # Find data locations on disk and ensure that they're there
            pickles_path = os.path.join(
                importlib.resources.files("EXOSIMS.TargetList"), "dat_uvk"
            )

            bpgs_path = os.path.join(
                importlib.resources.files("EXOSIMS.TargetList"), "bpgs"
            )

            spectral_catalog_file = os.path.join(
                importlib.resources.files("EXOSIMS.TargetList"),
                "spectral_catalog_index.json",
            )

            assert os.path.isdir(
                pickles_path
            ), f"Pickles Atlas path {pickles_path} does not appear to be a directory."
            assert os.path.isdir(
                bpgs_path
            ), f"BPGS Atlas path {bpgs_path} does not appear to be a directory."
            assert os.path.exists(
                spectral_catalog_file
            ), f"Spectral catalog index file {spectral_catalog_file} not found."

            # grab original spectral catalog index
            with open(spectral_catalog_file, "r") as f:
                spectral_catalog = json.load(f)

            # assign system-specific paths and repackage catalog info for easier
            # accessiblity downstream
            spectral_catalog_index = {}
            spectral_catalog_types = np.zeros((len(spectral_catalog), 4), dtype=object)

            for j, s in enumerate(spectral_catalog):
                if spectral_catalog[s]["file"].startswith("pickles"):
                    spectral_catalog_index[s] = os.path.join(
                        pickles_path, spectral_catalog[s]["file"]
                    )
                else:
                    spectral_catalog_index[s] = os.path.join(
                        bpgs_path, spectral_catalog[s]["file"]
                    )
                assert os.path.exists(spectral_catalog_index[s])

                spectral_catalog_types[j] = spectral_catalog[s]["specclass"]

            # cache the system-specific values for future use
            with open(spectral_catalog_cache, "wb") as f:
                pickle.dump(
                    {
                        "spectral_catalog_index": spectral_catalog_index,
                        "spectral_catalog_types": spectral_catalog_types,
                    },
                    f,
                )

            # now assign catalog values as class attributes
            self.spectral_catalog_index = spectral_catalog_index
            self.spectral_catalog_types = spectral_catalog_types

    def get_template_spectrum(self, spec):
        """Helper method for loading/retrieving spectra from the spectral catalog

        Args:
            spec (str):
                Spectral type string. Must be a keys in self.spectral_catalog_index
        Returns:
            synphot.SourceSpectrum:
                Template pectrum from file.
        """

        if spec not in self.template_spectra:
            self.template_spectra[spec] = SourceSpectrum.from_file(
                self.spectral_catalog_index[spec]
            )

        return self.template_spectra[spec]

    def load_standard_bands(self):
        """Helper method that defines standard photometric bandpasses

        This method defines the following class attributes:

        #. ``standard_bands_letters``: String with standard band letters
           (nominally UVBRI)
        #. ``standard_bands``: A dictionary (key of band letter) whose values are
           synphot SpectralElement objects for that bandpass.
        #. ``standard_bands_lam``: An array of band central wavelengths (same order as
           standard_bands_letters
        #. ``standard_bands_deltaLam``: An array of band bandwidths (same order as
           standard_bands_letters.

        """

        band_letters = "UBVRIJHK"
        band_file_names = [
            "johnson_u",
            "johnson_b",
            "johnson_v",
            "cousins_r",
            "cousins_i",
            "bessel_j",
            "bessel_h",
            "bessel_k",
        ]
        self.standard_bands = {}
        for b, bf in zip(band_letters, band_file_names):
            self.standard_bands[b] = SpectralElement.from_filter(bf)

        self.standard_bands_lam = (
            np.array(
                [
                    self.standard_bands[b].avgwave().to(u.nm).value
                    for b in self.standard_bands
                ]
            )
            * u.nm
        )
        self.standard_bands_deltaLam = (
            np.array(
                [
                    self.standard_bands[b].rectwidth().to(u.nm).value
                    for b in self.standard_bands
                ]
            )
            * u.nm
        )
        self.standard_bands_letters = band_letters

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

        # required catalog attributes for the Prototype
        self.required_catalog_atts = [
            "Name",
            "Vmag",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "dist",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
            "Spec",
            "parx",
        ]

        # generate list of possible Star Catalog attributes. If the StarCatalog provides
        # the list, use that.
        if hasattr(self.StarCatalog, "catalog_atts"):
            self.catalog_atts = copy.copy(self.StarCatalog.catalog_atts)
        else:
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

        # ensure that the catalog will provide our required attributes
        tmp = list(set(self.required_catalog_atts) - set(self.catalog_atts))
        assert len(tmp) == 0, (
            f"Star catalog {self.StarCatalog.__class__.__name__} "
            "does not provide required attribute(s): "
            f"{' ,'.join(tmp)}"
        )

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
                if isinstance(getattr(SC, att), np.ma.core.MaskedArray):
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
        # also populate inputs for calculations
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPop = self.PlanetPopulation
        Comp = self.Completeness

        # grab zodi vals for any required calculations
        sInds = np.arange(self.nStars)
        fZminglobal = ZL.global_zodi_min(self.filter_mode)
        fZ = np.repeat(fZminglobal, len(sInds))
        fEZ = (
            ZL.zodi_color_correction_factor(self.filter_mode["lam"], photon_units=True)
            * ZL.fEZ0
            * 10.0 ** (-0.4 * (self.MV - 4.83))
        )

        # compute proj separation bounds for any required calculations
        if PPop.scaleOrbits:
            tmp_smin = np.tan(self.filter_mode["IWA"]) * self.dist / np.sqrt(self.L)
            if np.isinf(self.filter_mode["OWA"]):
                tmp_smax = np.inf * self.dist
            else:
                tmp_smax = np.tan(self.filter_mode["OWA"]) * self.dist / np.sqrt(self.L)
        else:
            tmp_smin = np.tan(self.filter_mode["IWA"]) * self.dist
            if np.isinf(self.filter_mode["OWA"]):
                tmp_smax = np.inf * self.dist
            else:
                tmp_smax = np.tan(self.filter_mode["OWA"]) * self.dist

        # 0. Regardless of whatever else we do, we're going to need stellar fluxes in
        # the relevant observing mode.  So let's just compute them now and cache them
        # for later use.
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"nStars_{self.nStars}_mode_{self.filter_mode['hex']}.star_fluxes"
        )
        star_flux_path = Path(self.cachedir, fname)
        if star_flux_path.exists():
            with open(star_flux_path, "rb") as f:
                self.star_fluxes = pickle.load(f)
            self.vprint(f"Loaded star fluxes values from {star_flux_path}")
        else:
            _ = self.starFlux(np.arange(self.nStars), self.filter_mode)
            with open(star_flux_path, "wb") as f:
                pickle.dump(self.star_fluxes, f)
                self.vprint(f"Star fluxes stored in {star_flux_path}")

        # remove any zero-flux vals
        if np.any(self.star_fluxes[self.filter_mode["hex"]].value == 0):
            keepinds = np.where(self.star_fluxes[self.filter_mode["hex"]].value != 0)[0]
            self.revise_lists(keepinds)
            sInds = np.arange(self.nStars)
            tmp_smin = tmp_smin[keepinds]
            tmp_smax = tmp_smax[keepinds]
            fZ = fZ[keepinds]
            fEZ = fEZ[keepinds]
            if self.explainFiltering:
                print(
                    ("{} targets remain after removing those with zero flux. ").format(
                        self.nStars
                    )
                )

        # 1. Calculate the saturation dMag. This is stricly a function of
        # fZminglobal, ZL.fEZ0, self.int_WA, mode, the current targetlist
        # and the postprocessing factor
        zodi_vals_str = f"{str(ZL.global_zodi_min(self.filter_mode))} {str(ZL.fEZ0)}"
        stars_str = (
            f"ppFact:{self.PostProcessing._outspec['ppFact']}, "
            f"fillPhotometry:{self.fillPhotometry}, "
            f"fillMissingBandMags:{self.fillMissingBandMags}"
            ",".join(self.Name)
        )
        int_WA_str = ",".join(self.int_WA.value.astype(str)) + str(self.int_WA.unit)

        # cache filename is the three class names, the vals hash, and the mode hash
        vals_hash = genHexStr(zodi_vals_str + stars_str + int_WA_str)
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"{OS.__class__.__name__}_{ZL.__class__.__name__}_"
            f"vals_{vals_hash}_mode_{self.filter_mode['hex']}"
        )

        saturation_dMag_path = Path(self.cachedir, f"{fname}.sat_dMag")
        if saturation_dMag_path.exists():
            with open(saturation_dMag_path, "rb") as f:
                self.saturation_dMag = pickle.load(f)
            self.vprint(f"Loaded saturation_dMag values from {saturation_dMag_path}")
        else:
            if self.skipSaturationCalcs:
                self.saturation_dMag = np.zeros(self.nStars) * np.nan
            else:
                self.saturation_dMag = OS.calc_saturation_dMag(
                    self, sInds, fZ, fEZ, self.int_WA, self.filter_mode, TK=None
                )

                with open(saturation_dMag_path, "wb") as f:
                    pickle.dump(self.saturation_dMag, f)
                self.vprint(f"saturation_dMag values stored in {saturation_dMag_path}")

        # 2. Calculate the completeness value if the star is integrated for an
        # infinite time by using the saturation dMag
        if PPop.scaleOrbits:
            tmp_dMag = self.saturation_dMag - 2.5 * np.log10(self.L)
        else:
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
            with open(saturation_comp_path, "rb") as f:
                self.saturation_comp = pickle.load(f)
            self.vprint(f"Loaded saturation_comp values from {saturation_comp_path}")
        else:
            if self.skipSaturationCalcs:
                self.saturation_comp = np.zeros(self.nStars) * np.nan
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
            f"vals_{vals_hash}_mode_{self.filter_mode['hex']}"
        )

        intCutoff_dMag_path = Path(self.cachedir, f"{fname}.intCutoff_dMag")
        if intCutoff_dMag_path.exists():
            with open(intCutoff_dMag_path, "rb") as f:
                self.intCutoff_dMag = pickle.load(f)
            self.vprint(f"Loaded intCutoff_dMag values from {intCutoff_dMag_path}")
        else:
            self.vprint("Calculating intCutoff_dMag")
            intTimes = np.repeat(OS.intCutoff.value, len(sInds)) * OS.intCutoff.unit

            self.intCutoff_dMag = OS.calc_dMag_per_intTime(
                intTimes, self, sInds, fZ, fEZ, self.int_WA, self.filter_mode
            ).reshape((len(intTimes),))
            with open(intCutoff_dMag_path, "wb") as f:
                pickle.dump(self.intCutoff_dMag, f)
            self.vprint(f"intCutoff_dMag values stored in {intCutoff_dMag_path}")

        # 4. Calculate intCutoff completeness. This is a function of the exact same
        # things as the previous calculation, so we can recycle the filename
        if PPop.scaleOrbits:
            tmp_dMag = self.intCutoff_dMag - 2.5 * np.log10(self.L)
        else:
            tmp_dMag = self.intCutoff_dMag

        # cache filename is the two class names and the vals hash
        intcutoffcomp_valstr = (
            ",".join(tmp_smin.to(u.AU).value.astype(str))
            + ",".join(tmp_smax.to(u.AU).value.astype(str))
            + ",".join(tmp_dMag.astype(str))
        )

        vals_hash = genHexStr(stars_str + intcutoffcomp_valstr)
        fname = (
            f"TargetList_{self.StarCatalog.__class__.__name__}_"
            f"{Comp.__class__.__name__}_vals_{vals_hash}"
        )

        intCutoff_comp_path = Path(self.cachedir, f"{fname}.intCutoff_comp")
        if intCutoff_comp_path.exists():
            with open(intCutoff_comp_path, "rb") as f:
                self.intCutoff_comp = pickle.load(f)
            self.vprint(f"Loaded intCutoff_comp values from {intCutoff_comp_path}")
        else:
            self.vprint("Calculating the integration cutoff time completeness")
            self.intCutoff_comp = Comp.comp_calc(
                tmp_smin.to(u.AU).value, tmp_smax.to(u.AU).value, tmp_dMag
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
            self.int_WA[np.where(self.int_WA > self.filter_mode["OWA"])[0]] = (
                self.filter_mode["OWA"] * (1.0 - 1e-14)
            )
            self.int_WA[np.where(self.int_WA < self.filter_mode["IWA"])[0]] = (
                self.filter_mode["IWA"] * (1.0 + 1e-14)
            )
            self.int_dMag = self.int_dMag + 2.5 * np.log10(self.L)

        # Go through the int_dMag values and replace with limiting dMag where
        # int_dMag is higher. Since the int_dMag will never be reached if
        # intCutoff_dMag is below it
        for i, int_dMag_val in enumerate(self.int_dMag):
            if int_dMag_val > self.intCutoff_dMag[i]:
                self.int_dMag[i] = self.intCutoff_dMag[i]

        # Finally, compute the nominal integration time at minimum zodi
        self.int_tmin = self.OpticalSystem.calc_intTime(
            self, sInds, fZ, fEZ, self.int_dMag, self.int_WA, self.filter_mode
        )

        # update catalog attributes for any future filtering
        self.catalog_atts.append("intCutoff_dMag")
        self.catalog_atts.append("intCutoff_comp")
        self.catalog_atts.append("saturation_dMag")
        self.catalog_atts.append("saturation_comp")
        self.catalog_atts.append("int_tmin")

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

        TODO: only use MeanStars for dwarfs. Otherwise use spectra.

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
                    # otherwise try for B-V color
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
            [
                j
                for j in range(self.nStars)
                if (
                    (self.spectral_class[j, 0] in "OBAFGKM")
                    and (self.spectral_class[j, 2] not in ["VI", "VII"])
                )
            ]
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

        # Add fourth column to spectral_class with the numerical class value
        # defined as specdict[specclass] * 10 + specsubclass
        spectypenum = [
            self.specdict[c] * 10 + sc
            for c, sc in zip(self.spectral_class[:, 0], self.spectral_class[:, 1])
        ]
        self.spectral_class = np.hstack(
            (self.spectral_class, np.array(spectypenum, ndmin=2).transpose())
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

        # if we don't need to fill band mag values, we're done here
        if not (self.fillMissingBandMags):
            return

        # finally, get as many other bands as we can from table colors
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

    def filter_target_list(self, filters):
        """This function is responsible for filtering by any required metrics.

        This should be used for *optional* filters. The ones in the prototype
        are:
            binary_filter
            outside_IWA_filter
            life_expectancy_filter
            main_sequence_filter
            fgk_filter
            vis_mag_filter (takes Vmagcrit as input)
            max_dmag_filter
            completeness_filter (Has to be run after the completeness values are calculated)
            vmag_filter (takes vmag_range as input)
            ang_diam_filter

        Args:
            filters (dict):
                Dictionary of filters to apply to the target list.  Keys are
                filter names, and values are the filter functions to apply.
                Looks like:
                filters = {
                    "binary_filter": {"enabled": True},
                    "outside_IWA_filter": {"enabled": True},
                    vmag_filter: {"enabled": True, "params": {"vmag_range": [4, 10]}},
                }
        """
        for filter_name, filter_config in filters.items():
            # check if the filter is enabled
            if filter_config.get("enabled", False):
                # get the filter function
                if hasattr(self, filter_name):
                    # Get the filter method
                    filter_func = getattr(self, filter_name)
                    # Apply the filter
                    filter_func(**filter_config.get("params", {}))
                    if self.explainFiltering:
                        print(
                            f"{self.nStars} targets remain after {filter_name.replace('_', ' ')}."
                        )
                else:
                    raise AttributeError(
                        (f"No filter '{filter_name}' in {self.__class__.__name__}.")
                    )

    def vmag_filter(self, vmag_range):
        """Removes stars with Vmag outside of specified range

        Args:
            vmag_range (list):
                2-element list of min, max Vmag values

        """
        meets_lower_bound = self.Vmag > vmag_range[0]
        meets_upper_bound = self.Vmag < vmag_range[1]
        i = np.where(meets_lower_bound & meets_upper_bound)[0]
        self.revise_lists(i)

    def ang_diam_filter(self):
        """Remove stars which are larger than the IWA."""
        # angular radius of each star
        ang_rad = self.diameter / 2
        IWA = self.OpticalSystem.IWA
        # indices of stars with angular radius less than IWA
        i = np.where(ang_rad < IWA)[0]
        self.revise_lists(i)

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
                elif att.dtype == bool:
                    continue
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
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        if len(sInds) == 0:
            raise IndexError("Requested target revision would leave 0 stars.")

        for att in self.catalog_atts:
            if getattr(self, att).size != 0:
                setattr(self, att, getattr(self, att)[sInds])
        for key in self.star_fluxes:
            self.star_fluxes[key] = self.star_fluxes[key][sInds]
        try:
            self.Completeness.revise_updates(sInds)
        except AttributeError:
            pass
        self.nStars = len(sInds)

    def stellar_diameter(self):
        """Populates target list with approximate stellar diameters

        Stellar radii are computed using the BV target colors according to the model
        from [Boyajian2014]_.  This model has a standard deviation error of 7.8%.

        Updates/creates attribute ``diameter``, as needed.

        """

        # if any diameters were populated from the star catalog, do not
        # overwrite those
        if hasattr(self, "diameter"):
            sInds = np.where(np.isnan(self.diameter) | (self.diameter.value == 0))[0]
        else:
            sInds = np.arange(self.nStars)
            self.diameter = np.zeros(self.nStars) * u.mas

        if "diameter" not in self.catalog_atts:
            self.catalog_atts.append("diameter")

        # B-V polynomial fit coefficients:
        coeffs = [0.49612, 1.11136, -1.18694, 0.91974, -0.19526]

        # Evaluate eq. 2 using B-V color
        logth_zero = np.zeros(sInds.shape)
        for j, ai in enumerate(coeffs):
            logth_zero += ai * self.BV[sInds] ** j

        # now invert eq. 1 to get the actual diameter in mas
        self.diameter[sInds] = 10 ** (logth_zero - 0.2 * self.Vmag[sInds]) * u.mas

    def stellar_Teff(self):
        """Calculate the effective stellar temperature based on B-V color.

        This method uses the empirical fit from [Ballesteros2012]_
        doi:10.1209/0295-5075/97/34008

        Updates/creates attribute ``Teff``, as needed.
        """

        # if any effective temperatures were populated from the star catalog, do not
        # overwrite those
        if hasattr(self, "Teff"):
            sInds = np.where(np.isnan(self.Teff) | (self.Teff.value == 0))[0]
        else:
            sInds = np.arange(self.nStars)
            self.Teff = np.zeros(self.nStars) * u.K

        if "Teff" not in self.catalog_atts:
            self.catalog_atts.append("Teff")

        self.Teff[sInds] = (
            4600.0
            * u.K
            * (
                1.0 / (0.92 * self.BV[sInds] + 1.7)
                + 1.0 / (0.92 * self.BV[sInds] + 0.62)
            )
        )

    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses

        Approximate stellar masses are calculated from absolute magnitudes using the
        model from [Henry1993]_. "True" masses are generated by a uniformly
        distributed, 7%-mean error term to the apprxoimate masses.

        All values are in units of solar mass.

        Function called by reset sim.

        """

        if self.massLuminosityRelationship == "Henry1993":
            # good generalist, but out of date
            # 'approximate' stellar mass
            self.MsEst = (
                10.0 ** (0.002456 * self.MV**2 - 0.09711 * self.MV + 0.4365)
            ) * u.solMass
            # normally distributed 'error' of 7%
            err = (np.random.random(len(self.MV)) * 2.0 - 1.0) * 0.07
            self.MsTrue = (1.0 + err) * self.MsEst

        elif self.massLuminosityRelationship == "Fernandes2021":
            # only good for FGK
            # 'approximate' stellar mass without error
            self.MsEst = (
                10
                ** (
                    (0.219 * np.log10(self.L))
                    + (0.063 * ((np.log10(self.L)) ** 2))
                    - (0.119 * ((np.log10(self.L)) ** 3))
                )
            ) * u.solMass
            # error distribution in literature as 3% in approxoimate masses
            err = (np.random.random(len(self.L)) * 2.0 - 1.0) * 0.03
            self.MsTrue = (1.0 + err) * self.MsEst

        elif self.massLuminosityRelationship == "Henry1993+1999":
            # more specific than Henry1993
            # initialize MsEst attribute
            self.MsEst = np.zeros(self.nStars)
            for j, MV in enumerate(self.MV):
                if 0.50 <= MV <= 2.0:
                    mass = (10.0 ** (0.002456 * MV**2 - 0.09711 * MV + 0.4365)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.07
                elif 0.18 <= MV < 0.50:
                    mass = (10.0 ** (-0.1681 * MV + 1.4217)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.07
                elif 0.08 <= MV < 0.18:
                    mass = (10 ** (0.005239 * MV**2 - 0.2326 * MV + 1.3785)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    # 5% error desccribed in 1999 paper
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.05
                else:
                    # default to Henry 1993
                    mass = (10.0 ** (0.002456 * MV**2 - 0.09711 * MV + 0.4365)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.07
                self.MsEst[j] = mass
            self.MsEst = self.MsEst * u.solMass
            self.MsTrue = (1.0 + err) * self.MsEst

        elif self.massLuminosityRelationship == "Fang2010":
            # for all main sequence stars, good generalist
            self.MsEst = np.zeros(self.nStars)
            for j, MV in enumerate(self.MV):
                if MV <= 1.05:
                    mass = (10 ** (0.558 - 0.182 * MV - 0.0028 * MV**2)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.05
                else:
                    mass = (10 ** (0.489 - 0.125 * MV + 0.00511 * MV**2)).item()
                    self.MsEst = np.append(self.MsEst, mass)
                    err = (np.random.random(1) * 2.0 - 1.0) * 0.07
                self.MsEst[j] = mass
            self.MsEst = self.MsEst * u.solMass
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
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

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

    def starFlux(self, sInds, mode):
        """Return the total spectral flux of the requested stars for the
        given observing mode.  Caches results internally for faster access in
        subsequent calls.

        Args:
            sInds (~numpy.ndarray(int)):
                Indices of the stars of interest
            mode (dict):
                Observing mode dictionary (see :ref:`OpticalSystem`)

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Spectral fluxes in units of ph/m**2/s.

        """

        # If we've never been asked for fluxes in this mode before, create a new array
        # of flux values for it and set them all to nan.
        if mode["hex"] not in self.star_fluxes:
            self.star_fluxes[mode["hex"]] = np.full(self.nStars, np.nan) * (
                u.ph / u.s / u.m**2
            )

        # figure out which target indices (if any) need new calculations to be done
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        novals = np.isnan(self.star_fluxes[mode["hex"]][sInds])
        inds = np.unique(sInds[novals])  # calculations needed for these sInds
        if len(inds) > 0:
            # find out distances (in wavelength) between standard bands and mode
            band_dists = np.abs(self.standard_bands_lam - mode["lam"]).to(u.nm).value
            # this is our order of preferred band manigutdes to use
            band_pref_inds = np.argsort(band_dists)

            # now loop through all required calculations and pick the best approach
            # for each one
            for sInd in tqdm(inds, "Computing star fluxes", delay=2):
                # try each band in descending preference order until you get a valid
                # magnitude value
                for band_ind in band_pref_inds:
                    mag_to_use = None
                    tmp = getattr(self, f"{self.standard_bands_letters[band_ind]}mag")
                    if np.all(tmp == 0):
                        continue
                    if not (np.isnan(tmp[sInd])):
                        band_to_use = self.standard_bands_letters[band_ind]
                        mag_to_use = tmp[sInd]
                        break
                assert (
                    mag_to_use is not None
                ), f"No valid magnitudes found for {self.Name[sInd]}"

                # if bandpass goes beyond 2.4 microns, use black-body spectrum
                if mode["bandpass"].waveset.max() > 2.4 * u.um:
                    if self.blackbody_spectra[sInd] is None:
                        self.blackbody_spectra[sInd] = SourceSpectrum(
                            BlackBodyNorm1D, temperature=self.Teff[sInd]
                        )
                    template = self.blackbody_spectra[sInd]
                else:
                    # find the closest template spectrum
                    if self.Spec[sInd] in self.spectral_catalog_index:
                        spec_to_use = self.Spec[sInd]
                    else:
                        # match closest row within the same luminosity class, if we have
                        # templates for it
                        tmp = self.spectral_catalog_types[
                            self.spectral_catalog_types[:, 2]
                            == self.spectral_class[sInd][2]
                        ]
                        if len(tmp) == 0:
                            tmp = self.spectral_catalog_types

                        row = tmp[
                            np.argmin(np.abs(tmp[:, 3] - self.spectral_class[sInd][3]))
                        ]

                        spec_to_use = f"{row[0]}{row[1]}{row[2]}"

                    # load the template
                    template = self.get_template_spectrum(spec_to_use)

                # renormalize the template to the band we've decided to use
                try:
                    template_renorm = template.normalize(
                        mag_to_use * VEGAMAG,
                        self.standard_bands[band_to_use],
                        vegaspec=self.OpticalSystem.vega_spectrum,
                    )

                    # finally, write the result back to the star_fluxes
                    self.star_fluxes[mode["hex"]][sInd] = Observation(
                        template_renorm, mode["bandpass"], force="taper"
                    ).integrate()
                except (DisjointError, SynphotError):
                    self.star_fluxes[mode["hex"]][sInd] = 0 * (u.ph / u.s / u.m**2)

        return self.star_fluxes[mode["hex"]][sInds]

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

        M = self.MsEst[sInds].value  # use MsEst as that's the deterministic one
        a = -0.073
        b = 0.668
        starRadius = 10 ** (a + b * np.log10(M))

        return starRadius * u.R_sun

    def gen_inclinations(self, Irange):
        """Randomly Generate Inclination of Target System Orbital Plane for
        all stars in the target list

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

    def gen_Omegas(self, Orange):
        """Randomly Generate longitude of the ascending node of target system
        orbital planes for all stars in the target list

        Args:
            Orange (~numpy.ndarray(float)):
                The range to generate Omegas over

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                System Omegas
        """
        return (
            np.random.uniform(
                low=Orange[0].to(u.deg).value,
                high=Orange[1].to(u.deg).value,
                size=self.nStars,
            )
            * u.deg
        )

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
            **kwargs (any):
                Extra keyword arguments

        Returns:
            Quantity array:
                separation from the star of the IWA in AU
        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
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
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        T_eff = self.Teff[sInds]

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
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

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

            neacache = os.path.join(
                importlib.resources.files("EXOSIMS.TargetList"),
                "NASA_EXOPLANET_ARCHIVE_SYSTEMS.json.gz",
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
