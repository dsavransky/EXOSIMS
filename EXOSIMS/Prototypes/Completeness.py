# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
from EXOSIMS.util._numpy_compat import copy_if_needed


class Completeness(object):
    """:ref:`Completeness` Prototype

    Args:
        minComp (float):
            Minimum completeness for target filtering. Defaults to 0.1.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        minComp (float):
            Minimum completeness value for inclusion in target list
        PlanetPhysicalModel (:ref:`PlanetPhysicalModel`):
            Planet physical model object
        PlanetPopulation (:ref:`PlanetPopulation`):
            Planet population object
        updates (numpy.ndarray):
            Dynamic completeness updates array for revisists.
    """

    _modtype = "Completeness"

    def __init__(self, minComp=0.1, cachedir=None, **specs):

        # start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # find the cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # if specs contains a completeness_spec then we are going to generate separate
        # instances of planet population and planet physical model for completeness and
        # for the rest of the sim
        if "completeness_specs" in specs:
            if specs["completeness_specs"] is None:
                specs["completeness_specs"] = {}
                specs["completeness_specs"]["modules"] = {}
            if "modules" not in specs["completeness_specs"]:
                specs["completeness_specs"]["modules"] = {}
            if "PlanetPhysicalModel" not in specs["completeness_specs"]["modules"]:
                specs["completeness_specs"]["modules"]["PlanetPhysicalModel"] = specs[
                    "modules"
                ]["PlanetPhysicalModel"]
            if "PlanetPopulation" not in specs["completeness_specs"]["modules"]:
                specs["completeness_specs"]["modules"]["PlanetPopulation"] = specs[
                    "modules"
                ]["PlanetPopulation"]
            self.PlanetPopulation = get_module(
                specs["completeness_specs"]["modules"]["PlanetPopulation"],
                "PlanetPopulation",
            )(**specs["completeness_specs"])
            self._outspec["completeness_specs"] = specs.get("completeness_specs")
        else:
            self.PlanetPopulation = get_module(
                specs["modules"]["PlanetPopulation"], "PlanetPopulation"
            )(**specs)

        # assign phyiscal model object to attribute
        self.PlanetPhysicalModel = self.PlanetPopulation.PlanetPhysicalModel

        # set minimum completeness
        self.minComp = float(minComp)
        self._outspec["minComp"] = self.minComp

        # generate filenames for cached products (if any)
        self.generate_cache_names(**specs)

        # perform prelininary calcualtions (if any)
        self.completeness_setup()

    def generate_cache_names(self, **specs):
        """Generate unique filenames for cached products"""

        self.filename = "Completeness"
        self.dfilename = "DynamicCompleteness"

    def completeness_setup(self):
        """Preform any preliminary calculations needed for this flavor of completeness

        For the Prototype, this is just a dummy function for later overloading
        """

        pass

    def __str__(self):
        """String representation of Completeness object

        When the command 'print' is used on the Completeness object, this
        method will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Completeness class object attributes"

    def target_completeness(self, TL):
        """Generates completeness values for target stars

        This method is called from TargetList __init__ method.

        Args:
            TL (:ref:`TargetList`):
                TargetList object

        Returns:
            ~numpy.ndarray(float):
                Completeness values for each target star

        .. warning::

            The prototype implementation does not perform any real completeness
            calculations.  To be used when you need a completeness object but do not
            care about the actual values.

        """

        int_comp = np.array([0.2] * TL.nStars)

        return int_comp

    def gen_update(self, TL):
        """Generates any information necessary for dynamic completeness
        calculations (completeness at successive observations of a star in the
        target list)

        Args:
            TL (:ref:`TargetList`):
                TargetList object

        Returns:
            None

        """
        # Prototype does not use precomputed updates, so set these to zeros
        self.updates = np.zeros((TL.nStars, 5))

    def completeness_update(self, TL, sInds, visits, dt):
        """Updates completeness value for stars previously observed

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Indices of stars to update
            visits (~numpy.ndarray(int)):
                Number of visits for each star
            dt (~astropy.units.Quantity(~numpy.ndarray(float))):
                Time since previous observation

        Returns:
            ~numpy.ndarray(float):
                Completeness values for each star

        """
        # prototype returns the single-visit completeness value
        int_comp = TL.int_comp[sInds]

        return int_comp

    def revise_updates(self, ind):
        """Keeps completeness update values only for targets remaining in
        target list during filtering (called from TargetList.filter_target_list)

        Args:
            ind (~numpy.ndarray(int)):
                array of indices to keep

        """

        self.updates = self.updates[ind, :]

    def comp_per_intTime(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Calculates completeness values per integration time

        Note: Prototype does no calculations and always returns the same value

        Args:
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times
            TL (:ref:`TargetList`):
                TargetList object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (~astropy.units.Quantity(~numpy.ndarray(float)), optional):
                Background noise electron count rate in units of 1/s
            C_sp (~astropy.units.Quantity(~numpy.ndarray(float)), optional):
                Residual speckle spatial structure (systematic error) in units of 1/s
            TK (:ref:`TimeKeeping`, optional):
                TimeKeeping object

        Returns:
            ~numpy.ndarray(float):
                Completeness values

        """

        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1) * fEZ.unit
        WA = np.array(WA.value, ndmin=1) * WA.unit
        assert len(intTimes) in [
            1,
            len(sInds),
        ], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [
            1,
            len(sInds),
        ], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [
            1,
            len(sInds),
        ], "fEZ must be constant or have same length as sInds"
        assert len(WA) in [
            1,
            len(sInds),
        ], "WA must be constant or have same length as sInds"

        return np.array([0.2] * len(sInds))

    def comp_calc(self, smin, smax, dMag):
        """Calculates completeness for given minimum and maximum separations
        and dMag.

        Args:
            smin (~numpy.ndarray(float)):
                Minimum separation(s) in AU
            smax (~numpy.ndarray(float)):
                Maximum separation(s) in AU
            dMag (~numpy.ndarray(float)):
                Difference in brightness magnitude

        Returns:
            ~numpy.ndarray(float):
                Completeness values

        .. warning::

            The prototype implementation does not perform any real completeness
            calculations.  To be used when you need a completeness object but do not
            care about the actual values.

        """

        return np.array([0.2] * len(dMag))

    def dcomp_dt(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Calculates derivative of completeness with respect to integration time

        Note: Prototype does no calculations and always returns the same value

        Args:
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (~astropy.units.Quantity(~numpy.ndarray(float)), optional):
                Background noise electron count rate in units of 1/s
            C_sp (~astropy.units.Quantity(~numpy.ndarray(float)), optional):
                Residual speckle spatial structure (systematic error) in units of 1/s
            TK (:ref:`TimeKeeping`, optional):
                TimeKeeping object

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Derivative of completeness with respect to integration time
                (units 1/time)

        """

        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1) * fEZ.unit
        WA = np.array(WA.value, ndmin=1) * WA.unit
        assert len(intTimes) in [
            1,
            len(sInds),
        ], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [
            1,
            len(sInds),
        ], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [
            1,
            len(sInds),
        ], "fEZ must be constant or have same length as sInds"
        assert len(WA) in [
            1,
            len(sInds),
        ], "WA must be constant or have same length as sInds"

        return np.array([0.02] * len(sInds)) / u.d
