# -*- coding: utf-8 -*-
import importlib.resources
import os
import pickle
import sys

import astropy.units as u
import numpy as np
from astropy.time import Time
from scipy.interpolate import LinearNDInterpolator, interp1d
from synphot import Observation, SourceSpectrum, SpectralElement, units

from EXOSIMS.util._numpy_compat import copy_if_needed
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.vprint import vprint


class ZodiacalLight(object):
    """:ref:`ZodiacalLight` Prototype

    Args:
        magZ (float):
            Local zodi brightness (magnitudes per square arcsecond).
            Defaults to 23
        magEZ (float)
            Exozodi brightness (mangitudes per square arcsecond).
            Defaults to 22
        varEZ (float):
            Variance of exozodi brightness. If non-zero treat as the
            variance of a log-normal distribution. If zero, do not
            randomly distribute exozodi brightnesses. Defaults to 0
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
        F0V (astropy.units.quantity.Quantity):
            V band zero magnitude spectral flux density
        fEZ0 (astropy.units.quantity.Quantity):
            Default surface brightness of exo-zodiacal light in units of 1/arcsec2
        fZ0 (astropy.units.quantity.Quantity):
             Default surface brightness of zodiacal light in units of 1/arcsec2
        fZMap (dict):
            For each starlight suppression system (dict key), holds an array of the
            surface brightness of zodiacal light in units of 1/arcsec2 for each
            star over 1 year at discrete points defined by resolution
        fZTimes (~astropy.time.Time(~numpy.ndarray(float))):
                Absolute MJD mission times from start to end
        global_min (float):
            The global minimum zodiacal light value
        magEZ (float):
            1 exo-zodi brightness magnitude (per arcsec2)
        magZ (float):
            1 zodi brightness magnitude (per arcsec2)
        varEZ (float):
            Variance of exozodi brightness. If non-zero treat as the
            variance of a log-normal distribution. If zero, do not
            randomly distribute exozodi brightnesses.
        zodi_Blam (numpy.ndarray):
            Local zodi table data (W/m2/sr/um)
        zodi_lam (numpy.ndarray):
            Local zodi table data wavelengths (micrometers)

    """

    _modtype = "ZodiacalLight"

    def __init__(self, magZ=23, magEZ=22, varEZ=0, cachedir=None, **specs):
        # Define units
        self.zodi_intens_unit = u.Unit(u.W / u.m**2 / u.sr / u.um)
        self.zodi_intens_unit_photon = u.Unit(u.ph / u.s / u.m**2 / u.um / u.sr)
        self.inv_arcsec2 = 1 / u.arcsec**2

        # start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        self.magZ = float(magZ)  # 1 zodi brightness (per arcsec2)
        self.magEZ = float(magEZ)  # 1 exo-zodi brightness (per arcsec2)
        self.varEZ = float(varEZ)  # exo-zodi variation (variance of log-normal dist)
        assert self.varEZ >= 0, "Exozodi variation must be >= 0"

        # default zodi brightness
        self.fZ0 = 10 ** (-0.4 * self.magZ) << self.inv_arcsec2
        # default exo-zodi brightness
        self.fEZ0 = 10 ** (-0.4 * self.magEZ) << self.inv_arcsec2
        # global zodi minimum
        self.global_min = 10 ** (-0.4 * self.magZ)
        self.fZMap = {}
        self.fZTimes = Time(np.array([]), format="mjd", scale="tai")
        # Calculate the V band zero magnitude flux density
        vega_spectrum = SourceSpectrum.from_vega()
        v_band = SpectralElement.from_filter("johnson_v")
        self.F0V = (
            Observation(vega_spectrum, v_band).integrate() / v_band.equivwidth()
        ).to(u.ph / (u.s * u.m**2 * u.nm))

        # populate outspec
        for att in self.__dict__:
            if att not in [
                "vprint",
                "_outspec",
                "fZ0",
                "fEZ0",
                "global_min",
                "fZMap",
                "fZTimes",
                "zodi_intens_unit",
                "zodi_intens_unit_photon",
                "inv_arcsec2",
                "F0V",
            ]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # Load zodi data
        self.load_zodi_wavelength_data()
        self.load_zodi_spatial_data()

        # Set up faster interpolant
        self.lon_grid = np.linspace(0, 360, 360, endpoint=False)
        self.lat_grid = np.linspace(-90, 90, 180)
        self.grid_lon, self.grid_lat = np.meshgrid(self.lon_grid, self.lat_grid)

        # Create the RegularGridInterpolator object
        self.zodi_intensity_interp = LinearNDInterpolator(
            self.zodi_points.value,
            self.zodi_values.value,
            fill_value=np.nan,
        )

    def __str__(self):
        """String representation of the Zodiacal Light object

        When the command 'print' is used on the Zodiacal Light object, this
        method will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Zodiacal Light class object attributes"

    def fZ(self, Obs, TL, sInds, currentTimeAbs, mode):
        """Returns surface brightness of local zodiacal light

        Args:
            Obs (:ref:`Observatory`):
                Observatory class object
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTimeAbs (~astropy.time.Time):
                absolute time to evaluate fZ for
            mode (dict):
                Selected observing mode

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Surface brightness of zodiacal light in units of 1/arcsec2

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        # get all array sizes
        nStars = sInds.size
        nTimes = currentTimeAbs.size
        assert (
            nStars == 1 or nTimes == 1 or nTimes == nStars
        ), "If multiple times and targets, currentTimeAbs and sInds sizes must match."

        # compute correction factors
        nZ = np.ones(np.maximum(nStars, nTimes)) * self.zodi_color_correction_factor(
            mode["lam"], photon_units=True
        )
        fZ = nZ * 10 ** (-0.4 * self.magZ) << self.inv_arcsec2

        return fZ

    def calc_JEZ0(self, MV, L, flambda, bandwidth):
        """Returns intensity of exo-zodiacal light for the stars of interest

        Args:
            MV (~numpy.ndarray(int)):
                Absolute magnitude of the stars (in V band)
            L (~numpy.ndarray(float)):
                Bolometric luminosity of the stars (in solar units)
            flambda (~numpy.ndarray(float)):
                Color scale factor
            bandwidth (~astropy.units.Quantity(float)):
                Bandwidth of the observing mode

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Intensity of exo-zodiacal light in units of ph/s/m^2/arcsec^2

        """

        # Absolute magnitude of the star (in the V band)
        MV = np.array(MV, ndmin=1, copy=copy_if_needed)
        # Absolute magnitude of the Sun (in the V band)
        MVsun = 4.83

        JEZ0 = (
            self.F0V
            * 10.0 ** (-0.4 * (MV - MVsun))
            * self.fEZ0  # The x term in Fundamental Concepts documentation
            * flambda
            * bandwidth
            / L
        ).to(u.ph / u.s / u.m**2 / u.arcsec**2)

        return JEZ0

    def calc_fbeta(self, inclinations, model="interp"):
        """Returns the latitudinal correction factor for the zodiacal light.

        Args:
            inclinations (~astropy.units.Quantity(~numpy.ndarray(float))):
                Inclinations of the system planes
            model (str):
                Model to use.  Options are Lindler2006, Stark2014, or interp
                (case insensitive). See :ref:`exozodi` for details.
                Defaults to 'interp'

        Returns:
            ~numpy.ndarray(float):
                Correction factor of zodiacal light at requested angles.
                Has same dimension as the input inclinations.
        """
        # inclinations should be strictly in [0, pi], but allow for weird sampling:
        beta = inclinations.to_value(u.deg)
        beta[beta > 180] -= 180

        # latitudinal variations are symmetric about 90 degrees so compute the
        # supplementary angle for inclination > 90 degrees
        mask = beta > 90
        beta[mask] = 180.0 - beta[mask]

        # finally, the input to the model is 90-inclination
        beta = 90.0 - beta
        fbeta = self.zodi_latitudinal_correction_factor(beta << u.deg, model=model)
        return fbeta

    def gen_systemnEZ(self, nStars):
        """Ranomly generates the number of Exo-Zodi

        Args:
            nStars (int):
                number of exo-zodi to generate
        Returns:
            ~numpy.ndarray(float):
                numpy array of exo-zodi values in number of local zodi
        """

        # assume log-normal distribution of variance
        nEZ = np.ones(nStars)
        if self.varEZ != 0:
            mu = np.log(nEZ) - 0.5 * np.log(1.0 + self.varEZ / nEZ**2)
            v = np.sqrt(np.log(self.varEZ / nEZ**2 + 1.0))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=nStars)

        return nEZ

    def generate_fZ(self, Obs, TL, TK, mode, hashname, koTimes=None):
        """Calculates fZ values for all stars over an entire orbit of the sun

        Args:
            Obs (:ref:`Observatory`):
                Observatory class object
            TL (:ref:`TargetList`):
                TargetList class object
            TK (:ref:`TimeKeeping`):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (str):
                hashname describing the files specific to the current json script
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            None

        Updates Attributes:
            fZMap[n, TL.nStars] (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of zodiacal light in units of 1/arcsec2
                for every star for every ko_dtStep
            fZTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end, updated if koTimes
                does not exist
        """

        # Generate cache Name
        cachefname = hashname + "starkfZ"

        if koTimes is None:
            fZTimes = np.arange(
                TK.missionStart.value, TK.missionFinishAbs.value, Obs.ko_dtStep.value
            )
            self.fZTimes = Time(
                fZTimes, format="mjd", scale="tai"
            )  # scale must be tai to account for leap seconds
            koTimes = fZTimes

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZ from %s" % cachefname)
            with open(cachefname, "rb") as ff:
                tmpfZ = pickle.load(ff)
            self.fZMap[mode["syst"]["name"]] = tmpfZ

        else:
            self.vprint(f"Calculating fZ for {mode['syst']['name']}")
            sInds = np.arange(TL.nStars)
            # calculate fZ for every star at same times as keepout map
            fZ = np.zeros([sInds.shape[0], len(koTimes)])
            for i in range(len(koTimes)):  # iterate through all times of year
                fZ[:, i] = self.fZ(Obs, TL, sInds, koTimes[i], mode)

            with open(cachefname, "wb") as fo:
                pickle.dump(fZ, fo)
                self.vprint("Saved cached fZ to %s" % cachefname)
            self.fZMap[mode["syst"]["name"]] = fZ
            # index by hexkey instead of system name

    def calcfZmax(self, sInds, Obs, TL, TK, mode, hashname, koTimes=None):
        """Finds the maximum zodiacal light values for each star over an entire orbit
        of the sun not including keeoput angles.

        Args:
            sInds (~numpy.ndarray(int)):
                the star indicies we would like fZmax and fZmaxInds returned for
            Obs (:ref:`Observatory`):
                Observatory class object
            TL (:ref:`TargetList`):
                TargetList class object
            TK (:ref:`TimeKeeping`):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (str):
                hashname describing the files specific to the current json script
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            tuple:
                valfZmax[sInds] (~astropy.units.Quantity(~numpy.ndarray(float))):
                    the maximum fZ (for the prototype, these all have the same value)
                    with units 1/arcsec**2
                absTimefZmax[sInds] (astropy.time.Time):
                    returns the absolute Time the maximum fZ occurs (for the prototype,
                    these all have the same value)
        """
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        # get all array sizes
        nStars = sInds.size

        nZ = np.ones(nStars)
        valfZmax = nZ * 10 ** (-0.4 * self.magZ) << self.inv_arcsec2

        absTimefZmax = nZ * u.d + TK.currentTimeAbs

        return valfZmax[sInds], absTimefZmax[sInds]

    def calcfZmin(self, sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
        """Finds the minimum zodiacal light values for each star over an entire orbit
        of the sun not including keeoput angles.

        Args:
            sInds (~numpy.ndarray(int)):
                the star indicies we would like fZmins and fZtypes returned for
            Obs (:ref:`Observatory`):
                Observatory class object
            TL (:ref:`TargetList`):
                TargetList class object
            TK (:ref:`TimeKeeping`):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (str):
                hashname describing the files specific to the current json script
            koMap (~numpy.ndarray(bool), optional):
                True is a target unobstructed and observable, and False is a
                target unobservable due to obstructions in the keepout zone.
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end, in steps of 1 d as default

        Returns:
            tuple:
                fZmins[n, TL.nStars] (~astropy.units.Quantity(~numpy.ndarray(float))):
                    fZMap, but only fZmin candidates remain. All other values are set to
                    the maximum floating number. Units are 1/arcsec2
                fZtypes [n, TL.nStars] (~numpy.ndarray(float)):
                    ndarray of flags for fZmin types that map to fZmins
                    0 - entering KO
                    1 - exiting KO
                    2 - local minimum
                    max float - not a fZmin candidate
        """

        # Generate cache Name
        cachefname = hashname + "fZmin"

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZmins from %s" % cachefname)
            with open(cachefname, "rb") as f:  # load from cache
                tmp1 = pickle.load(f)
                fZmins = tmp1["fZmins"]
                fZtypes = tmp1["fZtypes"]
            return fZmins, fZtypes
        else:
            tmpAssert = np.any(self.fZMap[mode["syst"]["name"]])
            assert tmpAssert, "fZMap does not exist for the mode of interest"

            tmpfZ = np.asarray(self.fZMap[mode["syst"]["name"]])
            fZ_matrix = tmpfZ[sInds, :]  # Apply previous filters to fZMap

            # When are stars in KO regions
            # if calculated without a koMap
            if koMap is None:
                koTimes = self.fZTimes

                # calculating keepout angles and keepout values for 1 system in mode
                koStr = list(
                    filter(
                        lambda syst: syst.startswith("koAngles_"), mode["syst"].keys()
                    )
                )
                koangles = np.asarray([mode["syst"][k] for k in koStr]).reshape(1, 4, 2)
                kogoodStart = Obs.keepout(TL, sInds, koTimes[0], koangles)[0].T
                nn = len(sInds)
                mm = len(koTimes)
            else:
                # getting the correct koTimes to look up in koMap
                assert (
                    koTimes is not None
                ), "Corresponding koTimes not included with koMap."
                kogoodStart = koMap.T
                [nn, mm] = np.shape(koMap)

            fZmins = np.ones([nn, mm]) * sys.float_info.max
            fZtypes = np.ones([nn, mm]) * sys.float_info.max

            for k in np.arange(len(sInds)):
                i = sInds[k]  # Star ind
                # Find inds of local minima in fZ
                fZlocalMinInds = (
                    np.where(np.diff(np.sign(np.diff(fZ_matrix[i, :]))) > 0)[0] + 1
                )  # Find local minima of fZ, +1 to correct for indexing offset
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind, i]]
                # This happens in prototype module. Caused by all values in
                # fZ_matrix being the same
                if len(fZlocalMinInds) == 0:
                    fZlocalMinInds = [0]

                if len(fZlocalMinInds) > 0:
                    fZmins[i, fZlocalMinInds] = fZ_matrix[i, fZlocalMinInds]
                    fZtypes[i, fZlocalMinInds] = 2

            with open(cachefname, "wb") as fo:
                pickle.dump({"fZmins": fZmins, "fZtypes": fZtypes}, fo)
                self.vprint("Saved cached fZmins to %s" % cachefname)

            return fZmins, fZtypes

    def extractfZmin(self, fZmins, sInds, koTimes=None):
        """Extract the global fZminimum from fZmins

        Args:
            fZmins (~astropy.units.Quantity(~numpy.ndarray(float))):
                fZMap, but only fZmin candidates remain. All other values are set to
                the maximum floating number. Units are 1/arcsec2.
                Dimension [n, TL.nStars]
            sInds (~numpy.ndarray(int)):
                the star indicies we would like valfZmin and absTimefZmin returned
                for
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end, in steps of 1 d as
                default

        Returns:
            tuple:
                valfZmin[sInds] (~astropy.units.Quantity(~numpy.ndarray(float))):
                    the minimum fZ (for the prototype, these all have the same
                    value) with units 1/arcsec**2
                absTimefZmin[sInds] (astropy.time.Time):
                    returns the absolute Time the maximum fZ occurs (for the
                    prototype, these all have the same value)
        """

        if koTimes is None:
            koTimes = self.fZTimes

        # Find minimum fZ of each star of the fZmins set
        valfZmin = np.zeros(sInds.shape[0])
        absTimefZmin = np.zeros(sInds.shape[0])
        for i in range(len(sInds)):
            tmpfZmin = min(fZmins[i, :])  # fZ_matrix has dimensions sInds

            if tmpfZmin == sys.float_info.max:
                valfZmin[i] = np.nan
                absTimefZmin[i] = -1
            else:
                valfZmin[i] = tmpfZmin
                indfZmin = np.argmin(fZmins[i, :])  # Gets indices where fZmin occurs
                absTimefZmin[i] = koTimes[indfZmin].value
        # The np.asarray and Time must occur to create astropy Quantity arrays and
        # astropy Time arrays
        return np.asarray(valfZmin) / u.arcsec**2.0, Time(
            np.asarray(absTimefZmin), format="mjd", scale="tai"
        )

    def load_zodi_spatial_data(self):
        """
        Zodi spatial variation data from Table 17 of [Leinert1998]_

        Zodiacal Light brightness as function of solar LON (rows) and LAT (columns)
        Values are given in 10^-8 W m−2 sr−1 μm−1 at a wavelength of 500 nm

        Attributes:
            zodi_points (numpy.ndarry):
                nx2 lat,lon pairs
            zodi_values (astropy.units.Quantity):
                n intensity values (units of W/m2/sr/um)

        """

        # Read data from disk
        indexf = os.path.join(
            importlib.resources.files("EXOSIMS.ZodiacalLight"), "Leinert98_table17.txt"
        )

        Izod = np.loadtxt(indexf) * 1e-8  # W/m2/sr/um
        self.zodi_values = Izod.reshape(Izod.size) << self.zodi_intens_unit

        # create point coordinates
        lon_pts = np.array(
            [
                0.0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                60,
                75,
                90,
                105,
                120,
                135,
                150,
                165,
                180,
            ]
        )  # deg
        lat_pts = np.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90])  # deg
        y_pts, x_pts = np.meshgrid(lat_pts, lon_pts)
        points = np.array(list(zip(np.concatenate(x_pts), np.concatenate(y_pts))))

        self.zodi_points = points << u.deg

    def load_zodi_wavelength_data(self):
        """
        Zodi wavelength dependence, from Table 19 of [Leinert1998]_
        interpolated w/ a quadratic in log-log space

        Creates an interpolant (scipy.interpolate.interp1d) assigned to attribute
        ``logf`` which takes as an argument log_10(wavelength in um) and returns
        log10(specific intensity in W/m^2/um/sr)

        """
        self.zodi_lam = np.array(
            [
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
                1.0,
                1.2,
                2.2,
                3.5,
                4.8,
                12,
                25,
                60,
                100,
                140,
            ]
        )  # um
        self.zodi_Blam = np.array(
            [
                2.5e-8,
                5.3e-7,
                2.2e-6,
                2.6e-6,
                2.0e-6,
                1.3e-6,
                1.2e-6,
                8.1e-7,
                1.7e-7,
                5.2e-8,
                1.2e-7,
                7.5e-7,
                3.2e-7,
                1.8e-8,
                3.2e-9,
                6.9e-10,
            ]
        )  # W/m2/sr/um
        x = np.log10(self.zodi_lam)
        y = np.log10(self.zodi_Blam)

        self.logf = interp1d(x, y, kind="quadratic")

    def zodi_intensity_at_wavelength(self, lam, photon_units=False):
        """
        Compute zodiacal light specific intensity as a function of wavelength

        Args:
            lam (astropy.units.Quantity):
                Wavelength(s) of interest
            photon_units(bool):
                Convert all quantities to photon units before computing ratio.
                Defaults False (leave all quantities in power units).

        Returns:
            astropy.units.Quantity:
                Specific intensity of zodiacal light at requested wavelength(s).
                Has same dimension as input. Default units of W m-2 um-1 sr-1 if
                ``photon_units`` is False, otherwise  ph s-1 m-2 um-1 sr-1

        .. warning:

            This method uses the interpolant stored in ``logf`` and defined by method
            ``load_zodi_wavelength_data``.  This must return intensitites in units of
            log10(W/m^2/um/sr).
        """
        # Almost always lam is provided in nanometers so we can save time by calling
        # to_value(u.nm)/1000.0 to convert to micrometers since `to_value` will
        # turn the call into simply `_lam.value` when its unit is nanometers.
        val = (
            10.0 ** (self.logf(np.log10(lam.to_value(u.nm) / 1000)))
            << self.zodi_intens_unit
        )

        if photon_units:
            val = (units.convert_flux(lam, val * u.sr, units.PHOTLAM) / u.sr).to(
                self.zodi_intens_unit_photon
            )

        return val

    def zodi_color_correction_factor(self, lam, photon_units=False):
        """
        Compute zodiacal light color correction factor. This is a multiplicative
        factor to apply to zodiacal light intensity computed at a reference wavelength
        (500 nm for the Leinert data used in this prototype).

        Args:
            lam (astropy.units.Quantity):
                Wavelength(s) of interest
            photon_units(bool):
                Convert all quantities to photon units before computing ratio.
                Defaults False (leave all quantities in power units).

        Returns:
            float or numpy.ndarray:
                Specific intensity of zodiacal light at requested wavelength(s) scaled
                by the value at the reference wavelength (500 nm).
                Has same dimension as input.

        .. warning:

            While itself unitless, the units of the original intensities must match
            those of the intensity or flux being scaled. If the quantity being scaled
            has power units, ``photon_units`` must be False.
        """

        fcolor = (
            self.zodi_intensity_at_wavelength(lam, photon_units=photon_units).value
            / self.zodi_intensity_at_wavelength(
                0.5 * u.um, photon_units=photon_units
            ).value
        )

        return fcolor

    def zodi_intensity_at_location(self, lons, lats, photon_units=False):
        """
        Compute zodiacal light specific intensity as a function of look vector at
        reference wavelength (500 nm)

        Args:
            lons (astropy.units.Quantity):
                Ecliptic longitude minus solar ecliptic longitude
            lats (astropy.units.Quantity):
                Ecliptic latitude.  Must be of same dimension as lons.
            photon_units(bool):
                Convert all quantities to photon units before computing ratio.
                Defaults False (leave all quantities in power units).

        Returns:
            astropy.units.Quantity:
                Specific intensity of zodiacal light at requested wavelength(s).
                Has same dimension as input. Default units of W m-2 um-1 sr-1 if
                ``photon_units`` is False, otherwise  ph s-1 m-2 um-1 sr-1
        """

        target_points = np.column_stack(
            (lons.to_value(u.deg).flatten(), lats.to_value(u.deg).flatten())
        )

        # Perform interpolation
        interpolated_values = self.zodi_intensity_interp(target_points)

        # Reshape to original grid shape
        zodiflux = interpolated_values.reshape(lons.shape) << self.zodi_values.unit

        if photon_units:
            zodiflux = (
                units.convert_flux(500 * u.nm, zodiflux * u.sr, units.PHOTLAM) / u.sr
            ).to(self.zodi_intens_unit_photon)

        return zodiflux

    def zodi_latitudinal_correction_factor(self, theta, model=None, interp_at=135):
        """
        Compute zodiacal light latitudinal correction factor.  This is a multiplicative
        factor to apply to zodiacal light intensity to account for the orientation of
        the dust disk with respect to the observer.

        Args:
            theta (astropy.units.Quantity):
                Angle of disk. For local zodi, this is equivalent to the absolute value
                of the ecliptic latitude of the look vector. For exozodi, this is 90
                degrees minus the inclination of the orbital plane.
            model (str, optional):
                Model to use.  Options are Lindler2006, Stark2014, or interp
                (case insensitive). See :ref:`exozodi` for details.
                Defaults to None
            interp_at (float):
                If ``model`` is 'interp', interpolate Leinert Table 17 at this
                longitude. Defaults to 135.

        Returns:
            float or numpy.ndarray:
                Correction factor of zodiacal light at requested angles.
                Has same dimension as input.

        .. note::

            Unlike the color correction factor, this quantity is wavelength independent
            and thus does not change if using power or photon units.

        """

        if model is not None:
            model = model.lower()
            assert model in [
                "lindler2006",
                "stark2014",
                "interp",
            ], "Model must be one of Lindler2006, Stark2014, or interp"
        else:
            return np.ones(theta.shape)

        if model == "lindler2006":
            beta = theta.to(u.deg).value
            fbeta = (2.44 - 0.0403 * beta + 0.000269 * beta**2) / 2.44
        elif model == "stark2014":
            fbeta = (
                1.02
                - 0.566 * np.sin(theta)
                - 0.884 * np.sin(theta) ** 2
                + 0.853 * np.sin(theta) ** 3
            )
        else:
            # figure out the interpolant
            interpname = f"interp{np.round(interp_at)}"
            if not hasattr(self, interpname):
                inds = self.zodi_points[:, 0].value == interp_at
                vals = self.zodi_values[inds]
                vals = vals / vals[0]
                setattr(
                    self,
                    interpname,
                    interp1d(self.zodi_points[inds, 1].value, vals.value, kind="cubic"),
                )

            fbetafun = getattr(self, interpname)
            fbeta = fbetafun(theta.to_value(u.deg))

        return fbeta

    def global_zodi_min(self, mode):
        """
        This is used to determine the minimum zodi value globally, for the
        prototype it simply returns the same value that fZ always does

        Args:
            mode (dict):
                Selected observing mode

        Returns:
            ~astropy.units.Quantity:
                The global minimum zodiacal light value for the observing mode,
                in (1/arcsec**2)
        """
        fZminglobal = (
            self.zodi_color_correction_factor(mode["lam"], photon_units=True)
            * 10 ** (-0.4 * self.magZ)
            * self.inv_arcsec2
        )

        return fZminglobal
