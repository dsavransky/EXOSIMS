# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
import os
import pickle
import pkg_resources
from astropy.time import Time
from scipy.interpolate import griddata, interp1d
from synphot import units


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
        commonSystemfEZ (bool):
            Assume same zodi for planets in the same system.
            Defaults to False. TODO: Move to SimulatedUniverse
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        commonSystemfEZ (bool):
            Assume same zodi for planets in the same system.
        fEZ0 (astropy.units.quantity.Quantity):
            Default surface brightness of exo-zodiacal light in units of 1/arcsec2
        fZ0 (astropy.units.quantity.Quantity):
             Default surface brightness of zodiacal light in units of 1/arcsec2
        fZMap (dict):
            For each starlight suppression system (dict key), holds an array of the
            surface brightness of zodiacal light in units of 1/arcsec2 for each
            star over 1 year at discrete points defined by resolution
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

    def __init__(
        self, magZ=23, magEZ=22, varEZ=0, cachedir=None, commonSystemfEZ=False, **specs
    ):

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

        self.fZ0 = 10 ** (-0.4 * self.magZ) / u.arcsec**2  # default zodi brightness
        # default exo-zodi brightness
        self.fEZ0 = 10 ** (-0.4 * self.magEZ) / u.arcsec**2
        # global zodi minimum
        self.global_min = 10 ** (-0.4 * self.magZ)
        self.fZMap = {}

        # Common Star System Number of Exo-zodi
        self.commonSystemfEZ = commonSystemfEZ  # ZL.nEZ must be calculated in SU

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec", "fZ0", "fEZ0", "global_min", "fZMap"]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # Load zodi data
        self.load_zodi_wavelength_data()
        self.load_zodi_spatial_data()

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
        sInds = np.array(sInds, ndmin=1, copy=False)
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
        fZ = nZ * 10 ** (-0.4 * self.magZ) / u.arcsec**2

        return fZ

    def fEZ(self, MV, I, d, alpha=2, tau=1):
        """Returns surface brightness of exo-zodiacal light

        Args:
            MV (~numpy.ndarray(int)):
                Absolute magnitude of the star (in the V band)
            I (~astropy.units.Quantity(~numpy.ndarray(float))):
                Inclination of the planets of interest in units of deg
            d (~astropy.units.Quantity(~numpy.ndarray(float))):
                nx3 Distance to star of the planets of interest in units of AU
            alpha (float):
                power applied to radial distribution, default=2
            tau (float):
                disk morphology dependent throughput correction factor, default =1

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2

        """

        # Absolute magnitude of the star (in the V band)
        MV = np.array(MV, ndmin=1, copy=False)
        # Absolute magnitude of the Sun (in the V band)
        MVsun = 4.83

        if self.commonSystemfEZ:
            nEZ = self.nEZ
        else:
            nEZ = self.gen_systemnEZ(len(MV))

        # inclinations should be strictly in [0, pi], but allow for weird sampling:
        beta = I.to("deg").value
        beta[beta > 180] -= 180

        # latitudinal variations are symmetric about 90 degrees so compute the
        # supplementary angle for inclination > 90 degrees
        mask = beta > 90
        beta[mask] = 180.0 - beta[mask]

        # finally, the input to the model is 90-inclination
        beta = 90.0 - beta
        fbeta = self.zodi_latitudinal_correction_factor(beta * u.deg, model="interp")

        fEZ = (
            nEZ
            * 10 ** (-0.4 * self.magEZ)
            * 10.0 ** (-0.4 * (MV - MVsun))
            * fbeta
            / d.to("AU").value ** alpha
            / u.arcsec**2
            * tau
        )

        return fEZ

    def gen_systemnEZ(self, nStars):
        """Ranomly generates the number of Exo-Zodi

        Args:
            nStars (int):
                number of exo-zodi to generate
        Returns:
            ~numpy.ndarray:
                numpy array of exo-zodi values in number of local zodi
        """

        # assume log-normal distribution of variance
        nEZ = np.ones(nStars)
        if self.varEZ != 0:
            mu = np.log(nEZ) - 0.5 * np.log(1.0 + self.varEZ / nEZ**2)
            v = np.sqrt(np.log(self.varEZ / nEZ**2 + 1.0))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=nStars)

        return nEZ

    def generate_fZ(self, Obs, TL, TK, mode, hashname):
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

        Returns:
            None

        Updates Attributes:
            fZMap[1000, TL.nStars] (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of zodiacal light in units of 1/arcsec2 for each
                star over 1 year at discrete points defined by resolution
        """

        # Generate cache Name
        cachefname = hashname + "starkfZ"

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZ from %s" % cachefname)
            try:
                with open(cachefname, "rb") as ff:
                    tmpfZ = pickle.load(ff)
            except UnicodeDecodeError:
                with open(cachefname, "rb") as ff:
                    tmpfZ = pickle.load(ff, encoding="latin1")
            self.fZMap[mode["syst"]["name"]] = tmpfZ

        # IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        else:
            self.vprint(f"Calculating fZ for {mode['syst']['name']}")
            sInds = np.arange(TL.nStars)
            startTime = (
                np.zeros(sInds.shape[0]) * u.d + TK.currentTimeAbs
            )  # Array of current times
            resolution = [j for j in range(1000)]
            fZ = np.zeros([sInds.shape[0], len(resolution)])
            dt = 365.25 / len(resolution) * u.d
            for i in range(len(resolution)):  # iterate through all times of year
                time = startTime + dt * resolution[i]
                fZ[:, i] = self.fZ(Obs, TL, sInds, time, mode)

            with open(cachefname, "wb") as fo:
                pickle.dump(fZ, fo)
                self.vprint("Saved cached 1st year fZ to %s" % cachefname)
            self.fZMap[mode["syst"]["name"]] = fZ

    def calcfZmax(self, sInds, Obs, TL, TK, mode, hashname):
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
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get all array sizes
        nStars = sInds.size

        nZ = np.ones(nStars)
        valfZmax = nZ * 10 ** (-0.4 * self.magZ) / u.arcsec**2

        absTimefZmax = nZ * u.d + TK.currentTimeAbs

        return valfZmax[sInds], absTimefZmax[sInds]

    def calcfZmin(self, sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
        """Finds the minimum zodiacal light values for each star over an entire orbit
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
            koMap (~numpy.ndarray(bool)):
                True is a target unobstructed and observable, and False is a
                target unobservable due to obstructions in the keepout zone.
            koTimes (~astropy.time.Time(~numpy.ndarray(float))):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            list:
                list of local zodiacal light minimum and times they occur at
                (should all have same value for prototype)
        """

        # Generate cache Name
        cachefname = hashname + "fZmin"

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZQuads from %s" % cachefname)
            # load from cache. fZQuads has the form tmpDat len sInds, tmpDat[0] len
            # number of ko enter/exits and localmin occurences, tmpDat[0,0] form
            # [type,fZvalue,absTime]
            with open(cachefname, "rb") as f:  #
                fZQuads = pickle.load(f)
                # Convert Abs time to MJD object
                for i in np.arange(len(fZQuads)):
                    for j in np.arange(len(fZQuads[i])):
                        fZQuads[i][j][3] = Time(
                            fZQuads[i][j][3], format="mjd", scale="tai"
                        )
                        fZQuads[i][j][1] = fZQuads[i][j][1] / u.arcsec**2.0

            return [fZQuads[i] for i in sInds]
        else:
            # cast sInds to array
            sInds = np.array(sInds, ndmin=1, copy=False)

            # this whole block is deprecated
            # #get all array sizes
            # nStars = sInds.size
            # nZ = np.ones(nStars)
            # valfZmin = nZ * 10 ** (-0.4 * self.magZ) / u.arcsec**2
            # absTimefZmin = nZ * u.d + TK.currentTimeAbs

            tmpfZ = np.asarray(self.fZMap[mode["syst"]["name"]])
            fZ_matrix = tmpfZ[sInds, :]  # Apply previous filters to fZMap[sInds, 1000]
            dt = 365.25 / len(np.arange(1000))
            timeArray = [j * dt for j in np.arange(1000)]
            timeArrayAbs = TK.currentTimeAbs + timeArray * u.d

            # When are stars in KO regions
            missionLife = TK.missionLife.to("yr")
            # if this is being calculated without a koMap,
            # or if missionLife is less than a year
            if (koMap is None) or (missionLife.value < 1):
                # calculating keepout angles and keepout values for 1 system in mode
                koStr = list(
                    filter(
                        lambda syst: syst.startswith("koAngles_"), mode["syst"].keys()
                    )
                )
                koangles = np.asarray([mode["syst"][k] for k in koStr]).reshape(1, 4, 2)
                kogoodStart = Obs.keepout(TL, sInds, timeArrayAbs, koangles)[0].T
            else:
                # getting the correct koTimes to look up in koMap
                assert koTimes is not None, "koTimes not included in input statement."
                koInds = np.zeros(len(timeArray), dtype=int)
                for x in np.arange(len(timeArray)):
                    koInds[x] = np.where(
                        np.round((koTimes - timeArrayAbs[x]).value) == 0
                    )[0][0]
                # determining ko values within a year using koMap
                kogoodStart = koMap[:, koInds].T

            fZQuads = list()
            for k in np.arange(len(sInds)):
                i = sInds[k]  # Star ind
                # Find inds of local minima in fZ
                fZlocalMinInds = np.where(
                    np.diff(np.sign(np.diff(fZ_matrix[i, :]))) > 0
                )[
                    0
                ]  # Find local minima of fZ
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind, i]]
                # This happens in prototype module. Caused by all values in
                # fZ_matrix being the same
                if fZlocalMinInds == []:
                    fZlocalMinInds = [0]

                fZlocalMinIndsQuad = [
                    [
                        2,
                        fZ_matrix[i, fZlocalMinInds[j]],
                        timeArray[fZlocalMinInds[j]],
                        (
                            TK.currentTimeAbs.copy()
                            + TK.currentTimeNorm % (1.0 * u.year).to("day")
                            + fZlocalMinInds[j] * dt * u.d
                        ).value,
                    ]
                    for j in np.arange(len(fZlocalMinInds))
                ]
                fZQuads.append(fZlocalMinIndsQuad)

            with open(cachefname, "wb") as fo:
                pickle.dump(fZQuads, fo)
                self.vprint("Saved cached fZQuads to %s" % cachefname)

            # Convert Abs time to MJD object
            for i in np.arange(len(fZQuads)):
                for j in np.arange(len(fZQuads[i])):
                    fZQuads[i][j][3] = Time(fZQuads[i][j][3], format="mjd", scale="tai")
                    fZQuads[i][j][1] = fZQuads[i][j][1] / u.arcsec**2.0

            return [fZQuads[i] for i in sInds]

    def extractfZmin_fZQuads(self, fZQuads):
        """Extract the global fZminimum from fZQuads

        Args:
            fZQuads (list):
                fZQuads has shape [sInds][Number fZmin][4]

        Returns:
            tuple:
                valfZmin (astropy.units.Quantity(numpy.ndarray)):
                    fZ minimum for the target
                absTimefZmin (astropy.time.Time(numpy.ndarray)):
                    Absolute time the fZmin occurs

        .. note::

            This produces the same output as calcfZmin circa January 2019.

            For the prototype, fZQuads is equivalent to (valfZmin, absTimefZmin)
            so we simply return that

        """
        valfZmin = list()
        absTimefZmin = list()
        for i in np.arange(len(fZQuads)):  # Iterates over each star
            ffZmin = 100.0
            fabsTimefZmin = 0.0
            for j in np.arange(
                len(fZQuads[i])
            ):  # Iterates over each occurence of a minimum
                if fZQuads[i][j][1].value < ffZmin:
                    ffZmin = fZQuads[i][j][1].value
                    fabsTimefZmin = fZQuads[i][j][3].value

            if len(fZQuads[i]) == 0:
                ffZmin = np.nan
                fabsTimefZmin = -1

            valfZmin.append(ffZmin)
            absTimefZmin.append(fabsTimefZmin)

            assert ffZmin != 100.0, "fZmin not below 100 counts/arcsec^2"

            assert fabsTimefZmin != 0.0, "absTimefZmin is 0 days"

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
        indexf = pkg_resources.resource_filename(
            "EXOSIMS.ZodiacalLight", "Leinert98_table17.txt"
        )
        Izod = np.loadtxt(indexf) * 1e-8  # W/m2/sr/um
        self.zodi_values = Izod.reshape(Izod.size) * u.Unit("W m-2 sr-1 um-1")

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

        self.zodi_points = points * u.deg

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

        val = 10.0 ** (self.logf(np.log10(lam.to("um").value))) * u.Unit(
            "W m-2 sr-1 um-1"
        )

        if photon_units:
            val = (units.convert_flux(lam, val * u.sr, units.PHOTLAM) / u.sr).to(
                "ph s-1 m-2 um-1 sr-1"
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

        fcolor = self.zodi_intensity_at_wavelength(
            lam, photon_units=photon_units
        ) / self.zodi_intensity_at_wavelength(0.5 * u.um, photon_units=photon_units)

        return fcolor.value

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

        lons = np.array(lons.to(u.deg).value, ndmin=1)
        lats = np.array(lats.to(u.deg).value, ndmin=1)

        zodiflux = (
            griddata(
                self.zodi_points.value,
                self.zodi_values.value,
                np.vstack([lons.flatten(), lats.flatten()]).transpose(),
            )
            * self.zodi_values.unit
        )

        if photon_units:
            zodiflux = (
                units.convert_flux(500 * u.nm, zodiflux * u.sr, units.PHOTLAM) / u.sr
            ).to("ph s-1 m-2 um-1 sr-1")

        return zodiflux.reshape(lons.shape)

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
                (case insensitive). See :ref:`zodiandexozodi` for details.
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
            fbeta = fbetafun(theta.to(u.deg).value)

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
            / u.arcsec**2
        )

        return fZminglobal
