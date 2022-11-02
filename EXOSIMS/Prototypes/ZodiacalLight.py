# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
import os
import pickle
import pkg_resources
from astropy.time import Time
from scipy.interpolate import interp1d


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
        self.fZ0 = 10 ** (-0.4 * self.magZ) / u.arcsec**2  # default zodi brightness
        self.fEZ0 = (
            10 ** (-0.4 * self.magEZ) / u.arcsec**2
        )  # default exo-zodi brightness

        self.global_min = 10 ** (-0.4 * self.magZ)
        self.fZMap = {}

        assert self.varEZ >= 0, "Exozodi variation must be >= 0"

        # Common Star System Number of Exo-zodi
        self.commonSystemfEZ = commonSystemfEZ  # ZL.nEZ must be calculated in SU
        self._outspec["commonSystemfEZ"] = self.commonSystemfEZ

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec", "fZ0", "fEZ0", "global_min", "fZMap"]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat
        self.logf = self.calclogf()  # create an interpolant for the wavelength

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

        nZ = np.ones(np.maximum(nStars, nTimes))
        fZ = nZ * 10 ** (-0.4 * self.magZ) / u.arcsec**2

        return fZ

    def fEZ(self, MV, I, d, alpha=2, tau=1, fbeta=None):  # noqa: E741
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
            fbeta (float, optional):
                Correction factor for inclination, default is None.
                If None, is calculated from I according to Eq. 16 of [Savransky2010]_

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

        # supplementary angle for inclination > 90 degrees
        beta = I.to("deg").value
        mask = np.where(beta > 90)[0]
        beta[mask] = 180.0 - beta[mask]
        beta = 90.0 - beta
        if fbeta is None:
            fbeta = 2.44 - 0.0403 * beta + 0.000269 * beta**2  # ESD: needs citation?

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

    def calcfbetaInput(self):
        # table 17 in Leinert et al. (1998)
        # Zodiacal Light brightness function of solar LON (rows) and LAT (columns)
        # values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
        indexf = pkg_resources.resource_filename(
            "EXOSIMS.ZodiacalLight", "Leinert98_table17.txt"
        )
        Izod = np.loadtxt(indexf) * 1e-8  # W/m2/sr/um
        # create data point coordinates
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
        # create data values, normalized by (90,0) value due to table encoding
        z = Izod / Izod[12, 0]
        values = z.reshape(z.size)

        return points, values

    def calclogf(self):
        """
        Zodi wavelength dependence, from Table 19 in Leinert et al 1998
        interpolated w/ a quadratic in log-log space

        Args:
            None
        Returns:
            interpolant (scipy.interpolate.interp1d):
                a 1D quadratic interpolant of intensity vs wavelength

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
        return interp1d(x, y, kind="quadratic")

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
        fZminglobal = 10 ** (-0.4 * self.magZ) / u.arcsec**2

        return fZminglobal
