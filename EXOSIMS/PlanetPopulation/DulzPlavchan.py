from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import os
import inspect
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const
import scipy.interpolate as interpolate
from EXOSIMS.util._numpy_compat import copy_if_needed


class DulzPlavchan(PlanetPopulation):
    """
    Population based on occurrence rate tables from Shannon Dulz and Peter Plavchan.

    The data comes as either Period-Radius or semi-major axis-mass pairs.
    If occDataPath is not specified, the nominal Period-Radius table is loaded.

    Args:
        specs:
            user specified values

    Attributes:
        starMass:
            stellar mass in M_sun used to convert period to semi-major axis
        occDataPath:
            path on local disk to occurrence rate table
        esigma (float):
            Sigma value of Rayleigh distribution for eccentricity.

    Notes:
    1. Mass/Radius and semi-major axis are specified in occurrence rate tables.
    User specified values will be ignored.
    2. Albedo is sampled as in KeplerLike1 and KeplerLike2.
    3. Eccentricity is Rayleigh distributed with user defined sigma parameter.

    """

    def __init__(
        self,
        starMass=1.0,
        occDataPath=None,
        esigma=0.175 / np.sqrt(np.pi / 2.0),
        **specs
    ):
        # set local input attributes and call upstream init
        self.starMass = starMass * u.M_sun
        self.occDataPath = occDataPath
        self.esigma = float(esigma)
        PlanetPopulation.__init__(self, **specs)

        er = self.erange
        self.enorm = np.exp(-er[0] ** 2 / (2.0 * self.esigma**2)) - np.exp(
            -er[1] ** 2 / (2.0 * self.esigma**2)
        )
        self.dist_sma_built = None
        self.dist_radius_built = None
        self.dist_albedo_built = None
        # check occDataPath
        if self.occDataPath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = "NominalOcc_Radius.csv"
            self.occDataPath = os.path.join(classpath, filename)

        assert os.path.exists(
            self.occDataPath
        ), "occurrence rate table not found at {}".format(self.occDataPath)

        # load data
        occData = dict(ascii.read(self.occDataPath))
        self.aData = "a_min" in occData
        self.RData = "R_min" in occData
        # load occurrence rates
        eta2D = []
        for tmp in occData["Occ"]:
            eta2D.append(tmp)
        eta2D = np.array(eta2D)
        # load semi-major axis or period
        if self.aData:
            amin = []
            for tmp in occData["a_min"]:
                amin.append(tmp)
            amin = np.array(amin)
            amax = []
            for tmp in occData["a_max"]:
                amax.append(tmp)
            amax = np.array(amax)
            self.smas = np.hstack((np.unique(amin), np.unique(amax)[-1]))
            len1 = len(self.smas) - 1
            self.arange = np.array([self.smas[0], self.smas[-1]]) * u.AU
        else:
            Pmin = []
            for tmp in occData["P_min"]:
                Pmin.append(tmp)
            Pmin = np.array(Pmin)
            Pmax = []
            for tmp in occData["P_max"]:
                Pmax.append(tmp)
            Pmax = np.array(Pmax)
            self.Ps = np.hstack((np.unique(Pmin), np.unique(Pmax)[-1]))
            len1 = len(self.Ps) - 1
            self.arange = (
                (
                    const.G
                    * self.starMass
                    * (np.array([self.Ps[0], self.Ps[-1]]) * u.day) ** 2
                    / 4.0
                    / np.pi**2
                )
                ** (1.0 / 3.0)
            ).to("AU")
        # load radius or mass
        if self.RData:
            Rmin = []
            for tmp in occData["R_min"]:
                Rmin.append(tmp)
            Rmin = np.array(Rmin)
            Rmax = []
            for tmp in occData["R_max"]:
                Rmax.append(tmp)
            Rmax = np.array(Rmax)
            self.Rs = np.hstack((np.unique(Rmin), np.unique(Rmax)[-1]))
            len2 = len(self.Rs) - 1
            self.Rprange = np.array([self.Rs[0], self.Rs[-1]]) * u.R_earth
            # maximum from data sheets provided
            self.Mprange = np.array([0.08, 4768.0]) * u.earthMass
        else:
            Mmin = []
            for tmp in occData["M_min"]:
                Mmin.append(tmp)
            Mmin = np.array(Mmin)
            Mmax = []
            for tmp in occData["M_max"]:
                Mmax.append(tmp)
            Mmax = np.array(Mmax)
            self.Ms = np.hstack((np.unique(Mmin), np.unique(Mmax)[-1]))
            len2 = len(self.Ms) - 1
            self.Mprange = np.array([self.Ms[0], self.Ms[-1]]) * u.M_earth
            self.Rprange = (
                np.array(
                    [
                        1.008 * 0.08 ** (0.279),
                        17.739 * (0.225 * u.M_jup.to("earthMass")) ** (-0.044),
                    ]
                )
                * u.earthRad
            )
        if self.constrainOrbits:
            self.rrange = self.arange
        else:
            self.rrange = (
                np.array(
                    [
                        self.arange[0].to("AU").value * (1.0 - self.erange[1]),
                        self.arange[1].to("AU").value * (1.0 + self.erange[1]),
                    ]
                )
                * u.AU
            )
        # reshape eta2D array
        self.eta2D = eta2D.reshape((len1, len2))
        self.eta = self.eta2D.sum()

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and planetary radius come from the occurrence rate
        tables and are assumed to be log-uniformly distributed within bins.
        Eccentricity is a Rayleigh distribution. Albedo is dependent on the
        PlanetPhysicalModel but is calculated such that it is independent of
        other parameters.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            tuple:
            a (astropy Quantity array):
                Semi-major axis in units of AU
            e (float ndarray):
                Eccentricity
            p (float ndarray):
                Geometric albedo
            Rp (astropy Quantity array):
                Planetary radius in units of earthRad

        """
        n = self.gen_input_check(n)
        PPMod = self.PlanetPhysicalModel
        # generate semi-major axis and radius samples
        a, Rp = self.gen_sma_radius(n)
        # check for constrainOrbits == True for eccentricity samples
        # and generate eccentricity as Rayleigh distributed
        C1 = np.exp(-self.erange[0] ** 2 / (2.0 * self.esigma**2))
        ar = self.arange.to("AU").value
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array(
                [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
            )
            # clip sma values to sma range
            sma = np.clip(a.to("AU").value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1.0 - ar[0] / sma[sma <= amean]
            elim[sma > amean] = ar[1] / sma[sma > amean] - 1.0
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # constants
            C2 = C1 - np.exp(-(elim**2) / (2.0 * self.esigma**2))
            a = sma * u.AU
        else:
            C2 = self.enorm
        e = self.esigma * np.sqrt(-2.0 * np.log(C1 - C2 * np.random.uniform(size=n)))
        # generate albedo from semi-major axis
        p = PPMod.calc_albedo_from_sma(a, self.prange)

        return a, e, p, Rp

    def gen_sma_radius(self, n):
        """
        Generates semi-major axis and planetary radius samples

        Args:
            n (int):
                number of samples

        Returns:
            tuple:
            a (astropy Quantity array):
                Semi-major axis samples in units of AU
            Rp (astropy Quantity array):
                Planetary radius samples in units of Earth radius
        """

        # semi-major axis--planetary mass data given
        if self.aData:
            # data given as semi-major axis and planetary radius
            # sample Mp distribution first
            Mp_sums = np.cumsum(self.eta2D.sum(axis=0) / self.eta)
            U = np.random.random(n)
            Mp = np.ones(n)
            # sample bins log-uniformly
            for i in range(len(Mp_sums)):
                if i == 0:
                    inds = np.where((U > 0) & (U <= Mp_sums[0]))[0]
                    U2 = np.random.random(len(inds))
                    Mp[inds] = self.Ms[0] * np.exp(U2 * np.log(self.Ms[1] / self.Ms[0]))
                else:
                    inds = np.where((U > Mp_sums[i - 1]) & (U <= Mp_sums[i]))[0]
                    U2 = np.random.random(len(inds))
                    Mp[inds] = self.Ms[i] * np.exp(
                        U2 * np.log(self.Ms[i + 1] / self.Ms[i])
                    )
            # convert to planetary radius
            Rp = self.RpfromM(Mp * u.earthMass)
            # now sample semi-major axis
            a = np.ones(n)  # samples will be placed here
            for i in range(len(self.Ms) - 1):
                # find where the Mp samples lie
                inds = np.where((Mp >= self.Ms[i]) & (Mp <= self.Ms[i + 1]))[0]
                atmp = np.ones(len(inds))
                a_sums = np.cumsum(self.eta2D[:, i]) / self.eta2D[:, i].sum()
                U = np.random.random(len(inds))
                # sample log-uniformly in bins
                for j in range(len(a_sums)):
                    if j == 0:
                        inds2 = np.where((U > 0) & (U <= a_sums[0]))[0]
                        U2 = np.random.random(len(inds2))
                        atmp[inds2] = self.smas[0] * np.exp(
                            U2 * np.log(self.smas[1] / self.smas[0])
                        )
                    else:
                        inds2 = np.where((U > a_sums[j - 1]) & (U <= a_sums[j]))[0]
                        U2 = np.random.random(len(inds2))
                        atmp[inds2] = self.smas[j] * np.exp(
                            U2 * np.log(self.smas[j + 1] / self.smas[j])
                        )
                a[inds] = atmp
            a = a * u.AU
        # Period--planetary radius data given
        else:
            # sum over rows to get distribution on R
            R_pdf = self.eta2D.sum(axis=0) / self.eta
            R_sums = np.cumsum(R_pdf)
            # generate samples on R
            U = np.random.random(n)
            R_samp = np.ones(n)
            # sample bins log-uniformly
            for i in range(len(R_sums)):
                if i == 0:
                    inds = np.where((U > 0) & (U <= R_sums[0]))[0]
                    U2 = np.random.random(len(inds))
                    R_samp[inds] = self.Rs[0] * np.exp(
                        U2 * np.log(self.Rs[1] / self.Rs[0])
                    )
                else:
                    inds = np.where((U > R_sums[i - 1]) & (U <= R_sums[i]))[0]
                    U2 = np.random.random(len(inds))
                    R_samp[inds] = self.Rs[i] * np.exp(
                        U2 * np.log(self.Rs[i + 1] / self.Rs[i])
                    )
            Rp = R_samp * u.earthRad
            # sample period
            P_samp = np.ones(n)  # samples will be placed here
            for i in range(len(self.Rs) - 1):
                # find where the Rp samples lie
                inds = np.where((R_samp >= self.Rs[i]) & (R_samp <= self.Rs[i + 1]))[0]
                Ptmp = np.ones(len(inds))
                P_sums = np.cumsum(self.eta2D[:, i]) / self.eta2D[:, i].sum()
                U = np.random.random(len(inds))
                for j in range(len(P_sums)):
                    if j == 0:
                        inds2 = np.where((U > 0) & (U <= P_sums[0]))[0]
                        U2 = np.random.random(len(inds2))
                        Ptmp[inds2] = self.Ps[0] * np.exp(
                            U2 * np.log(self.Ps[1] / self.Ps[0])
                        )
                    else:
                        inds2 = np.where((U > P_sums[j - 1]) & (U <= P_sums[j]))[0]
                        U2 = np.random.random(len(inds2))
                        Ptmp[inds2] = self.Ps[j] * np.exp(
                            U2 * np.log(self.Ps[j + 1] / self.Ps[j])
                        )
                P_samp[inds] = Ptmp
            # convert period samples to semi-major axis
            a = (
                (const.G * self.starMass * (P_samp * u.day) ** 2 / 4.0 / np.pi**2)
                ** (1.0 / 3.0)
            ).to("AU")

        return a, Rp

    def gen_albedo(self, n):
        """Generate geometric albedo values

        The albedo is determined by sampling the semi-major axis distribution,
        and then calculating the albedo from the physical model.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            float ndarray:
                Planet albedo values

        """
        n = self.gen_input_check(n)
        a, _ = self.gen_sma_radius(n)
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(a, self.prange)

        return p

    def RpfromM(self, M):
        """
        Converts mass to radius using Chen and Kipping

        Args:
            M (astropy Quantity array):
                Planet mass in units of Earth mass

        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of Earth radius
        """

        group1 = np.where(M < 2.04 * u.earthMass)[0]
        group2 = np.where((M >= 2.04 * u.earthMass) & (M < 0.225 * u.M_jup))[0]
        group3 = np.where(M >= 0.225 * u.M_jup)

        Rp = np.ones(len(M))
        Rp[group1] = 1.008 * M[group1].to("earthMass").value ** (0.279)
        Rp[group2] = 0.80811 * M[group2].to("earthMass").value ** (0.589)
        Rp[group3] = 17.739 * M[group3].to("earthMass").value ** (-0.044)

        return Rp * u.R_earth

    def MfromRp(self, Rp):
        """
        Converts mass to radius using Chen and Kipping

        Args:
            Rp (astropy Quantity array):
                Planet mass in units of Earth mass

        Returns:
            astropy Quantity array:
                Planet mass
        """

        group1 = np.where(Rp < 1.23 * u.R_earth)[0]
        group2 = np.where((Rp >= 1.23 * u.R_earth) & (Rp < 9.99 * u.R_earth))[0]
        group3 = np.where((Rp >= 9.99 * u.R_earth))[0]

        M = np.ones(len(Rp))

        M[group1] = np.exp(np.log(Rp[group1].to("earthRad").value / 1.008) / 0.279)
        M[group2] = np.exp(np.log(Rp[group2].to("earthRad").value / 0.80811) / 0.589)
        M[group3] = np.exp(np.log(Rp[group3].to("earthRad").value / 17.739) / -0.044)

        return M * u.M_earth

    def dist_sma(self, a):
        """Probability density function for semi-major axis.

        Note that this is a marginalized distribution.

        Args:
            a (float ndarray):
                Semi-major axis value(s)

        Returns:
            float ndarray:
                Semi-major axis probability density

        """

        # if called for the first time, define distribution for albedo
        if self.dist_sma_built is None:
            agen, _ = self.gen_sma_radius(int(1e6))
            ar = self.arange.to("AU").value
            ap, aedges = np.histogram(
                agen.to("AU").value, bins=2000, range=(ar[0], ar[1]), density=True
            )
            aedges = 0.5 * (aedges[1:] + aedges[:-1])
            aedges = np.hstack((ar[0], aedges, ar[1]))
            ap = np.hstack((0.0, ap, 0.0))
            self.dist_sma_built = interpolate.InterpolatedUnivariateSpline(
                aedges, ap, k=1, ext=1
            )

        f = self.dist_sma_built(a)

        return f

    def dist_radius(self, Rp):
        """Probability density function for planetary radius.

        Note that this is a marginalized distribution.

        Args:
            Rp (float ndarray):
                Planetary radius value(s)

        Returns:
            float ndarray:
                Planetary radius probability density

        """

        # if called for the first time, define distribution for albedo
        if self.dist_radius_built is None:
            _, Rgen = self.gen_sma_radius(int(1e6))
            Rpr = self.Rprange.to("earthRad").value
            Rpp, Rpedges = np.histogram(
                Rgen.to("earthRad").value,
                bins=2000,
                range=(Rpr[0], Rpr[1]),
                density=True,
            )
            Rpedges = 0.5 * (Rpedges[1:] + Rpedges[:-1])
            Rpedges = np.hstack((Rpr[0], Rpedges, Rpr[1]))
            Rpp = np.hstack((0.0, Rpp, 0.0))
            self.dist_radius_built = interpolate.InterpolatedUnivariateSpline(
                Rpedges, Rpp, k=1, ext=1
            )

        f = self.dist_radius_built(Rp)

        return f

    def dist_eccen(self, e):
        """Probability density function for eccentricity

        Args:
            e (float ndarray):
                Eccentricity value(s)

        Returns:
            float ndarray:
                Eccentricity probability density

        """

        # cast to array
        e = np.array(e, ndmin=1, copy=copy_if_needed)

        # Rayleigh distribution sigma
        f = np.zeros(e.shape)
        mask = np.array((e >= self.erange[0]) & (e <= self.erange[1]), ndmin=1)
        f[mask] = (
            e[mask]
            / self.esigma**2
            * np.exp(-e[mask] ** 2 / (2.0 * self.esigma**2))
            / self.enorm
        )

        return f

    def dist_eccen_from_sma(self, e, a):
        """Probability density function for eccentricity constrained by
        semi-major axis, such that orbital radius always falls within the
        provided sma range.

        This provides a Rayleigh distribution between the minimum and
        maximum allowable values.

        Args:
            e (float ndarray):
                Eccentricity values
            a (float ndarray):
                Semi-major axis value in AU. Not an astropy quantity.

        Returns:
            float ndarray:
                Probability density of eccentricity constrained by semi-major
                axis

        """

        # cast a and e to array
        e = np.array(e, ndmin=1, copy=copy_if_needed)
        a = np.array(a, ndmin=1, copy=copy_if_needed)
        # if a is length 1, copy a to make the same shape as e
        if a.ndim == 1 and len(a) == 1:
            a = a * np.ones(e.shape)

        # unitless sma range
        ar = self.arange.to("AU").value
        arcon = np.array(
            [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
        )
        # upper limit for eccentricity given sma
        elim = np.zeros(a.shape)
        amean = np.mean(arcon)
        elim[a <= amean] = 1.0 - ar[0] / a[a <= amean]
        elim[a > amean] = ar[1] / a[a > amean] - 1.0
        elim[elim > self.erange[1]] = self.erange[1]
        elim[elim < self.erange[0]] = self.erange[0]

        # if e and a are two arrays of different size, create a 2D grid
        if a.size not in [1, e.size]:
            elim, e = np.meshgrid(elim, e)

        norm = np.exp(-self.erange[0] ** 2 / (2.0 * self.esigma**2)) - np.exp(
            -(elim**2) / (2.0 * self.esigma**2)
        )
        ins = np.array((e >= self.erange[0]) & (e <= elim), dtype=float, ndmin=1)
        f = np.zeros(e.shape)
        mask = (a >= arcon[0]) & (a <= arcon[1])
        f[mask] = (
            ins[mask]
            * e[mask]
            / self.esigma**2
            * np.exp(-e[mask] ** 2 / (2.0 * self.esigma**2))
            / norm[mask]
        )

        return f

    def dist_albedo(self, p):
        """Probability density function for albedo

        Args:
            p (float ndarray):
                Albedo value(s)

        Returns:
            float ndarray:
                Albedo probability density

        """

        # if called for the first time, define distribution for albedo
        if self.dist_albedo_built is None:
            pgen = self.gen_albedo(int(1e6))
            pr = self.prange
            hp, pedges = np.histogram(
                pgen, bins=2000, range=(pr[0], pr[1]), density=True
            )
            pedges = 0.5 * (pedges[1:] + pedges[:-1])
            pedges = np.hstack((pr[0], pedges, pr[1]))
            hp = np.hstack((0.0, hp, 0.0))
            self.dist_albedo_built = interpolate.InterpolatedUnivariateSpline(
                pedges, hp, k=1, ext=1
            )

        f = self.dist_albedo_built(p)

        return f
