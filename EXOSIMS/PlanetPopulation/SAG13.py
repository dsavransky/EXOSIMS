from EXOSIMS.PlanetPopulation.KeplerLike2 import KeplerLike2
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate
from EXOSIMS.util._numpy_compat import copy_if_needed


class SAG13(KeplerLike2):
    """Planet Population module based on SAG13 occurrence rates.

    This is the current working model based on averaging multiple studies.
    These do not yet represent official scientific values.

    Attributes:
        SAG13coeffs (float 4x2 ndarray):
            Coefficients used by the SAG13 broken power law. The 4 lines
            correspond to Gamma, alpha, beta, and the minimum radius.
        Gamma (float ndarray):
            Gamma coefficients used by SAG13 broken power law.
        alpha (float ndarray):
            Alpha coefficients used by SAG13 broken power law.
        beta (float ndarray):
            Beta coefficients used by SAG13 broken power law.
        Rplim (float ndarray):
            Minimum radius used by SAG13 broken power law.
        SAG13starMass (astropy Quantity):
            Assumed stellar mass corresponding to the given set of coefficients.
        mu (astropy Quantity):
            Gravitational parameter associated with SAG13starMass.
        Ca (float 2x1 ndarray):
            Constants used for sampling.

    """

    def __init__(
        self,
        SAG13coeffs=[[0.38, -0.19, 0.26, 0.0], [0.73, -1.18, 0.59, 3.4]],
        SAG13starMass=1.0,
        Rprange=[2 / 3.0, 17.0859375],
        arange=[0.09084645, 1.45354324],
        **specs
    ):

        # first initialize with KeplerLike constructor
        specs["Rprange"] = Rprange
        specs["arange"] = arange
        # load SAG13 star mass in solMass: 1.3 (F), 1 (G), 0.70 (K), 0.35 (M)
        self.SAG13starMass = float(SAG13starMass) * u.solMass
        self.mu = const.G * self.SAG13starMass

        # load SAG13 coefficients (Gamma, alpha, beta, Rplim)
        self.SAG13coeffs = np.array(SAG13coeffs, dtype=float)
        assert self.SAG13coeffs.ndim <= 2, "SAG13coeffs array dimension must be <= 2."
        # if only one row of coefficients, make sure the forth element
        # (minimum radius) is set to zero
        if self.SAG13coeffs.ndim == 1:
            self.SAG13coeffs = np.array(np.append(self.SAG13coeffs[:3], 0.0), ndmin=2)
        # make sure the array is of shape (4, n) where the forth row
        # contains the minimum radius values (broken power law)
        if len(self.SAG13coeffs) != 4:
            self.SAG13coeffs = self.SAG13coeffs.T
        assert len(self.SAG13coeffs) == 4, "SAG13coeffs array must have 4 rows."
        # sort by minimum radius
        self.SAG13coeffs = self.SAG13coeffs[:, np.argsort(self.SAG13coeffs[3, :])]

        # split out SAG13 coeffs
        self.Gamma = self.SAG13coeffs[0, :]
        self.alpha = self.SAG13coeffs[1, :]
        self.beta = self.SAG13coeffs[2, :]
        self.Rplim = np.append(self.SAG13coeffs[3, :], np.inf)

        KeplerLike2.__init__(self, **specs)

        # intermediate function
        m = self.mu.to("AU3/year2").value
        ftmp = (
            lambda x, b, m=m, ak=self.smaknee: (2.0 * np.pi * np.sqrt(x**3 / m))
            ** (b - 1.0)
            * (3.0 * np.pi * np.sqrt(x / m))
            * np.exp(-((x / ak) ** 3))
        )
        # intermediate constants used elsewhere
        self.Ca = np.zeros((2,))
        for i in range(2):
            self.Ca[i] = integrate.quad(
                ftmp,
                self.arange[0].to("AU").value,
                self.arange[1].to("AU").value,
                args=(self.beta[i],),
            )[0]

        # set up samplers for sma and Rp
        # probability density function of sma given Rp < Rplim[1]
        f_sma_given_Rp1 = lambda a, beta=self.beta[0], m=m, C=self.Ca[
            0
        ], smaknee=self.smaknee: self.dist_sma_given_radius(a, beta, m, C, smaknee)
        # sampler for Rp < Rplim:
        # unitless sma range
        ar = self.arange.to("AU").value
        self.sma_sampler1 = InverseTransformSampler(f_sma_given_Rp1, ar[0], ar[1])
        # probability density function of sma given Rp > Rplim[1]
        f_sma_given_Rp2 = lambda a, beta=self.beta[1], m=m, C=self.Ca[
            1
        ], smaknee=self.smaknee: self.dist_sma_given_radius(a, beta, m, C, smaknee)
        self.sma_sampler2 = InverseTransformSampler(f_sma_given_Rp2, ar[0], ar[1])

        self.Rp_sampler = InverseTransformSampler(
            self.dist_radius,
            self.Rprange[0].to("earthRad").value,
            self.Rprange[1].to("earthRad").value,
        )

        # determine eta
        if self.Rprange[1].to("earthRad").value < self.Rplim[1]:
            self.eta = (
                self.Gamma[0]
                * (
                    self.Rprange[1].to("earthRad").value ** self.alpha[0]
                    - self.Rprange[0].to("earthRad").value ** self.alpha[0]
                )
                / self.alpha[0]
                * self.Ca[0]
            )
        elif self.Rprange[0].to("earthRad").value > self.Rplim[1]:
            self.eta = (
                self.Gamma[1]
                * (
                    self.Rprange[1].to("earthRad").value ** self.alpha[1]
                    - self.Rprange[0].to("earthRad").value ** self.alpha[1]
                )
                / self.alpha[1]
                * self.Ca[1]
            )
        else:
            self.eta = (
                self.Gamma[0]
                * (
                    self.Rplim[1] ** self.alpha[0]
                    - self.Rprange[0].to("earthRad").value ** self.alpha[0]
                )
                / self.alpha[0]
                * self.Ca[0]
            )
            self.eta += (
                self.Gamma[1]
                * (
                    self.Rprange[1].to("earthRad").value ** self.alpha[1]
                    - self.Rplim[1] ** self.alpha[1]
                )
                / self.alpha[1]
                * self.Ca[1]
            )

        self._outspec["eta"] = self.eta

        # populate _outspec with SAG13 specific attributes
        self._outspec["SAG13starMass"] = self.SAG13starMass.to("solMass").value
        self._outspec["SAG13coeffs"] = self.SAG13coeffs

    def gen_radius_sma(self, n):
        """Generate radius values in earth radius and semi-major axis values in AU.

        This method is called by gen_radius and gen_sma.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            tuple:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
            a (astropy Quantity array):
                Semi-major axis values in units of AU

        """

        Rp = self.Rp_sampler(n)
        a = np.zeros(Rp.shape)
        a[Rp < self.Rplim[1]] = self.sma_sampler1(len(Rp[Rp < self.Rplim[1]]))
        a[Rp >= self.Rplim[1]] = self.sma_sampler2(len(Rp[Rp >= self.Rplim[1]]))
        Rp = Rp * u.earthRad
        a = a * u.AU

        return Rp, a

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and planetary radius are jointly distributed.
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
        # generate semi-major axis and planetary radius samples
        Rp, a = self.gen_radius_sma(n)

        # check for constrainOrbits == True for eccentricity samples
        # constants
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
            # additional constant
            C2 = C1 - np.exp(-(elim**2) / (2.0 * self.esigma**2))
            a = sma * u.AU
        else:
            C2 = self.enorm
        e = self.esigma * np.sqrt(-2.0 * np.log(C1 - C2 * np.random.uniform(size=n)))
        # generate albedo from semi-major axis
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(a, self.prange)

        return a, e, p, Rp

    def dist_sma_radius(self, a, R):
        """Joint probability density function for semi-major axis (AU) and
        planetary radius in Earth radius.

        This method performs a change of variables on the SAG13 broken power
        law (originally in planetary radius and period).

        Args:
            a (float ndarray):
                Semi-major axis values in AU. Not an astropy quantity
            R (float ndarray):
                Planetary radius values in Earth radius. Not an astropy quantity

        Returns:
            float ndarray:
                Joint (semi-major axis and planetary radius) probability density
                matrix of shape (len(R),len(a))

        """
        # cast to arrays
        a = np.array(a, ndmin=1, copy=copy_if_needed)
        R = np.array(R, ndmin=1, copy=copy_if_needed)

        assert (
            a.shape == R.shape
        ), "input semi-major axis and planetary radius must have same shape"

        mu = self.mu.to("AU3/year2").value

        f = np.zeros(a.shape)
        mask1 = (
            (R < self.Rplim[1])
            & (R > self.Rprange[0].value)
            & (R < self.Rprange[1].value)
            & (a > self.arange[0].value)
            & (a < self.arange[1].value)
        )
        mask2 = (
            (R > self.Rplim[1])
            & (R > self.Rprange[0].value)
            & (R < self.Rprange[1].value)
            & (a > self.arange[0].value)
            & (a < self.arange[1].value)
        )

        # for R < boundary radius
        f[mask1] = self.Gamma[0] * R[mask1] ** (self.alpha[0] - 1.0)
        f[mask1] *= (2.0 * np.pi * np.sqrt(a[mask1] ** 3 / mu)) ** (self.beta[0] - 1.0)
        f[mask1] *= (3.0 * np.pi * np.sqrt(a[mask1] / mu)) * np.exp(
            -((a[mask1] / self.smaknee) ** 3)
        )
        f[mask1] /= self.eta

        # for R > boundary radius
        f[mask2] = self.Gamma[1] * R[mask2] ** (self.alpha[1] - 1.0)
        f[mask2] *= (2.0 * np.pi * np.sqrt(a[mask2] ** 3 / mu)) ** (self.beta[1] - 1.0)
        f[mask2] *= (3.0 * np.pi * np.sqrt(a[mask2] / mu)) * np.exp(
            -((a[mask2] / self.smaknee) ** 3)
        )
        f[mask2] /= self.eta

        return f

    def dist_sma(self, a):
        """Marginalized probability density function for semi-major axis in AU.

        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.

        Returns:
            float ndarray:
                Semi-major axis probability density

        """
        # cast to array
        a = np.array(a, ndmin=1, copy=copy_if_needed)
        # unitless sma range
        ar = self.arange.to("AU").value
        mu = self.mu.to("AU3/year2").value
        f = np.zeros(a.shape)
        mask = np.array((a >= ar[0]) & (a <= ar[1]), ndmin=1)

        Rmin = self.Rprange[0].to("earthRad").value
        Rmax = self.Rprange[1].to("earthRad").value

        f[mask] = (3.0 * np.pi * np.sqrt(a[mask] / mu)) * np.exp(
            -((a[mask] / self.smaknee) ** 3)
        )

        if Rmin < self.Rplim[1] and Rmax < self.Rplim[1]:
            C1 = (
                self.Gamma[0]
                * (Rmax ** self.alpha[0] - Rmin ** self.alpha[0])
                / self.alpha[0]
            )
            f[mask] *= C1 * (2.0 * np.pi * np.sqrt(a[mask] ** 3 / mu)) ** (
                self.beta[0] - 1.0
            )
        elif Rmin > self.Rplim[1] and Rmax > self.Rplim[1]:
            C2 = (
                self.Gamma[1]
                * (Rmax ** self.alpha[1] - Rmin ** self.alpha[1])
                / self.alpha[1]
            )
            f[mask] *= C2 * (2.0 * np.pi * np.sqrt(a[mask] ** 3 / mu)) ** (
                self.beta[1] - 1.0
            )
        else:
            C1 = (
                self.Gamma[0]
                * (self.Rplim[1] ** self.alpha[0] - Rmin ** self.alpha[0])
                / self.alpha[0]
            )
            C2 = (
                self.Gamma[1]
                * (Rmax ** self.alpha[1] - self.Rplim[1] ** self.alpha[1])
                / self.alpha[1]
            )
            f[mask] *= C1 * (2.0 * np.pi * np.sqrt(a[mask] ** 3 / mu)) ** (
                self.beta[0] - 1.0
            ) + C2 * (2.0 * np.pi * np.sqrt(a[mask] ** 3 / mu)) ** (self.beta[1] - 1.0)

        f /= self.eta

        return f

    def dist_radius(self, Rp):
        """Marginalized probability density function for planetary radius in
        Earth radius.

        Args:
            Rp (float ndarray):
                Planetary radius value(s) in Earth radius. Not an astropy quantity.

        Returns:
            float ndarray:
                Planetary radius probability density

        """

        # cast Rp to array
        Rp = np.array(Rp, ndmin=1, copy=copy_if_needed)
        f = np.zeros(Rp.shape)
        # unitless Rp range
        Rr = self.Rprange.to("earthRad").value

        mask1 = np.array((Rp >= Rr[0]) & (Rp <= self.Rplim[1]), ndmin=1)
        mask2 = np.array((Rp >= self.Rplim[1]) & (Rp <= Rr[1]), ndmin=1)

        masks = [mask1, mask2]
        for i in range(2):
            f[masks[i]] = (
                self.Gamma[i]
                * Rp[masks[i]] ** (self.alpha[i] - 1.0)
                * self.Ca[i]
                / self.eta
            )

        return f

    def dist_sma_given_radius(self, a, beta, m, C, smaknee):
        """Conditional probability density function of semi-major axis given
        planetary radius.

        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity
            beta (float):
                Exponent for distribution
            m (float):
                Gravitational parameter (AU3/year2)
            C (float):
                Normalization for distribution
            smaknee (float):
                Coefficient for decay

        Returns:
            float ndarray:
                Probability density

        """
        # cast a to array
        a = np.array(a, ndmin=1, copy=copy_if_needed)
        ar = self.arange.to("AU").value
        mask = np.array((a >= ar[0]) & (a <= ar[1]), ndmin=1)
        f = np.zeros(a.shape)
        f[mask] = (
            (2.0 * np.pi * np.sqrt(a[mask] ** 3 / m)) ** (beta - 1.0)
            * (3.0 * np.pi * np.sqrt(a[mask] / m))
            * np.exp(-((a / smaknee) ** 3))
            / C
        )

        return f
