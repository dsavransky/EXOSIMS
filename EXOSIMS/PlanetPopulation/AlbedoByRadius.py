from EXOSIMS.PlanetPopulation.SAG13 import SAG13
import astropy.units as u
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class AlbedoByRadius(SAG13):
    """Planet Population module based on SAG13 occurrence rates.

    NOTE: This assigns constant albedo based on radius ranges.

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
        ps (float nx1 ndarray):
            Constant geometric albedo values.
        Rb (float (n-1)x1 ndarray):
            Planetary radius break points for albedos in earthRad.
        Rbs (float (n+1)x1 ndarray):
            Planetary radius break points with 0 padded on left and np.inf
            padded on right

    """

    def __init__(
        self,
        SAG13coeffs=[[0.38, -0.19, 0.26, 0.0], [0.73, -1.18, 0.59, 3.4]],
        SAG13starMass=1.0,
        Rprange=[2 / 3.0, 17.0859375],
        arange=[0.09084645, 1.45354324],
        ps=[0.2, 0.5],
        Rb=[1.4],
        **specs
    ):

        # pop required values back into specs, set input attributes and call upstream
        # init
        self.ps = np.array(ps, ndmin=1, copy=copy_if_needed)
        self.Rb = np.array(Rb, ndmin=1, copy=copy_if_needed)
        specs["prange"] = [np.min(ps), np.max(ps)]
        SAG13.__init__(
            self,
            SAG13coeffs=SAG13coeffs,
            SAG13starMass=SAG13starMass,
            Rprange=Rprange,
            arange=arange,
            **specs
        )

        # check to ensure proper inputs
        assert (
            len(self.ps) - len(self.Rb) == 1
        ), "input albedos must have one more element than break radii"
        self.Rbs = np.hstack((0.0, self.Rb, np.inf))

        # albedo is constant for planetary radius range
        self.pfromRp = True

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and planetary radius are jointly distributed.
        Eccentricity is a Rayleigh distribution. Albedo is a constant value
        based on planetary radius.

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
        # generate albedo from planetary radius
        p = self.get_p_from_Rp(Rp)

        return a, e, p, Rp

    def get_p_from_Rp(self, Rp):
        """Generate constant albedos for radius ranges

        Args:
            Rp (astropy Quantity array):
                Planetary radius with units of earthRad

        Returns:
            float ndarray:
                Albedo values

        """
        Rp = np.array(Rp.to("earthRad").value, ndmin=1, copy=copy_if_needed)
        p = np.zeros(Rp.shape)
        for i in range(len(self.Rbs) - 1):
            mask = np.where((Rp >= self.Rbs[i]) & (Rp < self.Rbs[i + 1]))
            p[mask] = self.ps[i]

        return p
