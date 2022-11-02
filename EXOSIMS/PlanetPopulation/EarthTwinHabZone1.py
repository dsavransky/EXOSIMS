from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u


class EarthTwinHabZone1(PlanetPopulation):
    """Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On circular Habitable zone orbits (0.7 to 1.5 AU)

    Note that these values may not be overwritten by user inputs.
    This implementation is intended to enforce this population regardless
    of JSON inputs.
    """

    def __init__(self, eta=0.1, **specs):

        specs["eta"] = eta
        specs["arange"] = [0.7, 1.5]
        specs["erange"] = [0, 0]
        specs["prange"] = [0.367, 0.367]
        specs["Rprange"] = [1, 1]
        specs["Mprange"] = [1, 1]
        specs["scaleOrbits"] = True
        PlanetPopulation.__init__(self, **specs)

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis is uniformly distributed and all other parameters are
        constant.

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
        # generate samples of semi-major axis
        ar = self.arange.to("AU").value
        a = np.random.uniform(low=ar[0], high=ar[1], size=n) * u.AU
        # generate eccentricity
        e = np.zeros((n,))
        # generate samples of geometric albedo
        p = 0.367 * np.ones((n,))
        # generate samples of planetary radius
        Rp = np.ones((n,)) * u.earthRad

        return a, e, p, Rp

    def dist_sma(self, a):
        """Probability density function for uniform semi-major axis distribution in AU


        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.

        Returns:
            float ndarray:
                Semi-major axis probability density

        """

        return self.uniform(a, self.arange.to("AU").value)
