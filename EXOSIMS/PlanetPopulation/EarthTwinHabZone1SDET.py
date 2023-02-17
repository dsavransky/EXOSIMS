from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u


class EarthTwinHabZone1SDET(PlanetPopulation):
    """Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On circular Habitable zone orbits (0.7 to 1.5 AU)

    .. warning::

        Note that these values may not be overwritten by user inputs.
        This implementation is intended to enforce this population regardless
        of the :ref:`sec:inputspec`.

    Args:
        eta (float):
            Occurrence rate.
        specs (dict):
            :ref:`sec:inputspec`
    """

    def __init__(self, eta=0.1, **specs):

        specs["eta"] = eta
        specs["arange"] = [0.95, 1.67]
        specs["erange"] = [0, 0]
        specs["prange"] = [0.2, 0.2]
        specs["Rprange"] = [0.61905858602731, 1.4]
        specs["Mprange"] = [1, 1]
        specs["scaleOrbits"] = True
        PlanetPopulation.__init__(self, **specs)

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis is uniformly distributed and all other parameters are
        constant.

        Args:
            n (int):
                Number of samples to generate

        Returns:
            tuple:
                a (astropy.units.Quantity numpy.ndarray):
                    Semi-major axis in units of AU
                e (float numpy.ndarray):
                    Eccentricity
                p (float numpy.ndarray):
                    Geometric albedo
                Rp (astropy.units.Quantity numpy.ndarray):
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
        # Rr = self.Rprange.to("earthRad").value
        Rp = np.ones((n,)) * u.earthRad

        return a, e, p, Rp

    def dist_sma(self, a):
        """Probability density function for uniform semi-major axis distribution in AU


        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.

        Returns:
            float numpy.ndarray:
                Semi-major axis probability density

        """

        return self.uniform(a, self.arange.to("AU").value)
