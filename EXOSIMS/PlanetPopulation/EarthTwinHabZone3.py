from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u


class EarthTwinHabZone3(PlanetPopulation):
    """Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On circular Habitable zone orbits (0.75 to 1.77 AU)

    Note that these values may not be overwritten by user inputs.
    This implementation is intended to enforce this population regardless
    of JSON inputs.
    """

    def __init__(self, eta=0.1, **specs):

        specs["eta"] = eta
        specs["arange"] = [0.75, 1.77]
        specs["erange"] = [0, 0]
        specs["prange"] = [0.2, 0.2]
        specs["Rprange"] = [1, 1]
        specs["Mprange"] = [1, 1]
        specs["scaleOrbits"] = True
        PlanetPopulation.__init__(self, **specs)

    def gen_sma(self, n):
        """Generate semi-major axis values in AU

        The Earth-twin population assumes a uniform distribution between the minimum and
        maximum values.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            a (astropy Quantity array):
                Semi-major axis values in units of AU

        """
        n = self.gen_input_check(n)
        ar = self.arange.to("AU").value
        a = np.random.uniform(low=ar[0], high=ar[1], size=n) * u.AU

        return a

    def dist_sma(self, a):
        """Probability density function for uniform semi-major axis distribution in AU


        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.

        Returns:
            f (float ndarray):
                Semi-major axis probability density

        """

        return self.uniform(a, self.arange.to("AU").value)
