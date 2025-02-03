from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u


class Guimond2019(PlanetPopulation):
    """
    Population of Earth-Like Planets from Brown 2005 paper

    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(
        self,
        eta=1,
        arange=[0.1, 50.0],
        erange=[0.0, 0.999],
        prange=[0.434, 0.434],
        constrainOrbits=False,
        **specs
    ):
        # prange comes from nowhere
        # eta is probability of planet occurance in a system. I set this to 1
        specs["erange"] = erange
        specs["constrainOrbits"] = constrainOrbits
        # pE = 0.26 # From Brown 2005 #0.33 listed in paper but q=0.26 is listed
        # in the paper in the figure
        # specs being modified in JupiterTwin
        specs["eta"] = eta
        specs["arange"] = arange  # *u.AU
        specs["Rprange"] = [1.0, 1.0]  # *u.earthRad
        # specs['Mprange'] = [1*MpEtoJ,1*MpEtoJ]
        specs["prange"] = prange
        specs["scaleOrbits"] = True

        self.prange = prange

        PlanetPopulation.__init__(self, **specs)

    def loguniform(self, low=0.001, high=1, size=None):
        """
        Args:
            low (float):
                Minimum value
            high (float):
                Maximum value
            size (int, optional):
                number of values to sample

        Returns:
            ~numpy.ndarray:
                of logarithmically distributed values
        """
        return np.exp(np.random.uniform(low, high, size))

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and eccentricity are uniformly distributed with all
        other parameters constant.

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
        a = self.loguniform(low=np.log(ar[0]), high=np.log(ar[1]), size=n) * u.AU
        e = np.random.uniform(low=self.erange[0], high=self.erange[1], size=n)

        # generate geometric albedo
        p = np.random.uniform(low=self.prange[0], high=self.prange[1], size=n)
        # generate planetary radius
        Rp = np.ones((n,)) * u.earthRad  # *self.RpEtoJ

        return a, e, p, Rp

    def gen_radius_nonorm(self, n):
        """Generate planetary radius values in Earth radius.
        This one just generates a bunch of EarthRad

        Args:
            n (integer):
                Number of target systems. Total number of samples generated will be,
                on average, n*self.eta

        Returns:
            astropy Quantity array:
                Planet radius values in units of Earth radius

        """
        n = self.gen_input_check(n)

        Rp = np.ones((n,))

        return Rp * u.earthRad
