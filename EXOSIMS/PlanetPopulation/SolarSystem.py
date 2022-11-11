from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u


class SolarSystem(PlanetPopulation):
    """
    Population of Earth-Like Planets from Brown 2005 paper

    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(self, prange=[0.1, 0.7], Rprange=[0.01, 30.0], **specs):

        specs["prange"] = prange  # adding so this module passes unittests
        specs["Rprange"] = Rprange  # adding so this module passes unittests

        PlanetPopulation.__init__(self, **specs)

    def gen_plan_params(self, nPlans):
        """Values taken From mallama2018PlantProperties Seidlemenn 1992
        Args:
            float:
                nPlans, the number of planets
        """

        # mercury,venus,earth,mars,jupiter,saturn,uranus,neptune
        R_orig = (
            np.asarray(
                [
                    2439.7 * 1000.0,
                    6051.8 * 1000.0,
                    6371.0 * 1000.0,
                    3389.92 * 1000.0,
                    69911.0 * 1000.0,
                    58232.0 * 1000.0,
                    25362.0 * 1000.0,
                    24622.0 * 1000.0,
                ]
            )
            * u.m
        )
        a_orig = (
            np.asarray(
                [
                    57.91 * 10.0**9.0,
                    108.21 * 10.0**9.0,
                    149.60 * 10.0**9.0,
                    227.92 * 10.0**9.0,
                    778.57 * 10.0**9.0,
                    1433.53 * 10.0**9.0,
                    2872.46 * 10.0**9.0,
                    4495.0 * 10.0**9.0,
                ]
            )
            * u.m
        )
        p_orig = np.asarray([0.142, 0.689, 0.434, 0.150, 0.538, 0.499, 0.488, 0.442])
        e_orig = np.ones(8) * 0.01

        R_orig = R_orig.to("earthRad")
        a_orig = a_orig.to("AU")

        # Tile them
        numTiles = int(nPlans / 8)
        R_tiled = np.tile(R_orig, (numTiles))
        a_tiled = np.tile(a_orig, (numTiles))
        p_tiled = np.tile(p_orig, (numTiles))
        e_tiled = np.tile(e_orig, (numTiles))

        return a_tiled, e_tiled, p_tiled, R_tiled
