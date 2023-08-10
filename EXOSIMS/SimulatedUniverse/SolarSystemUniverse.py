import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class SolarSystemUniverse(SimulatedUniverse):
    """Simulated Universe with copies of the solar system planets assigned to all
    stars"""

    def __init__(self, **specs):

        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generating copies of the solar system around all targets"""

        PPop = self.PlanetPopulation
        assert (
            PPop.__class__.__name__ == "SolarSystem"
        ), "SolarSystemUniverse requires SolarSystem PlanetPopulation"
        TL = self.TargetList

        nPlans = 8 * TL.nStars  # occurrence rate per system is fixed at 8
        self.nPlans = nPlans
        plan2star = np.tile(np.arange(TL.nStars), (8, 1)).T.flatten()
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)

        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(
            self.nPlans,
            commonSystemPlane=self.commonSystemPlane,
            commonSystemPlaneParams=self.commonSystemPlaneParams,
        )
        self.setup_system_planes()

        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)

        self.gen_M0()  # initial mean anomaly
        self.Mp = self.gen_solar_system_planet_mass(
            self.nPlans
        )  # mass #TODO grab from Tables
        self.phiIndex = np.tile(
            np.arange(8), (TL.nStars)
        )  # assign planet phase functions to planets

    def gen_solar_system_planet_mass(self, nPlans):
        """Generated planet masses for each planet
        Args:
            float:
                nPlans, the number of planets

        Returns:
            ndarray:
                Mp_tiled, the masses of each planet in kg
        """

        Mp_orig = (
            np.asarray(
                [
                    3.3022 * 10**23,
                    4.869 * 10**24,
                    5.9742 * 10**24,
                    6.4191 * 10**23,
                    1.8988 * 10**27,
                    5.685 * 10**26,
                    8.6625 * 10**25,
                    1.0278 * 10**26,
                ]
            )
            * u.kg
        )

        # Tile them
        numTiles = int(nPlans / 8)
        Mp_tiled = np.tile(Mp_orig, (numTiles))
        return Mp_tiled
