import numpy as np
import astropy.units as u
from astropy.time import Time

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class ExoversesUniverse(SimulatedUniverse):
    def __init__(self, **specs):
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        universe = specs.get("exoverses_universe")
        PPop = self.PlanetPopulation
        TL = self.TargetList

        # Get the number of planets for each system in the exoverses universe
        plan2star = np.array([], dtype=int)
        for i, system in enumerate(universe.systems):
            nPlanets = len(system.planets)
            sInd = np.argwhere(TL.Name == system.star.name)[0][0]
            plan2star = np.hstack((plan2star, [sInd] * nPlanets))
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        self.I = np.zeros(self.nPlans) * u.deg
        self.O = np.zeros(self.nPlans) * u.deg
        self.w = np.zeros(self.nPlans) * u.deg
        self.a = np.zeros(self.nPlans) * u.AU
        self.e = np.zeros(self.nPlans)
        self.Rp = np.zeros(self.nPlans) * u.earthRad
        self.Mp = np.zeros(self.nPlans) * u.earthMass
        self.p = np.zeros(self.nPlans)
        self.M0 = np.zeros(self.nPlans) * u.deg
        abs_planet_ind = 0
        for star_ind in self.sInds:
            system = universe.systems[star_ind]
            for planet in system.planets:
                self.I[abs_planet_ind] = planet.inc.to(u.deg)
                self.O[abs_planet_ind] = planet.W.to(u.deg)
                self.w[abs_planet_ind] = planet.w.to(u.deg)
                self.a[abs_planet_ind] = planet.a.to(u.AU)
                self.e[abs_planet_ind] = planet.e
                self.Rp[abs_planet_ind] = planet.radius.to(u.earthRad)
                self.Mp[abs_planet_ind] = planet.mass.to(u.earthMass)
                self.p[abs_planet_ind] = planet.p
                self.M0[abs_planet_ind] = planet.mean_anom(
                    Time(specs["missionStart"], format="mjd").ravel()
                ).to(u.deg)
                abs_planet_ind += 1
        self.phiIndex = np.ones(self.nPlans, dtype=int) * 2
