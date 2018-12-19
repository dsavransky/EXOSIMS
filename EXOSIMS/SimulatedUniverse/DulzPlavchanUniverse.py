from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np


class DulzPlavchanUniverse(SimulatedUniverse):
    """Simulated Universe module based on Dulz and Plavchan occurrence rates.

    """

    def __init__(self, **specs):

        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generating universe based on Dulz and Plavchan occurrence rate tables.

        """

        PPop = self.PlanetPopulation
        TL = self.TargetList

        # treat eta as the rate parameter of a Poisson distribution
        targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        plan2star = []
        for j, n in enumerate(targetSystems):
            plan2star = np.hstack((plan2star, [j] * n))
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)

        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()  # initial mean anomaly
        self.Mp = PPop.MfromRp(self.Rp)  # mass