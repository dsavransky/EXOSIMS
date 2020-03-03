from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u  

class DulzPlavchanUniverseEarthsOnly(SimulatedUniverse):
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

        a, e, p, Rp = PPop.gen_plan_params(len(plan2star))

        #filter to just Earth candidates
        #modify as needed NB: a has no yet been scaled by \sqrt{L}
        #so effectively everything is at 1 solar luminosity
        inds = (a < 1.67*u.AU) & (a > 0.95*u.AU) & (Rp < 1.4*u.R_earth ) & (Rp >0.8/np.sqrt(a.value)*u.R_earth)  

        self.a = a[inds]
        self.e = e[inds]
        self.p = p[inds]
        self.Rp = Rp[inds]
        plan2star = plan2star[inds]


        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)

        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])


        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        self.gen_M0()  # initial mean anomaly
        self.Mp = PPop.MfromRp(self.Rp)  # mass

