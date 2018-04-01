from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
import astropy.constants as const

class SAG13Universe(SimulatedUniverse):
    """Simulated Universe module based on SAG13 Planet Population module.
    
    """

    def __init__(self, **specs):
        
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generating universe based on planet radius and period sampling.
        
        This method requires the SAG13 PlanetPopulation module, and
        is calling the following SAG13 attributes: lnRp, lnT, eta2D.
        
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        
        # loop over eta grid, for all log values of radii and periods
        plan2star = []
        radius = []
        period = []
        for i in range(len(PPop.lnRp)-1):
            for j in range(len(PPop.lnT)-1):
                # treat eta as the rate parameter of a Poisson distribution
                targetSystems = np.random.poisson(lam=PPop.eta2D[i,j], size=TL.nStars)
                for m,n in enumerate(targetSystems):
                    plan2star = np.hstack((plan2star,[m]*n))
                    radius = np.hstack((radius,np.exp(np.random.uniform(low=PPop.lnRp[i],
                            high=PPop.lnRp[i+1], size=n)).tolist()))
                    period = np.hstack((period,np.exp(np.random.uniform(low=PPop.lnT[j],
                            high=PPop.lnT[j+1], size=n)).tolist()))
        
        # must generate at least one planet
        if plan2star.size == 0:
            plan2star = np.array([0])
            period = np.array([1.])
            radius = np.array([1.])
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        # sample all of the orbital and physical parameters
        self.Rp = radius*u.earthRad                         # radius
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)      # mass from radius
        self.T = period*u.year                              # period
        mu = const.G*TL.MsTrue[self.plan2star]
        self.a = ((mu*(self.T/(2*np.pi))**2)**(1/3.)).to('AU')  # semi-major axis
        _, self.e, _, _ = PPop.gen_plan_params(self.nPlans)     # eccentricity
        self.p = PPMod.calc_albedo_from_sma(self.a)             # albedo
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)   # orientation angles
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()                                    # initial mean anomaly