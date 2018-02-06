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
        """Generating universe based on SAG13 planet radius and period sampling.
        
        All parameters except for albedo and mass are sampled, while those are
        calculated via the physical model.
        
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        
        # treat eta as the rate parameter of a Poisson distribution
        targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        plan2star = []
        for j,n in enumerate(targetSystems):
            plan2star = np.hstack((plan2star, [j]*n))
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()                                    # initial mean anomaly
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)   # mass
        
        # The prototype StarCatalog module is made of one single G star at 1pc. 
        # In that case, generate one Jupiter at 5 AU to allow for characterization 
        # testing. Also generates at least one Jupiter if no planet was generated.
        if TL.Name[0] == 'Prototype' or self.nPlans == 0:
            self.plan2star = np.array([0], dtype=int)
            self.sInds = np.unique(self.plan2star)
            self.nPlans = len(self.plan2star)
            self.a = np.array([5.])*u.AU
            self.e = np.array([0.])
            self.I = np.array([0.])*u.deg # face-on
            self.O = np.array([0.])*u.deg
            self.w = np.array([0.])*u.deg
            self.M0 = np.array([0.])*u.deg
            self.Rp = np.array([10.])*u.earthRad
            self.Mp = np.array([300.])*u.earthMass
            self.p = np.array([0.6])