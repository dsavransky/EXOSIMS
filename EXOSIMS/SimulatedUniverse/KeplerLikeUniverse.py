from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u

class KeplerLikeUniverse(SimulatedUniverse):
    """
    Simulated universe implementation inteded to work with the Kepler-like
    planetary population implementations.
    
    Args: 
        specs: 
            user specified values
    
    Notes:
        The occurrence rate in these universes is set entirely by the radius
        distribution.
    
    """

    def __init__(self, **specs):
        
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self,**specs):
        """Generates the planetary systems' physical properties. Populates arrays 
        of the orbital elements, albedos, masses and radii of all planets, and 
        generates indices that map from planet to parent star.
        
        All parameters except for albedo and mass are sampled, while those are
        calculated via the physical model.
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        
        # Generate distribution of radii first
        self.Rp = PPop.gen_radius_nonorm(TL.nStars)
        
        # Map planets to target stars
        self.nPlans = self.Rp.size
        self.plan2star = np.random.randint(0,TL.nStars,self.nPlans)
        self.sInds = np.unique(self.plan2star)
        
        self.a, self.e, self.p, _ = PPop.gen_plan_params(self.nPlans)
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        # inflated planets have to be moved to tidally locked orbits
        self.a[self.Rp > np.nanmax(PPMod.ggdat['radii'])] = 0.02*u.AU
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()                                   # initial mean anomaly
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)  # mass