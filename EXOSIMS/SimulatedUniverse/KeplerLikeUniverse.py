from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
import astropy.constants as const

class KeplerLikeUniverse(SimulatedUniverse):
    """
    Simulated universe implementation inteded to work with the Kepler-like
    planetary population implementations.
    
    Args: 
        \*\*specs: 
            user specified values
    
    Notes:
        The occurrence rate in these universes is set entirely by the radius
        distribution.
    
    """

    def __init__(self, **specs):
        
        SimulatedUniverse.__init__(self, **specs)

    def gen_planetary_systems(self,**specs):
        """
        Generate the planetary systems for the current simulated universe.
        This routine populates arrays of the orbital elements and physical 
        characteristics of all planets, and generates indexes that map from 
        planet to parent star.
        
        All paramters except for albedo and mass are sampled, while those are
        calculated via the physical model.
        """
        
        TL = self.TargetList
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        # Generate distribution of radii first
        self.Rp = PPop.gen_radius_nonorm(TL.nStars)
        
        # Map planets to target stars
        self.nPlans = self.Rp.size
        self.plan2star = np.random.randint(0,TL.nStars,self.nPlans)
        self.sInds = np.unique(self.plan2star)
        
        self.a = PPop.gen_sma(self.nPlans)                  # semi-major axis
        # inflated planets have to be moved to tidally locked orbits
        self.a[self.Rp > np.nanmax(PPMod.ggdat['radii'])] = 0.02*u.AU
        self.e = PPop.gen_eccen(self.nPlans)         # eccentricity
        self.w = PPop.gen_w(self.nPlans)                    # argument of periapsis
        self.O = PPop.gen_O(self.nPlans)                    # longitude of ascending node
        self.I = PPop.gen_I(self.nPlans)                    # inclination
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)      # mass
        self.p = PPMod.calc_albedo_from_sma(self.a)         # albedo
        self.r, self.v = self.planet_pos_vel()              # initial position
        self.d = np.sqrt(np.sum(self.r**2, axis=1))         # planet-star distance
        self.s = np.sqrt(np.sum(self.r[:,0:2]**2, axis=1))  # apparent separation
        
        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(TL,self.plan2star,self.I)

