from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
from astropy import constants as const

class KeplerLikeUniverse(SimulatedUniverse):
    """
    Simulated universe implementation inteded to work with the Kepler-like
    planetary population implementations.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 

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

        # Generate distribution of radii first
        self.Rp = self.PlanetPopulation.gen_radius_nonorm(self.TargetList.nStars)
        self.nPlans = self.Rp.size
        # Map planets to target stars
        self.planet_to_star()

        PPop = self.PlanetPopulation
        PPhMod = self.PlanetPhysicalModel
        self.a = PPop.gen_sma(self.nPlans)                  # semi-major axis
        # inflated planets have to be moved to tidally locked orbits
        self.a[self.Rp > np.nanmax(self.PlanetPhysicalModel.ggdat['radii'])] = 0.02*u.AU
        self.e = PPop.gen_eccentricity(self.nPlans)         # eccentricity
        self.w = PPop.gen_w(self.nPlans)                    # argument of periapsis
        self.O = PPop.gen_O(self.nPlans)                    # longitude of ascending node
        self.I = PPop.gen_I(self.nPlans)                    # inclination
        self.Mp = PPhMod.calc_mass_from_radius(self.Rp)     # mass
        self.p = PPhMod.calc_albedo_from_sma(self.a)        # albedo
        self.r, self.v = self.planet_pos_vel()              # initial position
        self.d = np.sqrt(np.sum(self.r**2, axis=1))         # planet-star distance
        self.s = np.sqrt(np.sum(self.r[:,0:2]**2, axis=1))  # apparent separation

        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(self.TargetList,self.plan2star,self.I)

    def planet_to_star(self):
        """Assigns index of star in target star list to each planet
        
        This assumes that self.nPlans will have already been set by sampling the radius
        distribution.

        Attributes updated:
            plan2star (ndarray):
                1D numpy array containing indices of the target star to which 
                each planet (each element of the array) belongs
            sInds (ndarray):
                1D numpy array of indices of the subset of the targetlist with
                planets
        
        """
        
        self.plan2star = np.random.randint(low=0,high=self.TargetList.nStars-1,size=self.nPlans)
        self.sInds = np.unique(self.plan2star)
        
        return
