from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
from astropy import units as u
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

        # planet semi-major axis
        self.a = self.PlanetPopulation.gen_sma(self.nPlans)
        # inflated planets have to be moved to tidally locked orbits
        self.a[self.Rp > np.nanmax(self.PlanetPhysicalModel.ggdat['radii'])] = 0.02*u.AU
        # planet eccentricities
        self.e = self.PlanetPopulation.gen_eccentricity(self.nPlans)
        # planet argument of periapse
        self.w = self.PlanetPopulation.gen_w(self.nPlans)   
        # planet longitude of ascending node
        self.O = self.PlanetPopulation.gen_O(self.nPlans)
        # planet inclination
        self.I = self.PlanetPopulation.gen_I(self.nPlans)
        # planet masses
        self.Mp = self.PlanetPhysicalModel.calc_mass_from_radius(self.Rp)
        # planet albedos
        self.p = self.PlanetPhysicalModel.calc_albedo_from_sma(self.a)
        # planet initial positions
        self.r, self.v = self.planet_pos_vel() 
        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(self.TargetList,self.planInds,self.I)


    def planet_to_star(self):
        """Assigns index of star in target star list to each planet
        
        This assumes that self.nPlans will have already been set by sampling the radius
        distribution.

        Attributes updated:
            planInds (ndarray):
                1D numpy array containing indices of the target star to which 
                each planet (each element of the array) belongs
            sysInds (ndarray):
                1D numpy array of indices of the subset of the targetlist with
                planets
        
        """
        
        self.planInds = np.random.randint(low=0,high=self.TargetList.nStars-1,size=self.nPlans)    
        self.sysInds = np.unique(self.planInds)
        
        return 
