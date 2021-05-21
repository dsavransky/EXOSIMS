from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
from EXOSIMS.util.eccanom import eccanom
import astropy.units as u
import astropy.constants as const

class SolarSystemUniverse(SimulatedUniverse):
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
        
        
        nPlans = 8*TL.nStars #occurrence rate per system is fixed at 8
        self.nPlans = nPlans
        plan2star = np.ones(nPlans)*8
        self.plan2star = plan2star.astype(int)
        
        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        if self.commonSystemInclinations == True: #OVERWRITE I with TL.I+dI
            self.I = TL.I[self.plan2star]

        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)

        self.gen_M0()                                    # initial mean anomaly
        self.Mp = self.gen_solar_system_planet_mass(self.nPlans)   # mass #TODO grab from Tables
        self.phiIndex = np.tile(np.arange(8),(TL.nStars)) #assign planet phase functions to planets

    def gen_solar_system_planet_mass(self,nPlans):
        """ Generated planet masses for each planet
        Args:
            float:
                nPlans, the number of planets
                
        Returns:
            ndarray:
                Mp_tiled, the masses of each planet in kg
        """

        Mp_orig = np.asarray([3.3022*10**23,4.869*10**24,5.9742*10**24,6.4191*10**23,1.8988*10**27,5.685*10**26,8.6625*10**25,1.0278*10**26])*u.kg
        
        #Tile them
        numTiles = int(nPlans/8)
        Mp_tiled = np.tile(Mp_orig,(numTiles))
        return Mp_tiled

