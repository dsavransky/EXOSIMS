from EXOSIMS.util.vprint import vprint
import numpy as np
from EXOSIMS.Prototypes.PlanetPhysicalModel import PlanetPhysicalModel


class uniform_albedo(PlanetPhysicalModel):
    """Uniform albedo by sma distribution
    
    This class contains all variables and functions necessary to perform 
    Planet Physical Model Module calculations in exoplanet mission simulation.
    
    Args:
        specs:
            user specified values

    Attributes:
        cachedir (str):
            Path to EXOSIMS cache directory
        whichPlanetPhaseFunction (str or callable):
            planet phase function to use
            
    """

    _modtype = 'PlanetPhysicalModel'

    def __init__(self, prange=[0.1,0.6], **specs):

        PlanetPhysicalModel.__init__(self, **specs)
        self.prange = prange
        #do nothing

    def calc_albedo_from_sma(self,a):
        """
        Helper function for calculating albedo given the semi-major axis.
        The prototype provides only a dummy function that always returns the 
        same value of 0.367.
        
        Args:
            a (astropy Quanitity array):
               Semi-major axis values
        
        Returns:
            p (ndarray):
                Albedo values
        
        """
        p = np.random.uniform(low=self.prange[0],high=self.prange[1],size=a.size)
        
        return p
