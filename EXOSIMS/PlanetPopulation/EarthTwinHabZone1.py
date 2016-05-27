from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np

class EarthTwinHabZone1(PlanetPopulation):
    """
    Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On circular Habitable zone orbits (0.7 to 1.5 AU)
    
    Note that these values may not be overwritten by user inputs.  
    This implementation is intended to enforce this population regardless
    of JSON inputs.
    """

    def __init__(self, eta=0.1, **specs):
        
        specs['eta'] = eta
        specs['arange'] = [0.7, 1.5]
        specs['erange'] = [0,0]
        specs['Rrange'] = [1,1]
        specs['Mprange'] = [1,1]
        specs['prange'] = [0.367,0.367]
        specs['scaleOrbits'] = True
        PlanetPopulation.__init__(self, **specs)

    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        The Earth-twin population assumes a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity units AU)
        
        """
        n = self.gen_input_check(n)
        v = self.arange.value
        vals = np.random.uniform(low=v[0],high=v[1],size=n)
        
        return vals*self.arange.unit

