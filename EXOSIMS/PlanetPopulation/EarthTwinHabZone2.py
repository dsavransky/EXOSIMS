from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone1 import EarthTwinHabZone1
import numpy as np

class EarthTwinHabZone2(EarthTwinHabZone1):
    """
    Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On eccentric habitable zone orbits (0.7 to 1.5 AU).
    
    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(self, eta=0.1, erange=[0.,0.9], constrainOrbits=True, **specs):
        
        specs['eta'] = eta
        specs['arange'] = [0.7, 1.5]
        specs['erange'] = erange
        specs['Rprange'] = [1,1]
        specs['Mprange'] = [1,1]
        specs['prange'] = [0.367,0.367]
        specs['scaleOrbits'] = True
        specs['constrainOrbits'] = constrainOrbits
        PlanetPopulation.__init__(self, **specs)

