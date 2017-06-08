from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone1 import EarthTwinHabZone1
import numpy as np

r"""Population of Earth twins with fixed properties, used for test_BrownCompleteness.

Note [turmon Oct 2016]: Some of the characteristics of this implementation could have been
achieved by just manually supplying specs.  The reason for using this
method (may be?) that changes here force a re-computation of the Completeness
by EXOSIMS, due to its use of a file-hash to notice parameter changes.

Paul Nunez, JPL, Aug. 2016"""

class EarthTwinHabZone3(EarthTwinHabZone1):
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
        #specs['erange'] = erange
        # Reference works properly for zero eccentricity, is only approximate
        # for [0,0.35].  Try [0,0.01].
        specs['erange'] = [0, 0.35] #Paul change
        #specs['erange'] = [0.0, 0.10] #Paul change
        #specs['Rprange'] = [1,1]
        specs['Rprange'] = [1,1] # Paul Change
        specs['Mprange'] = [1,1]
        #specs['prange'] = [0.367,0.367]
        specs['prange'] = [0.33,0.33] #Paul change
        specs['scaleOrbits'] = True
        specs['constrainOrbits'] = constrainOrbits
        PlanetPopulation.__init__(self, **specs)

