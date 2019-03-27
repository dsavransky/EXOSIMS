from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.PlanetPopulation.EarthTwinHabZone1 import EarthTwinHabZone1
import numpy as np
import astropy.units as u

class EarthTwinHabZone2(EarthTwinHabZone1):
    """
    Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On eccentric habitable zone orbits (0.7 to 1.5 AU).
    
    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(self, eta=0.1, erange=[0.,0.9], constrainOrbits=True, **specs):
        
        specs['erange'] = erange
        specs['constrainOrbits'] = constrainOrbits
        
        # specs being modified in EarthTwinHabZone1
        specs['eta'] = eta
        specs['arange'] = [0.7, 1.5]
        specs['Rprange'] = [1,1]
        specs['Mprange'] = [1,1]
        specs['prange'] = [0.367,0.367]
        specs['scaleOrbits'] = True
        
        PlanetPopulation.__init__(self, **specs)
        
    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)
        
        Semi-major axis and eccentricity are uniformly distributed with all
        other parameters constant.
        
        Args:
            n (integer):
                Number of samples to generate
        
        Returns:
            tuple:
            a (astropy Quantity array):
                Semi-major axis in units of AU
            e (float ndarray):
                Eccentricity
            p (float ndarray):
                Geometric albedo
            Rp (astropy Quantity array):
                Planetary radius in units of earthRad
        
        """
        n = self.gen_input_check(n)
        # generate samples of semi-major axis
        ar = self.arange.to('AU').value
        # check if constrainOrbits == True for eccentricity
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array([ar[0]/(1.-self.erange[0]), ar[1]/(1.+self.erange[0])])
            a = np.random.uniform(low=arcon[0], high=arcon[1], size=n)*u.AU
            tmpa = a.to('AU').value

            # upper limit for eccentricity given sma
            elim = np.zeros(len(a))
            amean = np.mean(ar)
            elim[tmpa <= amean] = 1. - ar[0]/tmpa[tmpa <= amean]
            elim[tmpa > amean] = ar[1]/tmpa[tmpa>amean] - 1.
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
        
            # uniform distribution
            e = np.random.uniform(low=self.erange[0], high=elim, size=n)
        else:
            a = np.random.uniform(low=ar[0], high=ar[1], size=n)*u.AU
            e = np.random.uniform(low=self.erange[0], high=self.erange[1], size=n)
        # generate geometric albedo
        p = 0.367*np.ones((n,))
        # generate planetary radius
        Rp = np.ones((n,))*u.earthRad
        
        return a, e, p, Rp
