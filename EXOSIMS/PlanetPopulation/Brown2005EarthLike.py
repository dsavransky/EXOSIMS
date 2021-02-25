from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u

class Brown2005EarthLike(PlanetPopulation):
    """
    Population of Earth-Like Planets from Brown 2005 paper
    
    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(self, eta=1, arange=[0.7*np.sqrt(0.83),1.5*np.sqrt(0.83)], erange=[0.,0.35], constrainOrbits=False, **specs):
        #eta is probability of planet occurance in a system. I set this to 1
        specs['erange'] = erange
        specs['constrainOrbits'] = constrainOrbits
        pE = 0.26 # From Brown 2005 #0.33 listed in paper but q=0.26 is listed in the paper in the figure
        # specs being modified in JupiterTwin
        specs['eta'] = eta
        specs['arange'] = arange #*u.AU
        specs['Rprange'] = [1.,1.] #*u.earthRad
        #specs['Mprange'] = [1*MpEtoJ,1*MpEtoJ]
        specs['prange'] = [pE,pE]
        specs['scaleOrbits'] = True

        self.pE = pE

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
        p = self.pE*np.ones((n,))
        # generate planetary radius
        Rp = np.ones((n,))*u.earthRad #*self.RpEtoJ
        
        return a, e, p, Rp


    def gen_radius_nonorm(self, n):
        """Generate planetary radius values in Earth radius.
        This one just generates a bunch of EarthRad
        
        Args:
            n (integer):
                Number of target systems. Total number of samples generated will be,
                on average, n*self.eta
                
        Returns:
            astropy Quantity array:
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)

        Rp = np.ones((n,))
        
        return Rp*u.earthRad

