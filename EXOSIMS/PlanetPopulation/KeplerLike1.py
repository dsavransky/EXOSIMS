from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import astropy.units as u
import astropy.constants as const
import numpy as np
#from scipy.stats import rv_continuous
from EXOSIMS.util import statsFun 
import scipy.integrate as integrate


class KeplerLike1(PlanetPopulation):
    """
    Population based on Kepler radius distribution with RV-like semi-major axis
    distribution with exponential decay.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
        smaknee (numeric) 
            Location (in AU) of semi-major axis decay point.
        esigma (numeric)
            Sigma value of Rayleigh distribution for eccentricity.
        

    Notes:  
    1. The gen_mass function samples the Radius and calculates the mass from
    there.  Any user-set mass limits are ignored.
    2. The gen_abledo function samples the sma, and then calculates the albedos
    from there. Any user-set albedo limits are ignored.
    3. The Rrange is fixed to (1,22.6) R_Earth and cannot be overwritten by user
    settings (the JSON input will be ignored) 
    4. The radius piece-wise distribution provides the normalization required to
    get the proper overall eta.  The gen_radius method provided here normalizes
    in order to return exactly the number of samples requested.  A second method
    (gen_radius_nonorm) is provided for generating the simulated universe
    population. The latter assumes a poisson distribution for occurences in each
    bin.
    5.  Eccentricity is assumed to be Rayleigh distributed with a user-settable 
    sigma parameter (defaults to 0.25).

    """

    def __init__(self, smaknee=30., esigma=0.25, **specs):

        PlanetPopulation.__init__(self, Rrange=[1,22.6], prange=[0.083,0.882], **specs)

        assert (smaknee >= self.arange[0].to('AU').value) and \
               (smaknee <= self.arange[1].to('AU').value), \
               "sma knee value must be in sma range."

        #define sma distribution
        self.smaknee = smaknee
        self.smadist = lambda x,s0=self.smaknee: x**(-0.62)*np.exp(-((x/s0)**2))
        
        #define Rayleigh eccentricity distribution
        self.esigma = esigma
        self.edist = lambda x,sigma=self.esigma: (x/sigma**2)*np.exp(-x**2/(2.*sigma**2))

        #define Kepler radius distribution
        Rs = np.array([1,1.4,2.0,2.8,4.0,5.7,8.0,11.3,16,22.6]) #Earth Radii
        Rvals85 = np.array([0.1555, 0.1671, 0.1739, 0.0609, 0.0187, 0.0071, 0.0102, 0.0049, 0.0014])
        a85 = (((85*u.day/2/np.pi)**2*const.M_sun*const.G)**(1./3)).to('AU') #sma of 85 days
        fac1 = integrate.quad(self.smadist,0,a85.value)[0]
        Rvals = integrate.quad(self.smadist,0,self.arange[1].to('AU').value)[0]*(Rvals85/fac1)
        Rvals[5:] *= 2.5 #account for longer orbital baseline data

        self.Rs = Rs
        self.Rvals = Rvals


    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        Samples a power law distribution with exponential turn-off 
        determined by class attribute smaknee
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity units AU)

        """
        n = self.gen_input_check(n)

        return statsFun.simpSample(self.smadist, n, self.arange[0].to('AU').value,\
                self.arange[1].to('AU').value)*self.arange.unit

    def gen_radius(self,n):
        """Generate planetary radius values in km
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.

        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            R (astropy Quantity units km)

        """

        n = self.gen_input_check(n)

        R = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = int(np.ceil(n*self.Rvals[j]/np.sum(self.Rvals)))
            R = np.hstack((R, np.exp(np.log(self.Rs[j])+\
                    (np.log(self.Rs[j+1])-np.log(self.Rs[j]))*\
                    np.random.uniform(size=nsamp))))

        if len(R) > n:
            R = R[np.random.choice(range(len(R)),size=n,replace=False)]

        return (R*const.R_earth).to('km')

    def gen_radius_nonorm(self,n):
        """Generate planetary radius values in km
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.  The rates in the bins determine the overall
        occurrence rates of all planets.

        Args:
            n (numeric):
                Number of targets to generate systems about

        Returns:
            R (astropy Quantity units km)

        """

        n = self.gen_input_check(n)

        R = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = np.random.poisson(lam=self.Rvals[j]*n)  
            R = np.hstack((R, np.exp(np.log(self.Rs[j])+\
                    (np.log(self.Rs[j+1])-np.log(self.Rs[j]))*\
                    np.random.uniform(size=nsamp))))

        return (R*const.R_earth).to('km')


    def gen_eccentricity(self, n):
        """Generate eccentricity values

        Rayleigh distribution, as in Kipping et. al (2013)
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            e (numpy ndarray)

        """
        n = self.gen_input_check(n)
      
        return statsFun.simpSample(self.edist, n, self.erange[0],self.erange[1])


    def gen_albedo(self, n):
        """Generate geometric albedo values

        The albedo is determined by sampling the semi-major axis distribution, 
        and then calculating the albedo from the physical model.

        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            p (numpy ndarray)

        """
        n = self.gen_input_check(n)

        atmp = self.gen_sma(n)
        
        return self.PlanetPhysicalModel.calc_albedo_from_sma(atmp)


    def gen_mass(self, n):
        """Generate planetary mass values in kg
        
        The mass is determined by sampling the radius and then calculating the
        mass from the physical model.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity units kg)

        """
        n = self.gen_input_check(n)
       
        Rtmp = self.gen_radius(n)

        return self.PlanetPhysicalModel.calc_mass_from_radius(Rtmp)
