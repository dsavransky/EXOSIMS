from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.util import statsFun 
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate


class KeplerLike1(PlanetPopulation):
    """
    Population based on Kepler radius distribution with RV-like semi-major axis
    distribution with exponential decay.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
        smaknee (float):
            Location (in AU) of semi-major axis decay point (knee).
        esigma (float):
            Sigma value of Rayleigh distribution for eccentricity.
        

    Notes:  
    1. The gen_mass function samples the Radius and calculates the mass from
    there.  Any user-set mass limits are ignored.
    2. The gen_albedo function samples the sma, and then calculates the albedos
    from there. Any user-set albedo limits are ignored.
    3. The Rprange is fixed to (1,22.6) R_Earth and cannot be overwritten by user
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

    def __init__(self, smaknee=30, esigma=0.25, **specs):
        
        specs['prange'] = [0.083,0.882]
        specs['Rprange'] = [1,22.6]
        PlanetPopulation.__init__(self, **specs)
        
        assert (smaknee >= self.arange[0].to('AU').value) and \
               (smaknee <= self.arange[1].to('AU').value), \
               "sma knee value must be in sma range."
        
        #define sma distribution
        self.smaknee = float(smaknee)
        self.smadist1 = lambda x,s0=self.smaknee: x**(-0.62)*np.exp(-((x/s0)**2))
        self.smanorm = integrate.quad(self.smadist1, self.arange.min().to('AU').value, self.arange.max().to('AU').value)[0]        
        self.smadist = lambda x,s0=self.smaknee: (x**(-0.62)*np.exp(-((x/s0)**2)))/self.smanorm
        
        #define Rayleigh eccentricity distribution
        self.esigma = float(esigma)
        self.edist1 = lambda x,sigma=self.esigma: (x/sigma**2)*np.exp(-x**2/(2.*sigma**2))
        self.enorm = integrate.quad(self.edist1, self.erange.min(), self.erange.max())[0]
        self.edist = lambda x,sigma=self.esigma: ((x/sigma**2)*np.exp(-x**2/(2.0*sigma**2)))/self.enorm
        
        #define Kepler radius distribution
        Rs = np.array([1,1.4,2.0,2.8,4.0,5.7,8.0,11.3,16,22.6]) #Earth Radii
        Rvals85 = np.array([0.1555, 0.1671, 0.1739, 0.0609, 0.0187, 0.0071, 0.0102, 0.0049, 0.0014])
        a85 = (((85*u.day/2/np.pi)**2*const.M_sun*const.G)**(1./3)).to('AU') #sma of 85 days
        fac1 = integrate.quad(self.smadist,0,a85.value)[0]
        Rvals = integrate.quad(self.smadist,0,self.arange[1].to('AU').value)[0]*(Rvals85/fac1)
        Rvals[5:] *= 2.5 #account for longer orbital baseline data
        
        self.Rs = Rs
        self.Rvals = Rvals
        self.eta = np.sum(Rvals)
        
        self.dist_albedo_built = False
        self.dist_radius_built = False
                
    def dist_sma(self, x):
        """Probability density function for semi-major axis in AU
        
        Args:
            x (float/ndarray):
                Semi-major axis value(s) in AU
                
        Returns:
            f (ndarray):
                Semi-major axis probability density
        
        """
        
        x = np.array(x, ndmin=1, copy=False)
        
        f = ((x >= self.arange[0].to('AU').value) & (x <= self.arange[1].to('AU').value)).astype(int)\
                *self.smadist(x)
        
        return f
        
    def dist_eccen(self, x):
        """Probability density function for eccentricity
        
        Args:
            x (float/ndarray):
                Eccentricity value(s)
        
        Returns:
            f (ndarray):
                Eccentricity probability density
        
        """
        
        x = np.array(x, ndmin=1, copy=False)
        
        f = ((x >= self.erange[0]) & (x <= self.erange[1])).astype(int)*self.edist(x)
        
        return f
        
    def dist_albedo(self, x):
        """Probability density function for albedo
        
        Args:
            x (float/ndarray):
                Albedo value(s)
        
        Returns:
            f (ndarray):
                Albedo probability density
                
        """
        
        if self.dist_albedo_built:
            f = self.dist_albedo(x)
        else:
            # define distribution for albedo
            p = self.gen_albedo(int(1e6))
            hp, pedges = np.histogram(p, bins=2000, range=(self.prange.min(),self.prange.max()), normed=True)
            pedges = 0.5*(pedges[1:]+pedges[:-1])
            pedges = np.hstack((self.prange.min(), pedges, self.prange.max()))
            hp = np.hstack((0.0, hp, 0.0))
            self.dist_albedo = interpolate.InterpolatedUnivariateSpline(pedges, hp, k=1, ext=1)
            f = self.dist_albedo(x)
            self.dist_albedo_built = True
            
        return f
        
    def dist_radius(self, x):
        """Probability density function for planetary radius
        
        Args:
            x (float/ndarray):
                Planetary radius value(s)
                
        Returns:
            f (ndarray):
                Planetary radius probability density
        
        """
        
        if self.dist_radius_built:
            f = self.dist_radius(x)
        else:
            # define distribution for radius
            R = self.gen_radius(int(1e6))
            hR, Redges = np.histogram(R.to('km').value, bins=2000, range=(self.Rprange.min().to('km').value, self.Rprange.max().to('km').value), normed=True)
            Redges = 0.5*(Redges[1:]+Redges[:-1])
            Redges = np.hstack((self.Rprange.min().to('km').value, Redges, self.Rprange.max().to('km').value))
            hR = np.hstack((0.0, hR, 0.0))
            self.dist_radius = interpolate.InterpolatedUnivariateSpline(Redges, hR, k=1, ext=1)
            f = self.dist_radius(x)
            self.dist_radius_built = True
            
        return f

    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        Samples a power law distribution with exponential turn-off 
        determined by class attribute smaknee
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity array):
                Semi-major axis in units of AU
        
        """
        n = self.gen_input_check(n)
        a = statsFun.simpSample(self.smadist, n, self.arange[0].to('AU').value,\
                self.arange[1].to('AU').value)*u.AU
        
        return a

    def gen_eccen(self, n):
        """Generate eccentricity values
        
        Rayleigh distribution, as in Kipping et. al (2013)
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            e (ndarray):
                Planet eccentricity
        
        """
        
        n = self.gen_input_check(n)
        e = statsFun.simpSample(self.edist, n, self.erange[0],self.erange[1])
        
        return e

    def gen_albedo(self, n):
        """Generate geometric albedo values
        
        The albedo is determined by sampling the semi-major axis distribution, 
        and then calculating the albedo from the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            p (ndarray):
                Planet albedo
        
        """
        
        n = self.gen_input_check(n)
        atmp = self.gen_sma(n)
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(atmp)
        
        return p

    def gen_radius(self,n):
        """Generate planetary radius values in km
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of km
        
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
        Rp = R*const.R_earth.to('km')
        
        return Rp

    def gen_radius_nonorm(self,n):
        """Generate planetary radius values in km
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.  The rates in the bins determine the overall
        occurrence rates of all planets.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of km
        
        """
        
        n = self.gen_input_check(n)
        R = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = np.random.poisson(lam=self.Rvals[j]*n)  
            R = np.hstack((R, np.exp(np.log(self.Rs[j])+\
                    (np.log(self.Rs[j+1])-np.log(self.Rs[j]))*\
                    np.random.uniform(size=nsamp))))
        Rp = R*const.R_earth.to('km')
        
        return Rp

    def gen_mass(self, n):
        """Generate planetary mass values in kg
        
        The mass is determined by sampling the radius and then calculating the
        mass from the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass in units of kg
        
        """
        
        n = self.gen_input_check(n)
        Rtmp = self.gen_radius(n)
        Mp = self.PlanetPhysicalModel.calc_mass_from_radius(Rtmp)
        
        return Mp
