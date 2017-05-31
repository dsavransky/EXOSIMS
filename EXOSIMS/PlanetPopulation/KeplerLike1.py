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
        
        # define sma distribution, with decay point (knee)
        smaknee = float(smaknee)
        a = self.arange.to('AU').value
        assert (smaknee >= a[0]) and (smaknee <= a[1]), \
               "sma knee value must be in sma range."
        norm = integrate.quad(lambda x,s0=smaknee: x**-0.62*np.exp(-(x/s0)**2),\
                a[0], a[1])[0]
        self.adist = lambda x,s0=smaknee,a=a: x**-0.62*np.exp(-(x/s0)**2)\
                / norm * np.array((x >= a[0])&(x <= a[1]),dtype=float, ndmin=1)
        
        # define Rayleigh eccentricity distribution
        esigma = float(esigma)
        norm = integrate.quad(lambda x,sig=esigma: x/sig**2*np.exp(-x**2/(2.*sig**2)),\
                self.erange[0], self.erange[1])[0]
        self.edist = lambda x,sig=esigma,e=self.erange: x/sig**2*np.exp(-x**2/(2.*sig**2))\
                / norm * np.array((x >= e[0])&(x <= e[1]),dtype=float, ndmin=1)
        
        # define Kepler radius distribution
        Rs = np.array([1,1.4,2.0,2.8,4.0,5.7,8.0,11.3,16,22.6]) #Earth Radii
        Rvals85 = np.array([0.1555,0.1671,0.1739,0.0609,0.0187,0.0071,0.0102,0.0049,0.0014])
        a85 = ((85.*u.day/2/np.pi)**2*const.M_sun*const.G)**(1./3) #sma of 85 days
        fac1 = integrate.quad(self.adist,0,a85.to('AU').value)[0]
        Rvals = integrate.quad(self.adist,0,a[1])[0]*(Rvals85/fac1)
        Rvals[5:] *= 2.5 #account for longer orbital baseline data
        self.Rs = Rs
        self.Rvals = Rvals
        self.eta = np.sum(Rvals)
        
        # populate outspec
        self._outspec['smaknee'] = smaknee
        self._outspec['esigma'] = esigma

    def pdist(self, x):
        """Probability density function for albedo
        
        Args:
            x (float/ndarray):
                Albedo value(s)
        
        Returns:
            f (ndarray):
                Albedo probability density
                
        """
        gen = self.gen_albedo(int(1e6))
        lim = tuple(self.prange)
        hist, edges = np.histogram(gen, bins=2000, range=lim, normed=True)
        edges = 0.5*(edges[1:] + edges[:-1])
        edges = np.hstack((lim[0], edges, lim[1]))
        hist = np.hstack((0., hist, 0.))
        self.pdist = interpolate.InterpolatedUnivariateSpline(edges,hist,k=1,ext=1)
        f = self.pdist(x)
        
        return f

    def Rpdist(self, x):
        """Probability density function for planetary radius
        
        Args:
            x (float/ndarray):
                Planetary radius value(s)
                
        Returns:
            f (ndarray):
                Planetary radius probability density
        
        """
        gen = self.gen_radius(int(1e6)).to('km').value
        lim = tuple(self.Rprange.to('km').value)
        hist, edges = np.histogram(gen, bins=2000, range=lim, normed=True)
        edges = 0.5*(edges[1:] + edges[:-1])
        edges = np.hstack((lim[0], edges, lim[1]))
        hist = np.hstack((0., hist, 0.))
        self.Rpdist = interpolate.InterpolatedUnivariateSpline(edges,hist,k=1,ext=1)
        f = self.Rpdist(x)
        
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
                Semi-major axis values in units of AU
        
        """
        n = self.gen_input_check(n)
        v = self.arange.to('AU').value
        a = statsFun.simpSample(self.adist, n, v[0], v[1])*u.AU
        
        return a

    def gen_eccen(self, n):
        """Generate eccentricity values
        
        Rayleigh distribution, as in Kipping et. al (2013)
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            e (ndarray):
                Planet eccentricity values
        
        """
        n = self.gen_input_check(n)
        v = self.erange
        e = statsFun.simpSample(self.edist, n, v[0], v[1])
        
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
                Planet albedo values
        
        """
        n = self.gen_input_check(n)
        a = self.gen_sma(n)
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(a)
        
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
                Planet radius values in units of km
        
        """
        n = self.gen_input_check(n)
        Rp = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = int(np.ceil(n*self.Rvals[j]/np.sum(self.Rvals)))
            Rp = np.hstack((Rp, np.exp(np.random.uniform(low=np.log(self.Rs[j]),\
                    high=np.log(self.Rs[j+1]),size=nsamp))))
        
        if len(Rp) > n:
            Rp = Rp[np.random.choice(range(len(Rp)),size=n,replace=False)]
        Rp = Rp*const.R_earth.to('km')
        
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
                Planet radius values in units of km
        
        """
        n = self.gen_input_check(n)
        Rp = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = np.random.poisson(lam=self.Rvals[j]*n)
            Rp = np.hstack((Rp, np.exp(np.random.uniform(low=np.log(self.Rs[j]),\
                    high=np.log(self.Rs[j+1]),size=nsamp))))
        Rp = Rp*const.R_earth.to('km')
        
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
                Planet mass values in units of kg
        
        """
        n = self.gen_input_check(n)
        Rp = self.gen_radius(n)
        Mp = self.PlanetPhysicalModel.calc_mass_from_radius(Rp).to('kg')
        
        return Mp
