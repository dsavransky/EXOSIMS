from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.util import statsFun 
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

class KeplerLike1(PlanetPopulation):
    """Population based on Kepler radius distribution with RV-like semi-major axis
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
        
        specs['prange'] = [0.083, 0.882]
        specs['Rprange'] = [1, 22.6]
        PlanetPopulation.__init__(self, **specs)
        
        # define sma distribution, with decay point (knee)
        smaknee = float(smaknee)
        a = self.arange.to('AU').value
        assert (smaknee >= a[0]) and (smaknee <= a[1]), \
               "sma knee value must be in sma range."
        norm = integrate.quad(lambda x,s0=smaknee: x**-0.62*np.exp(-(x/s0)**2),\
                a[0], a[1])[0]
        
        self.dist_sma = lambda x,s0=smaknee,a=a,norm=norm: x**-0.62*np.exp(-(x/s0)**2) \
                / norm * np.array((x >= a[0])&(x <= a[1]),dtype=float, ndmin=1)
        
        self.smaknee = smaknee
        # define Rayleigh eccentricity distribution
        self.esigma = float(esigma)
        norm = np.exp(-self.erange[0]**2/(2.0*self.esigma**2)) - np.exp(-self.erange[1]**2/(2.0*self.esigma**2))
        self.dist_eccen = lambda x,sig=self.esigma,e=self.erange,norm=norm: x/sig**2*np.exp(-x**2/(2.*sig**2))\
                / norm * np.array((x >= e[0])&(x <= e[1]),dtype=float, ndmin=1)
        
        # define Kepler radius distribution
        Rs = np.array([1,1.4,2.0,2.8,4.0,5.7,8.0,11.3,16,22.6]) #Earth Radii
        Rvals85 = np.array([0.1555,0.1671,0.1739,0.0609,0.0187,0.0071,0.0102,0.0049,0.0014])
        a85 = ((85.*u.day/2./np.pi)**2*u.solMass*const.G)**(1./3) #sma of 85 days
        fac1 = integrate.quad(self.dist_sma, 0, a85.to('AU').value)[0]
        Rvals = integrate.quad(self.dist_sma, 0, a[1])[0]*(Rvals85/fac1)
        Rvals[5:] *= 2.5 #account for longer orbital baseline data
        self.Rs = Rs
        self.Rvals = Rvals
        self.eta = np.sum(Rvals)
        
        # populate outspec
        self._outspec['smaknee'] = smaknee
        self._outspec['esigma'] = esigma
        
        self.dist_albedo_built = False
        self.dist_radius_built = False

    def dist_albedo(self, x):
        """Probability density function for albedo
        
        Args:
            x (float/ndarray):
                Albedo value(s)
        
        Returns:
            f (ndarray):
                Albedo probability density
                
        """
        if not self.dist_albedo_built:
            # define distribution for albedo
            p = self.gen_albedo(int(1e6))
            plim = self.prange
            hp, pedges = np.histogram(p, bins=2000, range=(plim[0], plim[1]), normed=True)
            pedges = 0.5*(pedges[1:] + pedges[:-1])
            pedges = np.hstack((plim[0], pedges, plim[1]))
            hp = np.hstack((0., hp, 0.))
            self.dist_albedo = interpolate.InterpolatedUnivariateSpline(pedges, 
                    hp, k=1, ext=1)
            self.dist_albedo_built = True
        
        f = self.dist_albedo(x)
        return f

    def dist_radius(self, x):
        """Probability density function for planetary radius in Earth radius
        
        Args:
            x (float/ndarray):
                Planetary radius value(s) in Earth radius. Not an astropy quantity.
                
        Returns:
            f (ndarray):
                Planetary radius probability density
        
        """
        
        if not isinstance(x, np.ndarray):
            x = np.array(x, ndmin=1, copy=False)
            
        f = np.zeros(x.shape)
        
        for i in xrange(len(self.Rvals)):
            inds = (x >= self.Rs[i]) & (x <= self.Rs[i+1])
            f[inds] = self.Rvals[i]/(x[inds]*np.log(self.Rs[i+1]/self.Rs[i])*self.eta)
        
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
        a = statsFun.simpSample(self.dist_sma, n, v[0], v[1])*u.AU
        
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
        e = statsFun.simpSample(self.dist_eccen, n, v[0], v[1])
        
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

    def gen_radius(self, n):
        """Generate planetary radius values in Earth radius
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        Rp = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = int(np.ceil(n*self.Rvals[j]/np.sum(self.Rvals)))
            Rp = np.hstack((Rp, np.exp(np.random.uniform(low=np.log(self.Rs[j]),
                    high=np.log(self.Rs[j+1]), size=nsamp))))
        
        if len(Rp) > n:
            Rp = Rp[np.random.choice(range(len(Rp)),size=n,replace=False)]
        Rp = Rp*u.earthRad
        
        return Rp

    def gen_radius_nonorm(self, n):
        """Generate planetary radius values in Earth radius.
        
        Samples a radius distribution defined as log-uniform in each of 9 radius bins
        with fixed occurrence rates.  The rates in the bins determine the overall
        occurrence rates of all planets.
        
        Args:
            n (integer):
                Number of target systems. Total number of samples generated will be,
                on average, n*self.eta
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        Rp = np.array([])
        for j in range(len(self.Rvals)):
            nsamp = np.random.poisson(lam=self.Rvals[j]*n)
            Rp = np.hstack((Rp, np.exp(np.random.uniform(low=np.log(self.Rs[j]),\
                    high=np.log(self.Rs[j+1]),size=nsamp))))
            
        np.random.shuffle(Rp) #randomize elements
        Rp = Rp*u.earthRad
        
        return Rp

    def gen_mass(self, n):
        """Generate planetary mass values in Earth Mass
        
        The mass is determined by sampling the radius and then calculating the
        mass from the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass values in units of Earth mass
        
        """
        n = self.gen_input_check(n)
        Rp = self.gen_radius(n)
        Mp = self.PlanetPhysicalModel.calc_mass_from_radius(Rp).to('earthMass')
        
        return Mp
    
    def gen_eccen_from_sma(self, n, a):
        """Generate eccentricity values constrained by semi-major axis, such that orbital
        radius always falls within the provided sma range.
        
        This provides a Rayleigh distribution between the minimum and 
        maximum allowable values.
        
        Args:
            n (integer):
                Number of samples to generate
            a (astropy Quantity array):
                Semi-major axis values in units of AU
            
        Returns:
            e (float ndarray):
                Eccentricity values
        
        """
        n = self.gen_input_check(n)
        assert len(a) == n, "a input must be of size n."
        
        # unitless sma range
        alim = self.arange.to('AU').value
        # mean sma value
        amean = np.mean(alim)
        # upper limit for eccentricity given sma
        sma = a.to('AU').value
        elim = np.zeros(sma.shape)
        elim[sma<=amean] = 1. - alim[0]/sma[sma<=amean]
        elim[sma>amean] = alim[1]/sma[sma>amean] - 1.
        
        # constants
        C1 = np.exp(-self.erange[0]**2/(2.*self.esigma**2))
        C2 = C1 - np.exp(-elim**2/(2.*self.esigma**2))
        
        e = self.esigma*np.sqrt(-2.*np.log(C1 - C2*np.random.uniform(size=n)))
        
        return e
    
    def dist_eccen_from_sma(self, e, a):
        """Probability density function for eccentricity constrained by 
        semi-major axis, such that orbital radius always falls within the 
        provided sma range.
        
        This provides a Rayleigh distribution between the minimum and 
        maximum allowable values.
        
        Args:
            e (ndarray):
                Eccentricity values
            a (float):
                Semi-major axis value in AU. Not an astropy quantity.
        
        Returns:
            f (ndarray):
                Probability density of eccentricity constrained by semi-major
                axis
        
        """
        if not isinstance(e,np.ndarray):
            e = np.array(e, ndmin=1, copy=False)
        if not isinstance(a,np.ndarray):
            a = np.array(a, ndmin=1, copy=False)
        
        if a.shape == e.shape or (len(a) == 1 and len(e) == e.size):
            # unitless sma range
            alim = self.arange.to('AU').value
            # mean sma value
            amean = np.mean(alim)
            elim = np.zeros(a.shape)
            elim[a<=amean] = 1. - alim[0]/a[a<=amean]
            elim[a>amean] = alim[1]/a[a>amean] - 1.
            
            norm = np.exp(-self.erange[0]**2/(2.*self.esigma**2)) \
                    - np.exp(-elim**2/(2.*self.esigma**2))
            ins = np.array((e >= self.erange[0]) & (e <= elim), dtype=float, ndmin=1)
            f = ins*e/self.esigma**2*np.exp(-e**2/(2.*self.esigma**2))/norm
            
        elif len(a) == a.size and len(e) == e.size:
            x, y = np.meshgrid(a,e)
            f = self.dist_eccen_from_sma(y,x)
        else:
            print 'Input mismatch between semi-major axis and eccentricity'
            print 'pdf set to zero'
            f = np.array([0.0])
        
        return f
