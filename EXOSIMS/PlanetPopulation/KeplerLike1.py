from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
from EXOSIMS.util import statsFun 
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import sys

# Python 3 compatibility:
if sys.version_info[0] > 2:
    xrange = range

class KeplerLike1(PlanetPopulation):
    """Population based on Kepler radius distribution with RV-like semi-major axis
    distribution with exponential decay.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
        smaknee (float):
            Location (in AU) of semi-major axis decay point (knee).
            Not an astropy quantity.
        esigma (float):
            Sigma value of Rayleigh distribution for eccentricity.
        
    Notes:  
    1. The gen_mass function samples the Radius and calculates the mass from
    there.  Any user-set mass limits are ignored.
    2. The gen_albedo function samples the sma, and then calculates the albedos
    from there. Any user-set albedo limits are ignored.
    3. The Rprange is fixed to (1,22.6) R_Earth and cannot be overwritten by user
    settings (the JSON input will be ignored) 
    4. The radius piece-wise distribution (from Fressin et al 2012) provides 
    the normalization required to get the proper overall eta.  The gen_radius 
    method provided here normalizes in order to return exactly the number of 
    samples requested.  A second method (gen_radius_nonorm) is provided for 
    generating the simulated universe population. The latter assumes a poisson 
    distribution for occurences in each bin.
    5.  Eccentricity is assumed to be Rayleigh distributed with a user-settable 
    sigma parameter (defaults to value from Fressin et al 2012).
    
    """

    def __init__(self, smaknee=30, esigma=0.175/np.sqrt(np.pi/2.), 
            prange=[0.083, 0.882], Rprange=[1, 22.6], **specs):
        
        specs['prange'] = prange
        specs['Rprange'] = Rprange
        PlanetPopulation.__init__(self, **specs)
        
        # calculate norm for sma distribution with decay point (knee)
        self.smaknee = float(smaknee)
        ar = self.arange.to('AU').value
        # sma distribution without normalization
        tmp_dist_sma = lambda x,s0=self.smaknee: x**(-0.62)*np.exp(-(x/s0)**2)
        self.smanorm = integrate.quad(tmp_dist_sma, ar[0], ar[1])[0]
        
        # calculate norm for eccentricity Rayleigh distribution 
        self.esigma = float(esigma)
        er = self.erange
        self.enorm = np.exp(-er[0]**2/(2.*self.esigma**2)) \
                - np.exp(-er[1]**2/(2.*self.esigma**2))
        
        # define Kepler radius distribution
        Rs = np.array([1,1.4,2.0,2.8,4.0,5.7,8.0,11.3,16,22.6]) #Earth Radii
        Rvals85 = np.array([0.1555,0.1671,0.1739,0.0609,0.0187,0.0071,0.0102,0.0049,0.0014])
        #sma of 85 days
        a85 = ((85.*u.day/2./np.pi)**2*u.solMass*const.G)**(1./3.)
        # sma of 0.8 days (lower limit of Fressin et al 2012)
        a08 = ((0.8*u.day/2./np.pi)**2*u.solMass*const.G)**(1./3.) 
        fac1 = integrate.quad(tmp_dist_sma, a08.to('AU').value, a85.to('AU').value)[0]
        Rvals = integrate.quad(tmp_dist_sma, ar[0], ar[1])[0]*(Rvals85/fac1)
        Rvals[5:] *= 2.5 #account for longer orbital baseline data
        self.Rs = Rs
        self.Rvals = Rvals
        self.eta = np.sum(Rvals)
        
        # populate outspec with attributes specific to KeplerLike1
        self._outspec['smaknee'] = self.smaknee
        self._outspec['esigma'] = self.esigma
        self._outspec['eta'] = self.eta
        
        self.dist_albedo_built = None

    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        Samples a power law distribution with exponential turn-off 
        determined by class attribute smaknee
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            astropy Quantity array:
                Semi-major axis values in units of AU
        
        """
        n = self.gen_input_check(n)
        ar = self.arange.to('AU').value
        a = statsFun.simpSample(self.dist_sma, n, ar[0], ar[1])*u.AU
        
        return a

    def gen_albedo(self, n):
        """Generate geometric albedo values
        
        The albedo is determined by sampling the semi-major axis distribution, 
        and then calculating the albedo from the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            float ndarray:
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
            astropy Quantity array:
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        
        # get number of samples per bin
        nsamp = np.ceil(n*self.Rvals/np.sum(self.Rvals)).astype(int)
        
        # generate random radii in each bin
        logRs = np.log(self.Rs)
        Rp = np.concatenate([np.exp(np.random.uniform(low=logRs[j], high=logRs[j+1], 
                size=nsamp[j])) for j in range(len(self.Rvals))])
        
        # select n radom elements from Rp
        ind = np.random.choice(len(Rp), size=n, replace=len(Rp)<n)
        Rp = Rp[ind]
        
        return Rp*u.earthRad

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
            astropy Quantity array:
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        
        # get number of samples per bin
        nsamp = np.random.poisson(n*self.Rvals)
        
        # generate random radii in each bin
        logRs = np.log(self.Rs)
        Rp = np.concatenate([np.exp(np.random.uniform(low=logRs[j], high=logRs[j+1], 
                size=nsamp[j])) for j in range(len(self.Rvals))])
        
        # randomize elements in Rp
        np.random.shuffle(Rp)
        
        return Rp*u.earthRad

    def gen_mass(self, n):
        """Generate planetary mass values in Earth Mass
        
        The mass is determined by sampling the radius and then calculating the
        mass from the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            astropy Quantity array:
                Planet mass values in units of Earth mass
        
        """
        n = self.gen_input_check(n)
        Rp = self.gen_radius(n)
        Mp = self.PlanetPhysicalModel.calc_mass_from_radius(Rp).to('earthMass')
        
        return Mp
    
    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)
        
        Semi-major axis is distributed RV like with exponential decay. 
        Eccentricity is a Rayleigh distribution. Albedo is dependent on the 
        PlanetPhysicalModel but is calculated such that it is independent of 
        other parameters. Planetary radius comes from the Kepler observations.
        
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
        PPMod = self.PlanetPhysicalModel
        # generate semi-major axis samples
        a = self.gen_sma(n)
        # check for constrainOrbits == True for eccentricity samples
        # constant
        C1 = np.exp(-self.erange[0]**2/(2.*self.esigma**2))
        ar = self.arange.to('AU').value
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array([ar[0]/(1.-self.erange[0]), ar[1]/(1.+self.erange[0])])
            # clip sma values to sma range
            sma = np.clip(a.to('AU').value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1. - ar[0]/sma[sma <= amean]
            elim[sma > amean] = ar[1]/sma[sma>amean] - 1.
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # constants
            C2 = C1 - np.exp(-elim**2/(2.*self.esigma**2))
            a = sma*u.AU
        else:
            C2 = self.enorm
        e = self.esigma*np.sqrt(-2.*np.log(C1 - C2*np.random.uniform(size=n)))
        # generate albedo from semi-major axis
        p = PPMod.calc_albedo_from_sma(a)
        # generate planetary radius
        Rp = self.gen_radius(n)
        
        return a, e, p, Rp

    def dist_sma(self, a):
        """Probability density function for semi-major axis in AU
        
        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.
                
        Returns:
            float ndarray:
                Semi-major axis probability density
        
        """
        
        # cast to array
        a = np.array(a, ndmin=1, copy=False)
        
        # unitless sma range
        ar = self.arange.to('AU').value
        
        # RV-like semi-major axis distribution with exponential decay
        f = np.zeros(a.shape)
        mask = np.array((a >= ar[0]) & (a <= ar[1]), ndmin=1)
        f[mask] = a[mask]**-0.62*np.exp(-(a[mask]/self.smaknee)**2)/self.smanorm
        
        return f

    def dist_eccen(self, e):
        """Probability density function for eccentricity
        
        Args:
            e (float ndarray):
                Eccentricity value(s)
        
        Returns:
            float ndarray:
                Eccentricity probability density
        
        """
        
        # cast to array
        e = np.array(e, ndmin=1, copy=False)
        
        # Rayleigh distribution sigma
        f = np.zeros(e.shape)
        mask = np.array((e >= self.erange[0]) & (e <= self.erange[1]), ndmin=1)
        f[mask] = e[mask]/self.esigma**2*np.exp(-e[mask]**2/(2.*self.esigma**2))/self.enorm
        
        return f

    def dist_eccen_from_sma(self, e, a):
        """Probability density function for eccentricity constrained by 
        semi-major axis, such that orbital radius always falls within the 
        provided sma range.
        
        This provides a Rayleigh distribution between the minimum and 
        maximum allowable values.
        
        Args:
            e (float ndarray):
                Eccentricity values
            a (float ndarray):
                Semi-major axis value in AU. Not an astropy quantity.
        
        Returns:
            float ndarray:
                Probability density of eccentricity constrained by semi-major
                axis
        
        """
        
        # cast a and e to array
        e = np.array(e, ndmin=1, copy=False)
        a = np.array(a, ndmin=1, copy=False)
        # if a is length 1, copy a to make the same shape as e
        if a.ndim == 1 and len(a) == 1:
            a = a*np.ones(e.shape)
        
        # unitless sma range
        ar = self.arange.to('AU').value
        arcon = np.array([ar[0]/(1.-self.erange[0]), ar[1]/(1.+self.erange[0])])
        # upper limit for eccentricity given sma
        elim = np.zeros(a.shape)
        amean = np.mean(arcon)
        elim[a <= amean] = 1. - ar[0]/a[a <= amean]
        elim[a > amean] = ar[1]/a[a > amean] - 1.
        elim[elim > self.erange[1]] = self.erange[1]
        elim[elim < self.erange[0]] = self.erange[0]
        
        # if e and a are two arrays of different size, create a 2D grid
        if a.size not in [1, e.size]:
            elim, e = np.meshgrid(elim, e)
        
        norm = np.exp(-self.erange[0]**2/(2.*self.esigma**2)) \
                - np.exp(-elim**2/(2.*self.esigma**2))
        ins = np.array((e >= self.erange[0]) & (e <= elim), dtype=float, ndmin=1)
        f = np.zeros(e.shape)
        mask = (a >= arcon[0]) & (a <= arcon[1])
        f[mask] = ins[mask]*e[mask]/self.esigma**2*np.exp(-e[mask]**2/(2.*self.esigma**2))/norm[mask]
        
        return f

    def dist_albedo(self, p):
        """Probability density function for albedo
        
        Args:
            p (float ndarray):
                Albedo value(s)
        
        Returns:
            float ndarray:
                Albedo probability density
                
        """
        
        # if called for the first time, define distribution for albedo
        if self.dist_albedo_built is None:
            pgen = self.gen_albedo(int(1e6))
            pr = self.prange
            hp, pedges = np.histogram(pgen, bins=2000, range=(pr[0], pr[1]), density=True)
            pedges = 0.5*(pedges[1:] + pedges[:-1])
            pedges = np.hstack((pr[0], pedges, pr[1]))
            hp = np.hstack((0., hp, 0.))
            self.dist_albedo_built = interpolate.InterpolatedUnivariateSpline(pedges, 
                    hp, k=1, ext=1)
        
        f = self.dist_albedo_built(p)
        
        return f

    def dist_radius(self, Rp):
        """Probability density function for planetary radius in Earth radius
        
        Args:
            Rp (float ndarray):
                Planetary radius value(s) in Earth radius. Not an astropy quantity.
                
        Returns:
            float ndarray:
                Planetary radius probability density
        
        """
        
        # cast Rp to array
        Rp = np.array(Rp, ndmin=1, copy=False)
        
        # radius distribution
        Rnorm = self.Rvals/np.log(self.Rs[1:]/self.Rs[:-1])/self.eta
        f = np.zeros(Rp.shape)
        for i in xrange(len(self.Rvals)):
            mask = (Rp >= self.Rs[i]) & (Rp <= self.Rs[i+1])
            f[mask] = Rnorm[i]/Rp[mask]
        
        return f
