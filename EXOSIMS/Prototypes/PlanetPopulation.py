import astropy.units as u
import numpy as np
import copy
import numbers
from EXOSIMS.util.get_module import get_module

class PlanetPopulation(object):
    """Planet Population Description class template
    
    This class contains all variables and functions necessary to perform 
    Planet Population Description Module calculations in exoplanet mission 
    simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        arange (astropy Quantity 1x2 array):
            Semi-major axis range in untis of AU
        erange (1x2 ndarray):
            Eccentricity range
        Irange (astropy Quantity 1x2 array):
            Orbital inclination range in units of deg
        Orange (astropy Quantity 1x2 array):
            Right ascension of the ascending node range in units of deg
        wrange (astropy Quantity 1x2 array):
            Argument of perigee range in units of deg
        prange (1x2 ndarray):
            Albedo range
        Rprange (astropy Quantity 1x2 array):
            Planet radius range in units of Earth radius
        Mprange (astropy Quantity 1x2 array):
            Planet mass range in units of Earth mass
        rrange (astropy Quantity 1x2 array):
            Orbital radius range in units of AU
        scaleOrbits (boolean):
            Scales orbits by sqrt(L) when True
        constrainOrbits (boolean):
            Constrains orbital radii to sma range when True
        eta (float):
            Global occurrence rate defined as expected number of planets 
            per star in a given universe
        uniform (float, callable):
            Uniform distribution over a given range
        logunif (float, callable):
            Log-uniform distribution over a given range
        
    """

    _modtype = 'PlanetPopulation'
    _outspec = {}

    def __init__(self, arange=[0.1,100.], erange=[0.01,0.99], Irange=[0.,180.],
            Orange=[0.,360.], wrange=[0.,360.], prange=[0.1,0.6], Rprange=[1.,30.],
            Mprange=[1.,4131.], scaleOrbits=False, constrainOrbits=False, eta=0.1, **specs):
        
        # check range of parameters
        self.arange = self.checkranges(arange,'arange')*u.AU
        self.erange = self.checkranges(erange,'erange')
        self.Irange = self.checkranges(Irange,'Irange')*u.deg
        self.Orange = self.checkranges(Orange,'Orange')*u.deg
        self.wrange = self.checkranges(wrange,'wrange')*u.deg
        self.prange = self.checkranges(prange,'prange')
        self.Rprange = self.checkranges(Rprange,'Rprange')*u.earthRad
        self.Mprange = self.checkranges(Mprange,'Mprange')*u.earthMass
        
        assert isinstance(scaleOrbits, bool), "scaleOrbits must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scaleOrbits
        
        assert isinstance(constrainOrbits,bool), "constrainOrbits must be boolean"
        # constrain planetary orbital radii to sma range
        self.constrainOrbits = constrainOrbits
        # derive orbital radius range from quantities above
        ar = self.arange.to('AU').value
        er = self.erange
        if self.constrainOrbits:
            self.rrange = [ar[0], ar[1]]*u.AU
        else:
            self.rrange = [ar[0]*(1. - er[1]), ar[1]*(1. + er[1])]*u.AU
        assert isinstance(eta, numbers.Number) and (eta > 0),\
                "eta must be strictly positive"
        # global occurrence rate defined as expected number of planets per 
        # star in a given universe
        self.eta = eta
        
        # populate all attributes to outspec
        for att in self.__dict__.keys():
            dat = copy.copy(self.__dict__[att])
            self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat
                
        # define prototype distributions of parameters (uniform and log-uniform)
        self.uniform = lambda x,v: np.array((np.array(x) >=v [0]) & 
                (np.array(x) <= v[1]), dtype=float, ndmin=1) / (v[1] - v[0])
        self.logunif = lambda x,v: np.array((np.array(x) >= v[0]) &
                (np.array(x) <= v[1]), dtype=float, ndmin=1) / (x*np.log(v[1]/v[0]))
        
        # import PlanetPhysicalModel
        self.PlanetPhysicalModel = get_module(specs['modules']['PlanetPhysicalModel'],
                'PlanetPhysicalModel')(**specs)

    def checkranges(self, var, name):
        """Helper function provides asserts on all 2 element lists of ranges
        """
        
        # reshape var
        assert len(var) == 2, "%s must have two elements,"%name
        var = np.array([float(v) for v in var])
        
        # check values
        if name in ['arange','Rprange','Mprange']:
            assert np.all(var > 0), "%s values must be strictly positive"%name
        if name in ['erange','prange']:
            assert np.all(var >= 0) and np.all(var <= 1),\
                    "%s values must be between 0 and 1"%name
        
        # the second element must be greater or equal to the first
        if var[1] < var[0]:
            var = var[::-1]
        
        return var

    def __str__(self):
        """String representation of the Planet Population object
        
        When the command 'print' is used on the Planet Population object, this 
        method will print the attribute values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Population class object attributes'

    def gen_input_check(self, n):
        """"
        Helper function checks that input is integer, casts to int, is >= 0
        """
        assert isinstance(n,numbers.Number) and float(n).is_integer(),\
            "Input must be an integer value."
        assert n >= 0, "Input must be nonnegative"
        
        return int(n)

    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity array):
                Semi-major axis values in units of AU
        
        """
        n = self.gen_input_check(n)
        ar = self.arange.to('AU').value
        a = np.exp(np.random.uniform(low=np.log(ar[0]), high=np.log(ar[1]), size=n))*u.AU
        
        return a

    def gen_eccen(self, n):
        """Generate eccentricity values
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
            
        Returns:
            e (float ndarray)
                Eccentricity values
        
        """
        n = self.gen_input_check(n)
        er = self.erange
        e = np.random.uniform(low=er[0], high=er[1], size=n)
        
        return e

    def gen_eccen_from_sma(self, n, a):
        """Generate eccentricity values constrained by semi-major axis, such that orbital
        radius always falls within the provided sma range.
        
        The prototype provides a uniform distribution between the minimum and 
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
        
        # cast a to array
        a = np.array(a.to('AU').value, ndmin=1, copy=False)
        assert len(a) in [1, n], "sma input must be of length 1 or n."
        
        # unitless sma range
        ar = self.arange.to('AU').value
        assert np.all((a >= ar[0]) & (a <= ar[1])), "sma input values must be within sma range."
        
        # upper limit for eccentricity given sma
        elim = np.zeros(len(a))
        amean = np.mean(ar)
        elim[a <= amean] = 1. - ar[0]/a[a <= amean]
        elim[a > amean] = ar[1]/a[a>amean] - 1.
        
        # uniform distribution
        e = np.random.uniform(low=self.erange[0], high=elim, size=n)
        
        return e

    def gen_I(self, n):
        """Generate inclination in degrees
        
        The prototype provides a sinusoidal distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            I (astropy Quantity array):
                Inclination values in units of deg
        
        """
        n = self.gen_input_check(n)
        I = np.arccos(1. - 2.*np.random.uniform(size=n))*u.rad
        
        return I.to('deg')

    def gen_O(self, n):
        """Generate longitude of the ascending node in degrees
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            O (astropy Quantity array):
                Right ascension of the ascending node values in units of deg
        
        """
        n = self.gen_input_check(n)
        Or = self.Orange.to('deg').value
        O = np.random.uniform(low=Or[0], high=Or[1], size=n)*u.deg
        
        return O

    def gen_w(self, n):
        """Generate argument of periapse in degrees
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            w (astropy Quantity array):
                Argument of periapse values in units of deg
        
        """
        n = self.gen_input_check(n)
        wr = self.wrange.to('deg').value
        w = np.random.uniform(low=wr[0], high=wr[1], size=n)*u.deg
        
        return w

    def gen_albedo(self, n):
        """Generate geometric albedo values
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            p (float ndarray):
                Albedo values
        
        """
        n = self.gen_input_check(n)
        pr = self.prange
        p = np.random.uniform(low=pr[0], high=pr[1], size=n)
        
        return p

    def gen_radius(self, n):
        """Generate planetary radius values in units of Earth radius.
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        Rpr = self.Rprange.to('earthRad').value
        Rp = np.exp(np.random.uniform(low=np.log(Rpr[0]), high=np.log(Rpr[1]), 
                size=n))*u.earthRad
        
        return Rp

    def gen_mass(self, n):
        """Generate planetary mass values in units of Earth mass.
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass values in units of Earth mass.
        
        """
        n = self.gen_input_check(n)
        Mpr = self.Mprange.to('earthMass').value
        Mp = np.exp(np.random.uniform(low=np.log(Mpr[0]), high=np.log(Mpr[1]), 
                size=n))*u.earthMass
        
        return Mp

    def dist_eccen_from_sma(self, e, a):
        """Probability density function for eccentricity constrained by 
        semi-major axis, such that orbital radius always falls within the 
        provided sma range.
        
        The prototype provides a uniform distribution between the minimum and 
        maximum allowable values.
        
        Args:
            e (float ndarray):
                Eccentricity values
            a (float ndarray):
                Semi-major axis value in AU. Not an astropy quantity.
        
        Returns:
            f (float ndarray):
                Probability density of eccentricity constrained by semi-major axis
        
        """
        
        # cast a and e to array
        e = np.array(e, ndmin=1, copy=False)
        a = np.array(a, ndmin=1, copy=False)
        
        # unitless sma range
        ar = self.arange.to('AU').value
        
        # upper limit for eccentricity given sma
        elim = np.zeros(a.shape)
        amean = np.mean(ar)
        elim[a <= amean] = 1. - ar[0]/a[a <= amean]
        elim[a > amean] = ar[1]/a[a > amean] - 1.
        
        # if e and a are two arrays of different size, create a 2D grid
        if a.size not in [1, e.size]:
            elim, e = np.meshgrid(elim, e)
        
        f = self.uniform(e, (self.erange[0], elim))
        
        return f

    def dist_sma(self, a):
        """Probability density function for semi-major axis in AU
        
        The prototype provides a log-uniform distribution between the minimum
        and maximum values.
        
        Args:
            a (float ndarray):
                Semi-major axis value(s) in AU. Not an astropy quantity.
                
        Returns:
            f (float ndarray):
                Semi-major axis probability density
        
        """
        
        return self.logunif(a, self.arange.to('AU').value)

    def dist_eccen(self, e):
        """Probability density function for eccentricity
        
        The prototype provides a uniform distribution between the minimum and
        maximum values.
        
        Args:
            e (float ndarray):
                Eccentricity value(s)
        
        Returns:
            f (float ndarray):
                Eccentricity probability density
        
        """
        
        return self.uniform(e, self.erange)

    def dist_albedo(self, p):
        """Probability density function for albedo
        
        The prototype provides a uniform distribution between the minimum and
        maximum values.
        
        Args:
            p (float ndarray):
                Albedo value(s)
        
        Returns:
            f (float ndarray):
                Albedo probability density
                
        """
       
        return self.uniform(p, self.prange)

    def dist_radius(self, Rp):
        """Probability density function for planetary radius in Earth radius
        
        The prototype provides a log-uniform distribution between the minimum
        and maximum values.
        
        Args:
            Rp (float ndarray):
                Planetary radius value(s) in Earth radius. Not an astropy quantity.
                
        Returns:
            f (float ndarray):
                Planetary radius probability density
        
        """
       
        return self.logunif(Rp, self.Rprange.to('earthRad').value)

    def dist_mass(self, Mp):
        """Probability density function for planetary mass in Earth mass
        
        The prototype provides an unbounded power law distribution. Note
        that this should really be a function of a density model and the radius 
        distribution for all implementations that use it.
        
        Args:
            Mp (float ndarray):
                Planetary mass value(s) in Earth mass. Not an astropy quantity.
                
        Returns:
            f (float ndarray):
                Planetary mass probability density
        
        """
        
        # convert to Jupiter mass
        Mjup = (Mp*u.earthMass).to('jupiterMass').value
        
        return Mjup**(-1.3)
