import astropy.units as u
import astropy.constants as const
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
            Planet radius range in units of km
        Mprange (astropy Quantity 1x2 array):
            Planet mass range in units of kg
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
        adist (float, callable):
            Semi-major axis distribution
        edist (float, callable):
            Eccentricity distribution
        Rpdist (float, callable):
            Planet radius distribution
        Mpdist (float, callable):
            Planet mass distribution
        
    """

    _modtype = 'PlanetPopulation'
    _outspec = {}

    def __init__(self,arange=[0.1,100],erange=[0.01,0.99],Irange=[0,180],\
            Orange=[0,360],wrange=[0,360],prange=[0.1,0.6],Rprange=[1,30],\
            Mprange = [1,4131],adist=None,edist=None,Idist=None,Odist=None,\
            wdist=None,pdist=None,Rpdist=None,Mpdist=None,rdist=None,\
            scaleOrbits=False,constrainOrbits=False,eta=0.1,**specs):
        
        # check range of parameters
        self.arange = self.checkranges(arange,'arange')*u.AU
        self.erange = self.checkranges(erange,'erange')
        self.Irange = self.checkranges(Irange,'Irange')*u.deg
        self.Orange = self.checkranges(Orange,'Orange')*u.deg
        self.wrange = self.checkranges(wrange,'wrange')*u.deg
        self.prange = self.checkranges(prange,'prange')
        self.Rprange = self.checkranges(Rprange,'Rprange')*const.R_earth.to('km')
        self.Mprange = self.checkranges(Mprange,'Mprange')*const.M_earth.to('kg')
        # orbital radius range
        a = self.arange.to('AU').value
        self.rrange = [a[0]*(1.-self.erange[1]),a[1]*(1.+self.erange[1])]*u.AU
        
        # define prototype distributions of parameters (uniform and log-uniform)
        self.uniform = lambda x,v: np.array((x >= v[0])&(x <= v[1]),\
                dtype=float, ndmin=1) / (v[1] - v[0])
        self.logunif = lambda x,v: np.array((x >= v[0])&(x <= v[1]),\
                dtype=float, ndmin=1) / (x*np.log(v[1]/v[0]))
        self.adist = lambda x,v=self.arange.to('AU').value: self.logunif(x,v)
        self.edist = lambda x,v=self.erange: self.uniform(x,v)
        self.pdist = lambda x,v=self.prange: self.uniform(x,v)
        self.Rpdist = lambda x,v=self.Rprange.to('km').value: self.logunif(x,v)
        # mass distribution function (in Jupiter masses)
        self.Mpdist = lambda x: x**(-1.3)
        
        assert isinstance(scaleOrbits,bool), "scaleOrbits must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scaleOrbits
        
        assert isinstance(constrainOrbits,bool), "constrainOrbits must be boolean"
        # constrain planetary orbital radii to sma range
        self.constrainOrbits = constrainOrbits
        
        assert isinstance(eta,numbers.Number) and (eta > 0),\
                "eta must be strictly positive"
        #global occurrence rate defined as expected number of planets per 
        #star in a given universe
        self.eta = eta
        
        #populate all attributes to outspec
        for att in self.__dict__.keys():
            dat = copy.copy(self.__dict__[att])
            self._outspec[att] = dat.value if isinstance(dat,u.Quantity) else dat
            if att == 'Mprange':
                self._outspec[att] /= const.M_earth.to('kg').value
            elif att == 'Rprange':
                self._outspec[att] /= const.R_earth.to('km').value
        
        # import PlanetPhysicalModel
        self.PlanetPhysicalModel = get_module(specs['modules']['PlanetPhysicalModel'], \
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
            assert np.all(var >= 0) and np.all(var <= 1), "%s values must be between 0 and 1"%name
        
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
        v = self.arange.to('AU').value
        a = np.exp(np.random.uniform(low=np.log(v[0]),high=np.log(v[1]),size=n))*u.AU
        
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
        v = self.erange
        e = np.random.uniform(low=v[0],high=v[1],size=n)
        
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
        assert len(a) == n, "a input must be of size n."
        
        elim = np.min(np.vstack((1 - (self.arange[0]/a).decompose().value,\
                (self.arange[1]/a).decompose().value - 1)),axis=0)
        
        e = np.random.uniform(low=self.erange[0],high=elim,size=n)
        
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
        v = self.Irange.to('deg').value
        I = np.random.uniform(low=v[0],high=v[1],size=n)*u.deg
        
        return I

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
        v = self.Orange.to('deg').value
        O = np.random.uniform(low=v[0],high=v[1],size=n)*u.deg
        
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
        v = self.wrange.to('deg').value
        w = np.random.uniform(low=v[0],high=v[1],size=n)*u.deg
        
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
        v = self.prange
        p = np.random.uniform(low=v[0],high=v[1],size=n)
        
        return p

    def gen_radius(self, n):
        """Generate planetary radius values in m
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of km
        
        """
        n = self.gen_input_check(n)
        v = self.Rprange.to('km').value
        Rp = np.exp(np.random.uniform(low=np.log(v[0]),high=np.log(v[1]),size=n))*u.km
        
        return Rp

    def gen_mass(self, n):
        """Generate planetary mass values in kg
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass values in units of kg
        
        """
        n = self.gen_input_check(n)
        v = self.Mprange.to('kg').value
        Mp = np.exp(np.random.uniform(low=np.log(v[0]),high=np.log(v[1]),size=n))*u.kg
        
        return Mp