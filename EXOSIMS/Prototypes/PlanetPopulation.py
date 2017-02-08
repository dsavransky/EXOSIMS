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
        eta (float)
            Global occurrence rate defined as expected number of planets 
            per star in a given universe
        
    """

    _modtype = 'PlanetPopulation'
    _outspec = {}

    def __init__(self, arange=[0.1,100], erange=[0.01,0.99],\
                 Irange=[0.,180.], Orange=[0.,360.], wrange=[0.,360.],\
                 prange=[0.1,0.6], Rprange=[1.,30.], Mprange = [1.,4131.],\
                 scaleOrbits=False, constrainOrbits=False, eta=0.1, **specs):
        
        #do all input checks
        self.arange = self.checkranges(arange,'arange')*u.AU
        self.erange = self.checkranges(erange,'erange')
        self.Irange = self.checkranges(Irange,'Irange')*u.deg
        self.Orange = self.checkranges(Orange,'Orange')*u.deg
        self.wrange = self.checkranges(wrange,'wrange')*u.deg
        self.prange = self.checkranges(prange,'prange')
        self.Rprange = self.checkranges(Rprange,'Rprange')*const.R_earth.to('km')
        self.Mprange = self.checkranges(Mprange,'Mprange')*const.M_earth.to('kg')
        
        assert isinstance(scaleOrbits,bool), "scaleOrbits must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scaleOrbits
        
        assert isinstance(constrainOrbits,bool), "constrainOrbits must be boolean"
        # constrain planetary orbital radii to sma range
        self.constrainOrbits = constrainOrbits
        
        assert isinstance(eta,numbers.Number) and (eta > 0),\
                "eta must be a positive number."
        #global occurrence rate defined as expected number of planets per 
        #star in a given universe
        self.eta = eta
        
        # orbital radius range
        self.rrange = [self.arange[0].value*(1.-self.erange[1]),\
                self.arange[1].value*(1.+self.erange[1])]*u.AU
        
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
        assert len(var) == 2, "%s must have two elements,"%name
        assert var[0] <= var[1],\
            "The second element of %s must be greater or equal to the first."%name
        
        return np.array([float(v) for v in var])

    def __str__(self):
        """String representation of the Planet Population object
        
        When the command 'print' is used on the Planet Population object, this 
        method will print the attribute values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Population class object attributes'
        
    def dist_sma(self, x):
        """Probability density function for semi-major axis in AU
        
        The prototype provides a log-uniform distribution between the minimum
        and maximum values.
        
        Args:
            x (float/ndarray):
                Semi-major axis value(s) in AU
                
        Returns:
            f (ndarray):
                Semi-major axis probability density
        
        """
        
        x = np.array(x, ndmin=1, copy=False)
            
        f = ((x >= self.arange[0].to('AU').value) & (x <= self.arange[1].to('AU').value)).astype(int)\
                /(x*(np.log(self.arange[1].to('AU').value) - np.log(self.arange[0].to('AU').value)))
        
        return f
        
    def dist_eccen(self, x):
        """Probability density function for eccentricity
        
        The prototype provides a uniform distribution between the minimum and
        maximum values.
        
        Args:
            x (float/ndarray):
                Eccentricity value(s)
        
        Returns:
            f (ndarray):
                Eccentricity probability density
        
        """
        
        x = np.array(x, ndmin=1, copy=False)
            
        f = ((x >= self.erange[0]) & (x <= self.erange[1])).astype(int)\
                /(self.erange[1] - self.erange[0])
        
        return f
        
    def dist_albedo(self, x):
        """Probability density function for albedo
        
        The prototype provides a uniform distribution between the minimum and
        maximum values.
        
        Args:
            x (float/ndarray):
                Albedo value(s)
        
        Returns:
            f (ndarray):
                Albedo probability density
                
        """
        
        x = np.array(x, ndmin=1, copy=False)
            
        f = ((x >= self.prange[0]) & (x <= self.prange[1])).astype(int)/(self.prange[1] - self.prange[0])
                
        return f
        
    def dist_radius(self, x):
        """Probability density function for planetary radius
        
        The prototype provides a log-uniform distribution between the minimum
        and maximum values.
        
        Args:
            x (float/ndarray):
                Planetary radius value(s)
                
        Returns:
            f (ndarray):
                Planetary radius probability density
        
        """
        
        x = np.array(x, ndmin=1, copy=False)
        
        f = ((x >= self.Rprange[0].value) & (x <= self.Rprange[1].value)).astype(int)\
                /(x*(np.log(self.Rprange[1].value) - np.log(self.Rprange[0].value)))
        
        return f

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
            n (numeric):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity units AU)
        
        """
        n = self.gen_input_check(n)
        v = self.arange.value
        vals = np.exp(np.log(v[0])+(np.log(v[1])-np.log(v[0]))*np.random.uniform(size=n))
        
        return vals*self.arange.unit

    def gen_eccen(self, n):
        """Generate eccentricity values
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            e (numpy ndarray)
        
        """
        n = self.gen_input_check(n)
        vals = self.erange[0] +(self.erange[1] - self.erange[0])*np.random.uniform(size=n)
        
        return vals

    def gen_eccen_from_sma(self, n, a):
        """Generate eccentricity values constrained by semi-major axis, such that orbital
        radius always falls within the provided sma range.
        
        The prototype provides a uniform distribution between the minimum and 
        maximum allowable values.
        
        Args:
            n (numeric):
                Number of samples to generate
        
            a (Quantity):
                Array of semi-major axis values of length n
                
        Returns:
            e (numpy ndarray)
        
        """
        n = self.gen_input_check(n)
        assert len(a) == n, "a input must be of size n."
        
        elim = np.min(np.vstack((1 - (self.arange[0]/a).decompose().value,\
                (self.arange[1]/a).decompose().value - 1)),axis=0)
        
        vals = self.erange[0] +(elim - self.erange[0])*np.random.uniform(size=n)
        
        return vals

    def gen_I(self, n):
        """Generate inclination in degrees
        
        The prototype provides a sinusoidal distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            I (astropy Quantity units degrees)
        
        """
        n = self.gen_input_check(n)
        v = np.sort(np.cos(self.Irange))
        vals = np.arccos(v[0]+(v[1]-v[0])*np.random.uniform(size=n)).to(self.Irange.unit)
        
        return vals

    def gen_O(self, n):
        """Generate longitude of the ascending node in degrees
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            O (astropy Quantity units degrees)
        
        """
        n = self.gen_input_check(n)
        v = self.Orange.value
        vals = v[0]+(v[1]-v[0])*np.random.uniform(size=n)
        
        return vals*self.Orange.unit

    def gen_w(self, n):
        """Generate argument of periapse in degrees
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            w (astropy Quantity units degrees)
        
        """
        n = self.gen_input_check(n)
        v = self.wrange.value
        vals = v[0]+(v[1]-v[0])*np.random.uniform(size=n)
        
        return vals*self.wrange.unit

    def gen_albedo(self, n):
        """Generate geometric albedo values
        
        The prototype provides a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            p (numpy ndarray)
        
        """
        n = self.gen_input_check(n)
        vals = self.prange[0] +(self.prange[1] - self.prange[0])*np.random.uniform(size=n)
        
        return vals

    def gen_radius(self, n):
        """Generate planetary radius values in m
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity units m)
        
        """
        n = self.gen_input_check(n)
        v = self.Rprange.value
        vals = np.exp(np.log(v[0])+(np.log(v[1])-np.log(v[0]))*np.random.uniform(size=n))
        
        return vals*self.Rprange.unit

    def gen_mass(self, n):
        """Generate planetary mass values in kg
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity units kg)
        
        """
        n = self.gen_input_check(n)
        v = self.Mprange.value
        vals = np.exp(np.log(v[0])+(np.log(v[1])-np.log(v[0]))*np.random.uniform(size=n))
        
        return vals*self.Mprange.unit
