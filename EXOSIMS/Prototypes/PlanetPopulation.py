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
        arange (Quantity):
            1D numpy ndarray containing minimum and maximum semi-major axis 
            (default units of AU)
        erange (ndarray):
            1D numpy ndarray containing minimum and maximum eccentricity
        wrange (ndarray):
            1D numpy ndarray containing minimum and maximum argument of perigee
            in degrees
        Orange (ndarray):
            1D numpy ndarray containing minimum and maximum right ascension of
            the ascending node in degrees
        prange (ndarray):
            1D numpy ndarray containing minimum and maximum albedo
        Irange (ndarray):
            1D numpy ndarray containing minimum and maximum orbital inclination
            in degrees
        Rrange (Quantity):
            1D numpy ndarray containing minimum and maximum planetary radius 
            (default units of km)
        Mprange (Quantity):
            1D numpy ndarray containing minimum and maximum planetary mass
            (default units of kg)
        rrange (Quantity):
            1D numpy array containing minimum and maximum orbital radius
            (default units of km)
        scaleOrbits (bool):
            Scales orbits by sqrt(L) when True
        PlanetPhysicalModel (PlanetPhysicalModel):
            Planet physical model object
        
    """

    _modtype = 'PlanetPopulation'
    _outspec = {}
    
    def __init__(self, arange=[0.1,100], erange=[0.01,0.99],\
                 wrange=[0.,360.], Orange=[0.,360.], Irange=[0.,180.],\
                 prange=[0.1,0.6], Rrange=[1.,30.], Mprange = [1.,4131.],\
                 scaleOrbits=False, constrainOrbits=False, **specs):
        
        #do all input checks
        self.arange = self.checkranges(arange,'arange')*u.AU
        self.erange = self.checkranges(erange,'erange')
        self.wrange = self.checkranges(wrange,'wrange')*u.deg
        self.Orange = self.checkranges(Orange,'Orange')*u.deg
        self.Irange = self.checkranges(Irange,'Irange')*u.deg
        self.prange = self.checkranges(prange,'prange')
        self.Rrange = self.checkranges(Rrange,'Rrange')*const.R_earth
        self.Mprange = self.checkranges(Mprange,'Mprange')*const.M_earth

        assert isinstance(scaleOrbits,bool), "scaleOrbits must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scaleOrbits

        assert isinstance(constrainOrbits,bool), "constrainOrbits must be boolean"
        # constrain planetary orbital radii to sma range
        self.constrainOrbits = constrainOrbits
        
        # orbital radius range
        self.rrange = [self.arange[0].value*(1.-self.erange[1]),\
                self.arange[1].value*(1.+self.erange[1])]*u.AU

        #populate all atributes to outspec
        for key in self.__dict__.keys():
            att = self.__dict__[key]
            self._outspec[key] = copy.copy(att.value) if isinstance(att,u.Quantity) else att
            if key == 'Mprange':
                self._outspec[key] /= const.M_earth.value
            elif key == 'Rrange':
                self._outspec[key] /= const.R_earth.value

        # import PlanetPhysicalModel
        PlanPhys = get_module(specs['modules']['PlanetPhysicalModel'], 'PlanetPhysicalModel')
        self.PlanetPhysicalModel = PlanPhys(**specs)
    
    def checkranges(self,var,name):
        """Helper function provides asserts on all 2 element lists of ranges
        """
        assert len(var) == 2, "%s must have two elements,"%name
        assert var[0] <= var[1],\
            "The second element of %s must be greater or equal to the first."%name

        return [float(v) for v in var]


    def __str__(self):
        """String representation of the Planet Population object
        
        When the command 'print' is used on the Planet Population object, this 
        method will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Population class object attributes'
    
    def gen_input_check(self,n):
        """"
        Helper function checks that input is integer and casts to int
        """
        assert isinstance(n,numbers.Number) and float(n).is_integer(),\
            "Input must be an integer value."
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

    def gen_eccentricity(self, n):
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

    def gen_eccentricity_from_sma(self,n,a):
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

    def gen_radius(self, n):
        """Generate planetary radius values in m
        
        The prototype provides a log-uniform distribution between the minimum and 
        maximum values.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            R (astropy Quantity units m)

        """
        n = self.gen_input_check(n)
        v = self.Rrange.value
        vals = np.exp(np.log(v[0])+(np.log(v[1])-np.log(v[0]))*np.random.uniform(size=n))
        
        return vals*self.Rrange.unit
   

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

    def calc_Phi(self,r):
        """Calculate the Lambert phase function from Sobolev 1975.

        Args:
            r (Quantity):
                numpy ndarray containing planet position vectors relative to 
                host stars (units of distance)
        
        Returns:
            Phi (Quantity):
                numpy ndarray of planet phase function
        """

        d = np.sqrt(np.sum(r**2, axis=1))
        beta = np.arccos(r[:,2]/d).value
        Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi

        return Phi
