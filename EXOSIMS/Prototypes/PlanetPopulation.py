# -*- coding: utf-8 -*-
from astropy import units as u
from astropy import constants as const
import numpy as np

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
        scaleOrbits (bool):
            boolean which signifies if planetary orbits should be scaled by the
            square root of the luminosity
        rrange (Quantity):
            1D numpy array containing minimum and maximum orbital radius
            (default units of km)
        
    """

    _modtype = 'PlanetPopulation'
    _outspec = {}
    
    def __init__(self, arange=[0.1,100], erange=[0.01,0.99],\
                 wrange=[0.,360.], Orange=[0.,360.], Irange=[0.,180.],\
                 prange=[0.1,0.6], Rrange=[1.,30.], Mprange = [1.,4131.],\
                 scalefac=False, **specs):
        
        #do all input checks
        self.arange = self.checkranges(arange,'arange')*u.AU
        self.erange = self.checkranges(erange,'erange')
        self.wrange = self.checkranges(wrange,'wrange')*u.deg
        self.Orange = self.checkranges(Orange,'Orange')*u.deg
        self.Irange = self.checkranges(Irange,'Irange')*u.deg
        self.prange = self.checkranges(prange,'prange')
        self.Rrange = self.checkranges(Rrange,'Rrange')*const.R_earth
        self.Mprange = self.checkranges(Mprange,'Mprange')*const.M_earth

        assert isinstance(scalefac,bool), "scalefac must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scalefac
        
        # orbital radius range
        self.rrange = [self.arange[0].value*(1.-self.erange[1]),\
                self.arange[1].value*(1.+self.erange[1])]*u.AU

        #populate all atributes to outspec
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],u.Quantity):
                self._outspec[key] = self.__dict__[key].value
            else:
                self._outspec[key] = self.__dict__[key]
    
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
        
    def semi_axis(self, x):
        """Probability density function for semi-major axis in AU
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            x (float):
                semi-major axis value in AU
                
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.arange[1].value - self.arange[0].value)
        
        return pdf
        
    def eccentricity(self, x):
        """Probability density function for eccentricity
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            x (float):
                eccentricity value
        
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.erange[1] - self.erange[0])
        
        return pdf
        
    def arg_perigee(self, x):
        """Probability density function for argument of perigee in degrees
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            x (float):
                argument of perigee in degrees
        
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.wrange[1].value - self.wrange[0].value)
        
        return pdf
        
    def RAAN(self, x):
        """Probability density function for right ascension of the ascending node
        
        The prototype gives a uniform distribution between the minimum and
        maximum values in degrees.
        
        Args:
            x (float):
                right ascension of the ascending node in degrees
                
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.Orange[1].value - self.Orange[0].value)
        
        return pdf
        
    def radius(self, x):
        """Probability density function for planet radius in km
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values in km.
        
        Args:
            x (float):
                planet radius in km
        
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.Rrange[1].to(u.km).value - self.Rrange[0].to(u.km).value)
        
        return pdf
        
    def mass(self, x):
        """Probability density function for planetary mass in kg
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values in kg.
        
        Args:
            x (float):
                planet mass in kg
                
        Returns:
            pdf (float):
                probability density function value
            
        """
        
        pdf = 1./(self.Mprange[1].to(u.kg).value - self.Mprange[0].to(u.kg).value)
        
        return pdf
        
    def albedo(self, x):
        """Probability density function for planetary albedo
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            x (float):
                planet albedo
                
        Returns:
            pdf (float):
                probability density function value
        
        """
        
        pdf = 1./(self.prange[1] - self.prange[0])
        
        return pdf
        
    def inclination(self, x):
        """Probability density function for planet orbital inclination (degrees)
        
        The prototype gives a uniform distribution between the minimum and 
        maximum values.
        
        Args:
            x (float):
                planet orbital inclination in degrees
                
        Returns:
            pdf (float):
                probability density function
        
        """
        
        pdf = (1./2.)*np.sin(np.radians(x))
        
        return pdf
