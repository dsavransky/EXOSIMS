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
    
    def __init__(self, **specs):
        
        # default values
        # minimum semi-major axis (AU)
        a_min = 0.01 
        # maximum semi-major axis (AU)
        a_max = 10. 
        # semi-major axis range
        self.arange = np.array([a_min, a_max])*u.AU 
        # minimum eccentricity
        e_min = 10.*np.finfo(np.float).eps 
        # maximum eccentricity
        e_max = 0.8 
        # eccentricity range
        self.erange = np.array([e_min, e_max]) 
        # minimum argument of perigee in degrees
        w_min = 0.
        # maximum argument of perigee in degrees
        w_max = 360.
        # argument of perigee range
        self.wrange = np.array([w_min, w_max])
        # minimum right ascension of the ascending node in degrees
        O_min = 0.
        # maximum right ascension of the ascending node in degrees
        O_max = 360.
        # right ascension of the ascending node range
        self.Orange = np.array([O_min, O_max])
        # minimum albedo
        p_min = 0.0004 
        # maximum albedo
        p_max = 0.6 
        # albedo range
        self.prange = np.array([p_min, p_max]) 
        # minimum inclination
        I_min = 0.
        # maximum inclination
        I_max = 180.
        # inclination range
        self.Irange = np.array([I_min, I_max])
        # minimum planetary radius
        R_min = 0.027*const.R_jup.to(u.km) 
        # maximum planetary radius
        R_max = 2.04*const.R_jup.to(u.km) 
        # planetary radius range
        self.Rrange = np.array([R_min.to(u.km).value, R_max.to(u.km).value])*u.km 
        # minimum planetary mass
        Mp_min = 6.3e-5*const.M_jup
        # maximum planetary mass
        Mp_max = 28.5*const.M_jup
        # planetary mass range
        self.Mprange = np.array([Mp_min.to(u.kg).value, Mp_max.to(u.kg).value])*u.kg
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = False 
        
        # replace default values with any user specified values
        atts = self.__dict__.keys()
        for att in atts:
            if att in specs:
                if att == 'a_min':
                    self.arange[0] = specs[att]*u.AU
                elif att == 'a_max':
                    self.arange[1] = specs[att]*u.AU
                elif att == 'e_min':
                    self.erange[0] = specs[att]                    
                elif att == 'e_max':
                    self.erange[1] = specs[att]
                elif att == 'w_min':
                    self.wrange[0] = specs[att]
                elif att == 'w_max':
                    self.wrange[1] = specs[att]
                elif att == 'O_min':
                    self.Orange[0] = specs[att]
                elif att == 'O_max':
                    self.Orange[1] = specs[att]
                elif att == 'p_min':
                    self.prange[0] = specs[att]
                elif att == 'p_max':
                    self.prange[1] = specs[att]
                elif att == 'R_min':
                    self.Rrange[0] = specs[att]*u.km
                elif att == 'R_max':
                    self.Rrange[1] = specs[att]*u.km
                elif att == 'Mp_min':
                    self.Mprange[0] = specs[att]*u.kg
                elif att == 'Mp_max':
                    self.Mprange[1] = specs[att]*u.kg
                else:
                    setattr(self, att, specs[att])
                    
        # initialize any values updated by functions
        # set values derived from quantities above
        # orbital radius range
        self.rrange = np.array([(np.min(self.arange)*(1.-np.max(self.erange))).to(u.km).value, (np.max(self.arange)*(1.+np.max(self.erange))).to(u.km).value])*u.km
                
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
        
        pdf = 1./(self.arange.max().value - self.arange.min().value)
        
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
        
        pdf = 1./(self.erange.max() - self.erange.min())
        
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
        
        pdf = 1./(self.wrange.max() - self.wrange.min())
        
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
        
        pdf = 1./(self.Orange.max() - self.Orange.min())
        
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
        
        pdf = 1./(self.Rrange.max().to(u.km).value - self.Rrange.min().to(u.km).value)
        
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
        
        pdf = 1./(self.Mprange.max().to(u.kg).value - self.Mprange.min().to(u.kg).value)
        
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
        
        pdf = 1./(self.prange.max() - self.prange.min())
        
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