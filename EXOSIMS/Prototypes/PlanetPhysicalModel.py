import numpy as np
from astropy import units as u
from astropy import constants as const

class PlanetPhysicalModel(object):
    """Planet Physical Model class template
    
    This class contains all variables and functions necessary to perform 
    Planet Physical Model Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    """

    _modtype = 'PlanetPhysicalModel'
    _outspec = {}

    def __init__(self, **specs):
        
        return

    def __str__(self):
        """String representation of Planet Physical Model object
        
        When the command 'print' is used on the Planet Physical Model object, 
        this method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Physical Model class object attributes'


    def calc_albedo_from_sma(self,a):
        """
        Helper function for calculating albedo.
        The prototype provides only a dummy function that always returns the 
        same value of 0.367.

        Args:
            a (Quanitity):
               Array of semi-major axis values

        Returns:
            p (numpy ndarray)

        """
        
        return np.array([0.367]*a.size)


    def calc_radius_from_mass(self, M):
        """
        Helper function for calculating radius given the mass.
        
        Prototype provides only a dummy function that assumes a density of water.

        Args:
            M (Quantity):
               Array of mass values

        Returns:
            R (Quantity)

        """

        rho = 1000*u.kg/u.m**3.

        return ((3.*M/rho/np.pi/4.)**(1./3.)).decompose()


    def calc_mass_from_radius(self, R):
        """
        Helper function for calculating mass given the radius.

        Args:
            R (Quantity):
               Array of radius values

        Returns:
            M (Quantity)

        """

        rho = 1000*u.kg/u.m**3.

        return (rho*4*np.pi*R**3./4.).decompose()

