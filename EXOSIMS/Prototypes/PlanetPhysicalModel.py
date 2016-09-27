import numpy as np
import astropy.units as u
import astropy.constants as const

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
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Physical Model class object attributes'

    def calc_albedo_from_sma(self,a):
        """
        Helper function for calculating albedo given the semi-major axis.
        The prototype provides only a dummy function that always returns the 
        same value of 0.367.
        
        Args:
            a (astropy Quanitity array):
               Semi-major axis values
        
        Returns:
            p (ndarray):
                Albedo values
        
        """
        p = np.array([0.367]*a.size)
        
        return p

    def calc_radius_from_mass(self, Mp):
        """
        Helper function for calculating radius given the mass.
        
        Prototype provides only a dummy function that assumes a density of water.
        
        Args:
            Mp (astropy Quantity array):
                Planet mass in units of kg
        
        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of km
        
        """
        
        rho = 1000*u.kg/u.m**3.
        Rp = ((3.*Mp/rho/np.pi/4.)**(1./3.)).decompose()
        
        return Rp.to('km')

    def calc_mass_from_radius(self, Rp):
        """
        Helper function for calculating mass given the radius.
        
        Args:
            Rp (astropy Quantity array):
                Planet radius in units of km
        
        Returns:
            Mp (astropy Quantity array):
                Planet mass in units of kg
        
        """
        
        rho = 1000*u.kg/u.m**3.
        Mp = (rho*4*np.pi*Rp**3./3.).decompose()
        
        return Mp.to('kg')

    def calc_Phi(self,beta):
        """Calculate the phase function. Prototype method uses the Lambert phase 
        function from Sobolev 1975.
        
        Args:
            beta (astropy Quantity array):
                Planet phase angles at which the phase function is to be calculated,
                in units of rad
                
        Returns:
            Phi (astropy Quantity array):
                Planet phase function
        """
        Phi = (np.sin(beta) + (np.pi - beta.value)*np.cos(beta))/np.pi
        
        return Phi

