from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
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

    Attributes:
        cachedir (str):
            Path to EXOSIMS cache directory
            
    """

    _modtype = 'PlanetPhysicalModel'

    def __init__(self, cachedir=None, **specs):
        
        #start the outspec
        self._outspec = {}

        # cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        return

    def __str__(self):
        """String representation of Planet Physical Model object
        
        When the command 'print' is used on the Planet Physical Model object, 
        this method will return the values contained in the object"""
        
        for att in self.__dict__:
            print('%s: %r' % (att, getattr(self, att)))
        
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
        """Helper function for calculating radius given the mass.
        
        Prototype provides only a dummy function that assumes a density of water.
        
        Args:
            Mp (astropy Quantity array):
                Planet mass in units of Earth mass
        
        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of Earth radius
        
        """
        
        rho = 1000*u.kg/u.m**3.
        Rp = ((3.*Mp/rho/np.pi/4.)**(1./3.)).to('earthRad')
        
        return Rp

    def calc_mass_from_radius(self, Rp):
        """Helper function for calculating mass given the radius.
        
        Args:
            Rp (astropy Quantity array):
                Planet radius in units of Earth radius
        
        Returns:
            Mp (astropy Quantity array):
                Planet mass in units of Earth mass
        
        """
        
        rho = 1*u.tonne/u.m**3.
        Mp = (rho*4*np.pi*Rp**3./3.).to('earthMass')
        
        return Mp

    def calc_Phi(self, beta):
        """Calculate the phase function. Prototype method uses the Lambert phase 
        function from Sobolev 1975.
        
        Args:
            beta (astropy Quantity array):
                Planet phase angles at which the phase function is to be calculated,
                in units of rad
                
        Returns:
            Phi (ndarray):
                Planet phase function
        
        """
        
        beta = beta.to('rad').value
        Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi
        
        return Phi

    def calc_Teff(self, starL, d, p):
        """Calcluates the effective planet temperature given the stellar luminosity,
        planet albedo and star-planet distance.
        
        This calculation represents a basic balckbody power balance, and does not
        take into account the actual emmisivity of the planet, or any non-equilibrium
        effects or temperature variations over the surface.
        
        Note:  The input albedo is taken to be the bond albedo, as required by the equilibrium
        calculation. For an isotropic scatterer (Lambert phase function) the Bond albedo is 
        1.5 times the geometric albedo. However, the Bond albedo must be strictly defined between
        0 and 1, and an albedo of 1 produces a zero effective temperature.
        
        Args:
            starL (float ndarray):
                Stellar luminosities in units of solar luminosity. Not an astropy quantity.
            d (astropy Quantity array):
                Star-planet distances
            p (float ndarray):
                Planet albedos
        
        Returns:
            Teff (astropy quantity):
                Planet effective temperature in degrees K
        
        """
        
        Teff = ((const.L_sun*starL*(1 - p)/16./np.pi/const.sigma_sb/d**2)**(1/4.)).to('K')
        
        return Teff
