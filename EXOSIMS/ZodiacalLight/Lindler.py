# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
from astropy import units as u

class Lindler(ZodiacalLight):
    """Lindler Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Lindler.

    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        ZodiacalLight.__init__(self, **specs)

    def fZ(self, targlist, sInd, lam):
        """Returns surface brightness of local zodiacal light
        
        Args:
            targlist (TargetList):
                TargetList class object
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            lam:
                Central wavelength (default units of nm)
        
        Returns:
            fZ (ndarray):
                1D numpy ndarray of surface brightness of zodiacal light (per arcsec2)
        """

        # ecliptic latitudes
        lat = targlist.coords.barycentrictrueecliptic.lat[sInd]
        fZ = 10**(-0.4*self.magZ) * self.fbeta(lat)

        return fZ/u.arcsec**2
        
    def fEZ(self, targlist, sInd, I):
        """Returns surface brightness of exo-zodiacal light
        
        Args:
            targlist (TargetList):
                TargetList class object
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I (ndarray):
                1D numpy ndarray or scalar value of inclination in degrees
        
        Returns:
            fEZ (ndarray):
                1D numpy ndarray of surface brightness of exo-zodiacal light (per arcsec2)

        """
        
        i = np.where(I.value > 90.)
        if type(I) == np.ndarray:
            I[i] = 180. - I[i]
        # maximum V magnitude
        MV = targlist.MV
        # assume log-normal distribution of variance
        if self.varEZ != 0:
            mu = np.log(self.nEZ) - 0.5*np.log(1. + self.varEZ/self.nEZ**2)
            v = np.sqrt(np.log(self.varEZ/self.nEZ**2 + 1.))
            self.nEZ = np.random.lognormal(mean=mu, sigma=v, size=(len(sInd),))

        fEZ = 10**(-0.4*self.magEZ) * self.nEZ * 2.*self.fbeta(I)*2.5**(4.78-MV[sInd])

        return fEZ/u.arcsec**2
        
    def fbeta(self, beta):
        """Empirically derived variation of zodiacal light with viewing angle
        
        This method encodes the empirically derived formula for zodiacal light
        with viewing angle from Lindler.
        
        Args:
            beta (ndarray):
                angle in degrees
                
        Returns:
            f (ndarray):
                zodiacal light in zodi
        
        """
        
        beta = beta.value
        f = 2.44 - 0.0403*beta + 0.000269*beta**2
        
        return f