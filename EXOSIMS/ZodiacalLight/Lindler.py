# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import astropy.units as u

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

    def fZ(self, TL, sInds, lam):
        """Returns surface brightness of local zodiacal light
        
        Args:
            TL (TargetList):
                TargetList class object
            sInds (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            lam:
                Central wavelength (default units of nm)
        
        Returns:
            fZ (ndarray):
                1D numpy ndarray of surface brightness of zodiacal light (per arcsec2)
        """
        
        # check type of sInds
        sInds = np.array(sInds)
        if not sInds.shape:
            sInds = np.array([sInds])
        
        # ecliptic latitudes
        lat = TL.coords.barycentrictrueecliptic.lat[sInds]
        nZ = self.fbeta(lat)
        
        fZ = nZ*10**(-0.4*self.magZ)/u.arcsec**2
        
        return fZ

    def fEZ(self, TL, sInds, I):
        """Returns surface brightness of exo-zodiacal light
        
        Args:
            TL (TargetList):
                TargetList class object
            sInds (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I (ndarray):
                1D numpy ndarray or scalar value of inclination in degrees
        
        Returns:
            fEZ (ndarray):
                1D numpy ndarray of surface brightness of exo-zodiacal light (per arcsec2)
        
        """
        
        # check type of sInds
        sInds = np.array(sInds)
        if not sInds.shape:
            sInds = np.array([sInds])
        
        # assume log-normal distribution of variance
        if self.varEZ == 0:
            nEZ = np.array([self.nEZ]*len(sInds))
        else:
            mu = np.log(self.nEZ) - 0.5*np.log(1. + self.varEZ/self.nEZ**2)
            v = np.sqrt(np.log(self.varEZ/self.nEZ**2 + 1.))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=len(sInds))
        
        # supplementary angle for inclination > 90 degrees
        mask = np.where(I.value > 90)[0]
        I.value[mask] = 180 - I.value[mask]
        
        # maximum V magnitude
        MV = TL.MV
        
        fEZ = nEZ*10**(-0.4*self.magEZ)*2*self.fbeta(I)*2.5**(4.78-MV[sInds])/u.arcsec**2
        
        return fEZ

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