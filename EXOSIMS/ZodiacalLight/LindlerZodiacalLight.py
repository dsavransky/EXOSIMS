# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
from astropy import units as u

class LindlerZodiacalLight(ZodiacalLight):
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

    def fzodi(self, sInd, I, targlist):
        """Returns total zodi flux levels (local and exo)  
        
        This method is called in __init__ of SimulatedUniverse.
        
        Args:
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I (ndarray):
                1D numpy ndarray or scalar value of inclination in degrees
            targlist (TargetList):
                TargetList class object
        
        Returns:
            fzodi (ndarray):
                1D numpy ndarray of zodiacal light levels

        """
         
        # maximum V magnitude
        MV = targlist.MV 
        # ecliptic latitudes
        lats = self.eclip_lats(targlist.coords).value 
        
        i = np.where(I > 90.)
        if type(I) == np.ndarray:
            I[i] = 180. - I[i]
        
        if self.exoZvar == 0:
            R = self.exoZnumber
        else:
            # assume log-normal distribution of variance
            mu = np.log(self.exoZnumber) - 0.5*np.log(1. + self.exoZvar/self.exoZnumber**2)
            v = np.sqrt(np.log(self.exoZvar/self.exoZnumber**2 + 1.))
            R = np.random.lognormal(mean=mu, sigma=v, size=(len(sInd),))

        fzodi = 10**(-0.4*self.Zmag) * self.fbeta(lats[sInd]) \
                 + 10**(-0.4*self.exoZmag) * R*2.*self.fbeta(I)*2.5**(4.78-MV[sInd])

        return fzodi

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
        
        f = 2.44 - 0.0403*beta + 0.000269*beta**2
        
        return f