# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u

class ZodiacalLight(object):
    """Zodiacal Light class template
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    Attributes:
        Zmag (float):
            zodi level in zodi
        exoZmag (float):
            1 zodi brightness in mag per asec2
        exoZnumber (float):
            exozodi level in zodi
        exoZvar (float):
            exozodi variation (variance of log-normal distribution)
        
    """

    _modtype = 'ZodiacalLight'
    _outspec = {}

    def __init__(self, Zmag=22.5, exoZmag=23.5, exoZnumber=1.5, exoZvar=0., **specs):

        self.Zmag = float(Zmag);                # 1 zodi brightness in mag per asec2
        self.exoZmag = float(exoZmag);          # 1 exozodi brightness in mag per asec2
        self.exoZnumber = float(exoZnumber);    # exozodi level in zodi
        self.exoZvar = float(exoZvar);          # exozodi variation (variance of log-normal distribution)

        for key in self.__dict__.keys():
            self._outspec[key] = self.__dict__[key]

    def __str__(self):
        """String representation of the Zodiacal Light object
        
        When the command 'print' is used on the Zodiacal Light object, this 
        method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Zodiacal Light class object attributes'
        
    def eclip_lats(self, coords):
        """Returns ecliptic latitudes 
        
        This method returns ecliptic latitudes for equatorial right ascension 
        and declination values in Star Catalog data (astropy.coordinates does 
        not yet support this conversion).
        
        Args:
            coords (SkyCoord):
                numpy ndarray of astropy SkyCoord objects with right ascension 
                and declination in degrees
                
        Returns:
            beta (ndarray):
                ecliptic latitudes in degrees
        
        """
        
        eps = 23.439281                 # J2000 obliquity of ecliptic in degrees
        a = np.cos(np.radians(eps))*np.sin(coords.dec)
        b = np.sin(coords.ra)*np.cos(coords.dec)*np.sin(np.radians(eps))
        beta = np.degrees(np.abs(np.arcsin(a-b)))

        return beta

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
            fzodicurr (ndarray):
                1D numpy ndarray of zodiacal light levels

        """
        
        fzodicurr = np.array([1e-9]*len(sInd))
        
        return fzodicurr

