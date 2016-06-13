# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u

class ZodiacalLight(object):
    """Zodiacal Light class template
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    Attributes:
        magZ (float):
            1 zodi brightness magnitude (per arcsec2)
        magEZ (float):
            1 exo-zodi brightness magnitude (per arcsec2)
        varEZ (float):
            exo-zodiacal light variation (variance of log-normal distribution)
        nEZ (float):
            exo-zodiacal light level in zodi
        
    """

    _modtype = 'ZodiacalLight'
    _outspec = {}

    def __init__(self, magZ=23, magEZ=22, varEZ=0, nEZ=1.5, **specs):
        
        self.magZ = float(magZ)         # 1 zodi brightness (per arcsec2)
        self.magEZ = float(magEZ)       # 1 exo-zodi brightness (per arcsec2)
        self.varEZ = float(varEZ)       # exo-zodi variation (variance of log-normal dist)
        self.nEZ = float(nEZ)           # exo-zodi level in zodi
        
        assert self.varEZ >= 0, "Exozodi variation must be >= 0"
        
        # populate outspec
        for att in self.__dict__.keys():
            dat = self.__dict__[att]
            self._outspec[att] = dat.value if isinstance(dat,u.Quantity) else dat

    def __str__(self):
        """String representation of the Zodiacal Light object
        
        When the command 'print' is used on the Zodiacal Light object, this 
        method will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Zodiacal Light class object attributes'

    def fZ(self, TL, sInds, lam, r_sc):
        """Returns surface brightness of local zodiacal light
        
        Args:
            TL (object):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest, with the length of 
                the number of planets of interest
            lam (astropy Quantity):
                Central wavelength in units of nm
            r_sc (astropy Quantity 1x3 array):
                Observatory (spacecraft) position vector in units of km
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        """
        
        # check type of sInds
        sInds = np.array(sInds)
        if not sInds.shape:
            sInds = np.array([sInds])
        
        nZ = np.ones(len(sInds))
        fZ = nZ*10**(-0.4*self.magZ)/u.arcsec**2
        
        return fZ

    def fEZ(self, TL, sInds, I):
        """Returns surface brightness of exo-zodiacal light
        
        Args:
            TL (object):
                TargetList class object
            sInds (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I (astropy Quantity array):
                Inclination of the planets of interest in units of deg
        
        Returns:
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
        
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
        beta = I.value
        fbeta = 2.44 - 0.0403*beta + 0.000269*beta**2
        
        # absolute V-band magnitude of the star
        MV = TL.MV[sInds]
        # absolute V-band magnitude of the Sun
        MVsun = 4.83
        
        fEZ = nEZ*10**(-0.4*self.magEZ) * 2*fbeta * 10.**(-0.4*(MV-MVsun))/u.arcsec**2
        
        return fEZ