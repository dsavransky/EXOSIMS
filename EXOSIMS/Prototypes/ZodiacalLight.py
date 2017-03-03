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
        fZ0 (astropy Quantity):
            default surface brightness of zodiacal light in units of 1/arcsec2
        fEZ0 (astropy Quantity):
            default surface brightness of exo-zodiacal light in units of 1/arcsec2
        
    """

    _modtype = 'ZodiacalLight'
    _outspec = {}

    def __init__(self, magZ=23, magEZ=22, varEZ=0, **specs):
        
        self.magZ = float(magZ)         # 1 zodi brightness (per arcsec2)
        self.magEZ = float(magEZ)       # 1 exo-zodi brightness (per arcsec2)
        self.varEZ = float(varEZ)       # exo-zodi variation (variance of log-normal dist)
        self.fZ0 = 10**(-0.4*self.magZ)/u.arcsec**2   # default zodi brightness
        self.fEZ0 = 10**(-0.4*self.magEZ)/u.arcsec**2 # default exo-zodi brightness
        
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
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            lam (astropy Quantity):
                Central wavelength in units of nm
            r_sc (astropy Quantity nx3 array):
                Observatory (spacecraft) position vector in units of km
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        
        Note: r_sc must be an array of shape (sInds.size x 3)
        
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        # check shape of r_sc
        assert r_sc.shape == (sInds.size,3), 'r_sc must be of shape (sInds.size x 3)'
        
        nZ = np.ones(len(sInds))
        fZ = nZ*10**(-0.4*self.magZ)/u.arcsec**2
        
        return fZ

    def fEZ(self, TL, sInds, I, r_orbit):
        """Returns surface brightness of exo-zodiacal light
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            I (astropy Quantity array):
                Inclination of the planets of interest in units of deg
            r_orbit (astropy Quantity nx3 array):
                Orbital radii of the planets of interest in units of AU
        
        Returns:
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
        
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        
        # assume log-normal distribution of variance
        nEZ = np.ones(len(sInds))
        if self.varEZ != 0:
            mu = np.log(nEZ) - 0.5*np.log(1. + self.varEZ/nEZ**2)
            v = np.sqrt(np.log(self.varEZ/nEZ**2 + 1.))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=len(sInds))
        
        # supplementary angle for inclination > 90 degrees
        beta = I.to('deg').value
        mask = np.where(beta > 90)[0]
        beta[mask] = 180 - beta[mask]
        fbeta = 2.44 - 0.0403*beta + 0.000269*beta**2
        
        # apparent magnitude of the star (in the V band)
        MV = TL.MV[sInds]
        # apparent magnitude of the Sun (in the V band)
        MVsun = 4.83
        
        fEZ = nEZ*10**(-0.4*self.magEZ)*10.**(-0.4*(MV-MVsun))\
                *2*fbeta/r_orbit.to('AU').value**2/u.arcsec**2
        
        return fEZ