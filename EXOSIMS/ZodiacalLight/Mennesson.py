# -*- coding: utf-8 -*-
from EXOSIMS.ZodiacalLight.Stark import Stark
import numpy as np
import os, inspect
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, interp1d
try:
    import cPickle as pickle
except:
    import pickle
from numpy import nan
from astropy.time import Time
from astropy.io import fits

class Mennesson(Stark):
    """Mennesson Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.
    
    """

    def __init__(self, EZ_distribution="nominal_maxL_distribution.fits", **specs):
        Stark.__init__(self, **specs)
        if os.path.exists(EZ_distribution):
            self.EZ_distribution = EZ_distribution
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), EZ_distribution)):
            self.EZ_distribution = os.path.join(os.path.dirname(os.path.abspath(__file__)), EZ_distribution)
        self.fitsdata = fits.open(self.EZ_distribution)[0].data


    def fEZ(self, MV, I, d):
        """Returns surface brightness of exo-zodiacal light
        
        Args:
            MV (integer ndarray):
                Apparent magnitude of the star (in the V band)
            I (astropy Quantity array):
                Inclination of the planets of interest in units of deg
            d (astropy Quantity nx3 array):
                Distance to star of the planets of interest in units of AU
        
        Returns:
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
        
        """
        
        # apparent magnitude of the star (in the V band)
        MV = np.array(MV, ndmin=1, copy=False)
        # apparent magnitude of the Sun (in the V band)
        MVsun = 4.83
        
        if self.commonSystemfEZ:
            nEZ = self.nEZ
        else:
            nEZ = self.gen_systemnEZ(len(MV))

        # supplementary angle for inclination > 90 degrees
        beta = I.to('deg').value
        mask = np.where(beta > 90)[0]
        beta[mask] = 180.0 - beta[mask]
        beta = 90.0 - beta

        fbeta = 2.44 - 0.0403*beta + 0.000269*beta**2 
        fbeta = fbeta/1.473 # 1.473 is adjustment for inputs being for 60 deg. inclination
        
        fEZ = nEZ*10**(-0.4*self.magEZ)*10.**(-0.4*(MV - 
                MVsun))*2*fbeta/d.to('AU').value**2/u.arcsec**2
        
        return fEZ

    def gen_systemnEZ(self, nStars):
        """ Ranomly generates the number of Exo-Zodi
        Args:
            nStars (int):
                number of exo-zodi to generate
        Returns:
            nEZ (numpy array):
                numpy array of exo-zodi randomly selected from fitsdata
        """
        nEZ_seed = np.random.randint(len(self.fitsdata) - nStars)
        nEZ = self.fitsdata[nEZ_seed:(nEZ_seed + nStars)]
        return nEZ
