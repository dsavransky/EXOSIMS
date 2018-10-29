# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
import os
try:
    import cPickle as pickle
except:
    import pickle

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
        cachedir (str):
            Path to cache directory
        
    """

    _modtype = 'ZodiacalLight'
    
    def __init__(self, magZ=23, magEZ=22, varEZ=0, cachedir=None, **specs):

        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        self.magZ = float(magZ)         # 1 zodi brightness (per arcsec2)
        self.magEZ = float(magEZ)       # 1 exo-zodi brightness (per arcsec2)
        self.varEZ = float(varEZ)       # exo-zodi variation (variance of log-normal dist)
        self.fZ0 = 10**(-0.4*self.magZ)/u.arcsec**2   # default zodi brightness
        self.fEZ0 = 10**(-0.4*self.magEZ)/u.arcsec**2 # default exo-zodi brightness
        
        assert self.varEZ >= 0, "Exozodi variation must be >= 0"
        
        # populate outspec
        for att in self.__dict__.keys():
            if att not in ['vprint','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

    def __str__(self):
        """String representation of the Zodiacal Light object
        
        When the command 'print' is used on the Zodiacal Light object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Zodiacal Light class object attributes'

    def fZ(self, Obs, TL, sInds, currentTimeAbs, mode):
        """Returns surface brightness of local zodiacal light
        
        Args:
            Obs (Observatory module):
                Observatory class object
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTimeAbs (astropy Time quantity):
                absolute time to evaluate fZ for
            mode (dict):
                Selected observing mode
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        
        """
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get all array sizes
        nStars = sInds.size
        nTimes = currentTimeAbs.size
        assert nStars == 1 or nTimes == 1 or nTimes == nStars, \
                "If multiple times and targets, currentTimeAbs and sInds sizes must match."
        
        nZ = np.ones(np.maximum(nStars, nTimes))
        fZ = nZ*10**(-0.4*self.magZ)/u.arcsec**2
        
        return fZ

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
        
        # assume log-normal distribution of variance
        nEZ = np.ones(len(MV))
        if self.varEZ != 0:
            mu = np.log(nEZ) - 0.5*np.log(1. + self.varEZ/nEZ**2)
            v = np.sqrt(np.log(self.varEZ/nEZ**2 + 1.))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=len(MV))
        
        # supplementary angle for inclination > 90 degrees
        beta = I.to('deg').value
        mask = np.where(beta > 90)[0]
        beta[mask] = 180.0 - beta[mask]
        beta = 90.0 - beta

        fbeta = 2.44 - 0.0403*beta + 0.000269*beta**2
        
        fEZ = nEZ*10**(-0.4*self.magEZ)*10.**(-0.4*(MV - 
                MVsun))*2*fbeta/d.to('AU').value**2/u.arcsec**2
        
        return fEZ

    def generate_fZ(self, Obs, TL, TK, mode, hashname):
        """Calculates fZ values for all stars over an entire orbit of the sun
        Args:
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Updates Attributes:
            fZ_startSaved[1000, TL.nStars] (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2 for each star over 1 year at discrete points defined by resolution
        """
        #Generate cache Name########################################################################
        cachefname = hashname+'starkfZ'

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZ from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                tmpfZ = pickle.load(f)
            return tmpfZ

        #IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        else:
            self.vprint("Calculating fZ")
            #OS = self.OpticalSystem#Testing to be sure I can remove this
            #WA = OS.WA0#Testing to be sure I can remove this
            sInds= np.arange(TL.nStars)
            startTime = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs#Array of current times
            resolution = [j for j in range(1000)]
            fZ = np.zeros([sInds.shape[0], len(resolution)])
            dt = 365.25/len(resolution)*u.d
            for i in xrange(len(resolution)):#iterate through all times of year
                time = startTime + dt*resolution[i]
                fZ[:,i] = self.fZ(Obs, TL, sInds, time, mode)
            
            with open(cachefname, "wb") as fo:
                pickle.dump(fZ,fo)
                self.vprint("Saved cached 1st year fZ to %s"%cachefname)
            return fZ

    def calcfZmax(self, sInds, Obs, TL, TK, mode, hashname):
        """Finds the maximum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles.
         (prototype includes keepout angles because the values are all the same)
        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmax and fZmaxInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            valfZmax[sInds] (astropy Quantity array):
                the maximum fZ (for the prototype, these all have the same value) with units 1/arcsec**2
            absTimefZmax[sInds] (astropy Time array):
                returns the absolute Time the maximum fZ occurs (for the prototype, these all have the same value)
        """
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get all array sizes
        nStars = sInds.size

        nZ = np.ones(nStars)
        valfZmax = nZ*10**(-0.4*self.magZ)/u.arcsec**2

        absTimefZmax = nZ*u.d + TK.currentTimeAbs

        return valfZmax[sInds], absTimefZmax[sInds]

    def calcfZmin(self,sInds, Obs, TL, TK, mode, hashname):
        """Finds the minimum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles. 
        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmin and fZminInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            valfZmin[sInds] (astropy Quantity array):
                the minimum fZ (for the prototype, these all have the same value) with units 1/arcsec**2
            absTimefZmin[sInds] (astropy Time array):
                returns the absolute Time the minimum fZ occurs (for the prototype, these all have the same value)
        """
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get all array sizes
        nStars = sInds.size

        nZ = np.ones(nStars)
        valfZmin = nZ*10**(-0.4*self.magZ)/u.arcsec**2

        absTimefZmin = nZ*u.d + TK.currentTimeAbs

        return valfZmin[sInds], absTimefZmin[sInds]
