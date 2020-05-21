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
import sys
from astropy.time import Time

# Python 3 compatibility:
if sys.version_info[0] > 2:
    xrange = range

class ZodiacalLight(object):
    """Zodiacal Light class template
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation.
    
    Args:
        specs:
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
    
    def __init__(self, magZ=23, magEZ=22, varEZ=0, cachedir=None, commonSystemfEZ=False, **specs):

        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 
     
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        self.magZ = float(magZ)         # 1 zodi brightness (per arcsec2)
        self.magEZ = float(magEZ)       # 1 exo-zodi brightness (per arcsec2)
        self.varEZ = float(varEZ)       # exo-zodi variation (variance of log-normal dist)
        self.fZ0 = 10**(-0.4*self.magZ)/u.arcsec**2   # default zodi brightness
        self.fEZ0 = 10**(-0.4*self.magEZ)/u.arcsec**2 # default exo-zodi brightness
        
        assert self.varEZ >= 0, "Exozodi variation must be >= 0"
        
        #### Common Star System Number of Exo-zodi
        self.commonSystemfEZ = commonSystemfEZ #ZL.nEZ must be calculated in SU
        self._outspec['commonSystemfEZ'] = self.commonSystemfEZ

        # populate outspec
        for att in self.__dict__:
            if att not in ['vprint','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

    def __str__(self):
        """String representation of the Zodiacal Light object
        
        When the command 'print' is used on the Zodiacal Light object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__:
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
            astropy Quantity array:
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
            astropy Quantity array:
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
        # assume log-normal distribution of variance
        nEZ = np.ones(nStars)
        if self.varEZ != 0:
            mu = np.log(nEZ) - 0.5*np.log(1. + self.varEZ/nEZ**2)
            v = np.sqrt(np.log(self.varEZ/nEZ**2 + 1.))
            nEZ = np.random.lognormal(mean=mu, sigma=v, size=nStars)

        return nEZ

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
        #Generate cache Name#########################################################
        cachefname = hashname+'starkfZ'

        #Check if file exists########################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZ from %s"%cachefname)
            try:
                with open(cachefname, "rb") as ff:
                    tmpfZ = pickle.load(ff)
            except UnicodeDecodeError:
                with open(cachefname, "rb") as ff:
                    tmpfZ = pickle.load(ff,encoding='latin1')
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
        
        Note:
            Prototype includes keepout angles because the values are all the same

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
            tuple:
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

    def calcfZmin(self,sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
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
            koMap (boolean ndarray):
                True is a target unobstructed and observable, and False is a 
                target unobservable due to obstructions in the keepout zone.
            koTimes (astropy Time ndarray):
                Absolute MJD mission times from start to end in steps of 1 d
                
        Returns:
            list:
                list of local zodiacal light minimum and times they occur at (should all have same value for prototype)
        """
        
        #Generate cache Name########################################################################
        cachefname = hashname + 'fZmin'

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZQuads from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                try:
                    fZQuads = pickle.load(f)  # of form tmpDat len sInds, tmpDat[0] len # of ko enter/exits and localmin occurences, tmpDat[0,0] form [type,fZvalue,absTime]
                except UnicodeDecodeError:
                    fZQuads = pickle.load(f,encoding='latin1')  # of form tmpDat len sInds, tmpDat[0] len # of ko enter/exits and localmin occurences, tmpDat[0,0] form [type,fZvalue,absTime]

                #Convert Abs time to MJD object
                for i in np.arange(len(fZQuads)):
                    for j in np.arange(len(fZQuads[i])):
                        fZQuads[i][j][3] = Time(fZQuads[i][j][3],format='mjd',scale='tai')
                        fZQuads[i][j][1] = fZQuads[i][j][1]/u.arcsec**2.

            return [fZQuads[i] for i in sInds]
        else:

            # cast sInds to array
            sInds = np.array(sInds, ndmin=1, copy=False)
            # get all array sizes
            nStars = sInds.size

            nZ = np.ones(nStars)
            valfZmin = nZ*10**(-0.4*self.magZ)/u.arcsec**2

            absTimefZmin = nZ*u.d + TK.currentTimeAbs


            if not hasattr(self,'fZ_startSaved'):
                self.fZ_startSaved = self.generate_fZ(Obs, TL, TK, mode, hashname)
            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            dt = 365.25/len(np.arange(1000))
            timeArray = [j*dt for j in np.arange(1000)]
            timeArrayAbs = TK.currentTimeAbs + timeArray*u.d

            #When are stars in KO regions
            missionLife = TK.missionLife.to('yr')
            # if this is being calculated without a koMap, or if missionLife is less than a year
            if (koMap is None) or (missionLife.value < 1):
                # calculating keepout angles and keepout values for 1 system in mode
                koStr     = list(filter(lambda syst: syst.startswith('koAngles_') , mode['syst'].keys()))
                koangles  = np.asarray([mode['syst'][k] for k in koStr]).reshape(1,4,2)
                kogoodStart = Obs.keepout(TL, sInds, timeArrayAbs, koangles)[0].T
            else:
                # getting the correct koTimes to look up in koMap
                assert koTimes != None, "koTimes not included in input statement."
                koInds = np.zeros(len(timeArray),dtype=int)
                for x in np.arange(len(timeArray)):
                    koInds[x] = np.where( np.round( (koTimes - timeArrayAbs[x]).value ) == 0 )[0][0]
                # determining ko values within a year using koMap
                kogoodStart = koMap[:,koInds].T

            fZQuads = list()
            for k in np.arange(len(sInds)):
                i = sInds[k] # Star ind
                # Find inds of local minima in fZ
                fZlocalMinInds = np.where(np.diff(np.sign(np.diff(fZ_matrix[i,:]))) > 0)[0] # Find local minima of fZ
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind,i]] # filter out local minimums based on those not occuring in keepout regions
                if fZlocalMinInds == []: #This happens in prototype module. Caused by all values in fZ_matrix being the same
                    fZlocalMinInds = [0]


                fZlocalMinIndsQuad = [[2,\
                            fZ_matrix[i,fZlocalMinInds[j]],\
                            timeArray[fZlocalMinInds[j]],\
                            (TK.currentTimeAbs.copy() + TK.currentTimeNorm%(1.*u.year).to('day') + fZlocalMinInds[j]*dt*u.d).value] for j in np.arange(len(fZlocalMinInds))]
                fZQuads.append(fZlocalMinIndsQuad)

            with open(cachefname, "wb") as fo:
                pickle.dump(fZQuads,fo)
                self.vprint("Saved cached fZQuads to %s"%cachefname)

            #Convert Abs time to MJD object
            for i in np.arange(len(fZQuads)):
                for j in np.arange(len(fZQuads[i])):
                    fZQuads[i][j][3] = Time(fZQuads[i][j][3],format='mjd',scale='tai')
                    fZQuads[i][j][1] = fZQuads[i][j][1]/u.arcsec**2.

            return [fZQuads[i] for i in sInds]

    def extractfZmin_fZQuads(self,fZQuads):
        """ Extract the global fZminimum from fZQuads
            
        *This produces the same output as calcfZmin circa January 2019
        
        Note: for the prototype, fZQuads is equivalent to (valfZmin, absTimefZmin) so we simply return that
            Args:
                fZQuads (list):
                    fZQuads has shape [sInds][Number fZmin][4]
            Returns:
                valfZmin (astropy Quantity array):
                    fZ minimum for the target
                absTimefZmin (astropy Time array):
                    Absolute time the fZmin occurs
        """
        valfZmin = list()
        absTimefZmin = list()
        for i in np.arange(len(fZQuads)):#Iterates over each star
            ffZmin = 100.
            fabsTimefZmin = 0.
            for j in np.arange(len(fZQuads[i])): # Iterates over each occurence of a minimum
                if fZQuads[i][j][1].value < ffZmin:
                    ffZmin = fZQuads[i][j][1].value
                    fabsTimefZmin = fZQuads[i][j][3].value
            valfZmin.append(ffZmin)
            absTimefZmin.append(fabsTimefZmin)
        #ADD AN ASSERT CHECK TO ENSURE NO FFZMIN=100 AND NO FABSTIMEFZMIN=0.
        #The np.asarray and Time must occur to create astropy Quantity arrays and astropy Time arrays
        # for i in np.arange(len(fZQuads)):
        #     valfZmin = fZQuads[i][1]
        #     absTimefZmin = fZQuads[i][3]
        # return np.asarray(valfZmin), np.asarray(absTimefZmin)
        return np.asarray(valfZmin)/u.arcsec**2., Time(np.asarray(absTimefZmin),format='mjd',scale='tai')

