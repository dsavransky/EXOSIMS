# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os, inspect, sys
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
import copy

# Python 3 compatibility:   
if sys.version_info[0] > 2: 
    xrange = range

class Stark(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.
    
    """

    def __init__(self, magZ=23., magEZ=22., varEZ=0., **specs):
        """
        """
        ZodiacalLight.__init__(self, magZ, magEZ, varEZ, **specs)
        self.logf = self.calclogf()
        self.points, self.values = self.calcfbetaInput()
        #Here we calculate the Zodiacal Light Model

    def fZ(self, Obs, TL, sInds, currentTimeAbs, mode):
        """Returns surface brightness of local zodiacal light
        
        Args:
            Obs (Observatory module):
                Observatory class object
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTimeAbs (astropy Time array):
                Current absolute mission time in MJD
            mode (dict):
                Selected observing mode
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        
        """

        # observatory positions vector in heliocentric ecliptic frame
        r_obs = Obs.orbit(currentTimeAbs, eclip=True)
        # observatory distance from heliocentric ecliptic frame center (projected in ecliptic plane)
        try:
            r_obs_norm = np.linalg.norm(r_obs[:,0:2], axis=1)
            # observatory ecliptic longitudes
            r_obs_lon = np.sign(r_obs[:,1])*np.arccos(r_obs[:,0]/r_obs_norm).to('deg').value # ensures the longitude is +/-180deg
        except:
            r_obs_norm = np.linalg.norm(r_obs[:,0:2], axis=1)*r_obs.unit
            # observatory ecliptic longitudes
            r_obs_lon = np.sign(r_obs[:,1])*np.arccos(r_obs[:,0]/r_obs_norm).to('deg').value # ensures the longitude is +/-180deg

        # longitude of the sun
        lon0 = (r_obs_lon + 180.) % 360. #turn into 0-360 deg heliocentric ecliptic longitude of spacecraft
        
        # target star positions vector in heliocentric true ecliptic frame
        r_targ = TL.starprop(sInds, currentTimeAbs, eclip=True)
        # target star positions vector wrt observatory in ecliptic frame
        r_targ_obs = (r_targ - r_obs).to('pc').value
        # tranform to astropy SkyCoordinates
        if sys.version_info[0] > 2:
            coord = SkyCoord(r_targ_obs[:,0], r_targ_obs[:,1], r_targ_obs[:,2],
                representation_type='cartesian').represent_as('spherical')
        else:
            coord = SkyCoord(r_targ_obs[:,0], r_targ_obs[:,1], r_targ_obs[:,2],
                representation='cartesian').represent_as('spherical')

        # longitude and latitude absolute values for Leinert tables
        lon = coord.lon.to('deg').value - lon0 # Get longitude relative to spacecraft
        lat = coord.lat.to('deg').value # Get latitude relative to spacecraft
        lon = abs((lon + 180.) % 360. - 180.) # converts to 0-180 deg
        lat = abs(lat)
        #technically, latitude is physically capable of being >90 deg
        
        #Interpolates 2D
        fbeta = griddata(self.points, self.values, list(zip(lon, lat)))
        
        lam = mode['lam'] # extract wavelength

        f = 10.**(self.logf(np.log10(lam.to('um').value)))*u.W/u.m**2/u.sr/u.um
        h = const.h                             # Planck constant
        c = const.c                             # speed of light in vacuum
        ephoton = h*c/lam/u.ph                  # energy of a photon
        F0 = TL.starF0(sInds, mode)             # zero-magnitude star (in ph/s/m2/nm)
        f_corr = f/ephoton/F0                   # color correction factor
        
        fZ = fbeta*f_corr.to('1/arcsec2')
        
        return fZ

    def calcfbetaInput(self):
        # table 17 in Leinert et al. (1998)
        # Zodiacal Light brightness function of solar LON (rows) and LAT (columns)
        # values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
        path = os.path.split(inspect.getfile(self.__class__))[0]
        Izod = np.loadtxt(os.path.join(path, 'Leinert98_table17.txt'))*1e-8 # W/m2/sr/um
        # create data point coordinates
        lon_pts = np.array([0., 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90,
                105, 120, 135, 150, 165, 180]) # deg
        lat_pts = np.array([0., 5, 10, 15, 20, 25, 30, 45, 60, 75, 90]) # deg
        y_pts, x_pts = np.meshgrid(lat_pts, lon_pts)
        points = np.array(list(zip(np.concatenate(x_pts), np.concatenate(y_pts))))
        # create data values, normalized by (90,0) value
        z = Izod/Izod[12,0]
        values = z.reshape(z.size)
        return  points, values

    def calclogf(self):
        """
        # wavelength dependence, from Table 19 in Leinert et al 1998
        # interpolated w/ a quadratic in log-log space
        Returns:
            interpolant (object):
                a 1D quadratic interpolant of intensity vs wavelength

        """
        self.zodi_lam = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5,
                4.8, 12, 25, 60, 100, 140]) # um
        self.zodi_Blam = np.array([2.5e-8, 5.3e-7, 2.2e-6, 2.6e-6, 2.0e-6, 1.3e-6,
                1.2e-6, 8.1e-7, 1.7e-7, 5.2e-8, 1.2e-7, 7.5e-7, 3.2e-7, 1.8e-8,
                3.2e-9, 6.9e-10]) # W/m2/sr/um
        x = np.log10(self.zodi_lam)
        y = np.log10(self.zodi_Blam)
        return interp1d(x, y, kind='quadratic')

    def calcfZmax(self, sInds, Obs, TL, TK, mode, hashname):
        """Finds the maximum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles
        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmax and fZmaxInds returned for
            Obs (module):
                Observatory module
            TL (TargetList object):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            valfZmax[sInds] (astropy Quantity array):
                the maximum fZ
            absTimefZmax[sInds] (astropy Time array):
                returns the absolute Time the maximum fZ occurs (for the prototype, these all have the same value)

        """
        #Generate cache Name########################################################################
        cachefname = hashname + 'fZmax'

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZmax from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                try:
                    tmpDat = pickle.load(f)
                except UnicodeDecodeError:
                    tmpDat = pickle.load(f,encoding='latin1')

                valfZmax = tmpDat[0,:]
                absTimefZmax = Time(tmpDat[1,:],format='mjd',scale='tai')
            return valfZmax[sInds]/u.arcsec**2, absTimefZmax[sInds]#, fZmaxInds

        #IF the fZmax File Does Not Exist, Calculate It
        else:
            self.vprint("Calculating fZmax")
            if not hasattr(self,'fZ_startSaved'):
                self.fZ_startSaved = self.generate_fZ(Obs, TL, TK, mode, hashname)

            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            
            #Generate Time array heritage from generate_fZ
            startTime = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs#Array of current times
            dt = 365.25/len(np.arange(1000))
            timeArray = [j*dt for j in np.arange(1000)]


            #Find maximum fZ of each star
            valfZmax = np.zeros(sInds.shape[0])
            indsfZmax = np.zeros(sInds.shape[0])
            relTimefZmax = np.zeros(sInds.shape[0])*u.d
            absTimefZmax = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs
            for i in xrange(len(sInds)):
                valfZmax[i] = min(fZ_matrix[i,:])#fZ_matrix has dimensions sInds 
                indsfZmax[i] = np.argmax(fZ_matrix[i,:])#Gets indices where fZmax occurs
                relTimefZmax[i] = TK.currentTimeNorm%(1*u.year).to('day') + indsfZmax[i]*dt*u.d
            absTimefZmax = TK.currentTimeAbs + relTimefZmax

            tmpDat = np.zeros([2,valfZmax.shape[0]])
            tmpDat[0,:] = valfZmax
            tmpDat[1,:] = absTimefZmax.value
            with open(cachefname, "wb") as fo:
                pickle.dump(tmpDat,fo)
                self.vprint("Saved cached fZmax to %s"%cachefname)
            return valfZmax/u.arcsec**2, absTimefZmax#, fZmaxInds

    def calcfZmin(self, sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
        """Finds the minimum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles
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
            if not hasattr(self,'fZ_startSaved'):
                self.fZ_startSaved = self.generate_fZ(Obs, TL, TK, mode, hashname)
            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            
            #Generate Time array heritage from generate_fZ
            startTime = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs#Array of current times
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
                
            # Find inds Entering, exiting ko
            #i = 0 # star ind
            fZQuads = list()
            for k in np.arange(len(sInds)):
                i = sInds[k] # Star ind
                indsEntering = list(np.where(np.diff(kogoodStart[:,i])==-1.)[0]) # double check this is entering
                indsExiting = np.where(np.diff(kogoodStart[:,i])==1.)[0]+1 # without the +1, this gives kogoodStart[indsExiting,i] = 0 meaning the stars are still in keepout
                indsExiting = [indsExiting[j]  if indsExiting[j] < len(kogoodStart[:,i])-1 else 0 for j in np.arange(len(indsExiting))] # need to ensure +1 increment doesnt exceed kogoodStart size

                # Find inds of local minima in fZ
                fZlocalMinInds = np.where(np.diff(np.sign(np.diff(fZ_matrix[i,:]))) > 0)[0] # Find local minima of fZ
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind,i]] # filter out local minimums based on those not occuring in keepout regions

                # Remove any indsEntering/indsExiting from fZlocalMinInds
                tmp1 = set(list(indsEntering)  + list(indsExiting)) # create set of just indsEntering and indsExiting
                fZlocalMinInds = list(set(list(fZlocalMinInds))-tmp1) # remove anything in tmp1 from fZlocalMinInds

                #Creates quads of fZ [type, value, timeOfYear, AbsTime]
                #0 - entering, 1 - exiting, 2 - local minimum
                dt = 365.25/len(np.arange(1000))
                enteringQuad = [[0,\
                                    fZ_matrix[i,indsEntering[j]],\
                                    timeArray[indsEntering[j]],\
                                    (TK.currentTimeAbs.copy() + TK.currentTimeNorm%(1.*u.year).to('day') + indsEntering[j]*dt*u.d).value] for j in np.arange(len(indsEntering))]
                exitingQuad = [[1,\
                                    fZ_matrix[i,indsExiting[j]],\
                                    timeArray[indsExiting[j]],\
                                    (TK.currentTimeAbs.copy() + TK.currentTimeNorm%(1.*u.year).to('day') + indsExiting[j]*dt*u.d).value] for j in np.arange(len(indsExiting))]
                fZlocalMinIndsQuad = [[2,\
                                        fZ_matrix[i,fZlocalMinInds[j]],\
                                        timeArray[fZlocalMinInds[j]],\
                                        (TK.currentTimeAbs.copy() + TK.currentTimeNorm%(1.*u.year).to('day') + fZlocalMinInds[j]*dt*u.d).value] for j in np.arange(len(fZlocalMinInds))]

                # Assemble Quads 
                fZQuads.append(enteringQuad + exitingQuad + fZlocalMinIndsQuad)

            with open(cachefname, "wb") as fo:
                pickle.dump(fZQuads,fo)
                self.vprint("Saved cached fZQuads to %s"%cachefname)

            #Convert Abs time to MJD object
            for i in np.arange(len(fZQuads)):
                for j in np.arange(len(fZQuads[i])):
                    fZQuads[i][j][3] = Time(fZQuads[i][j][3],format='mjd',scale='tai')
                    fZQuads[i][j][1] = fZQuads[i][j][1]/u.arcsec**2.
            # fZQuads has shape [sInds][Number fZmin][4]
            return [fZQuads[i] for i in sInds] #valfZmin, absTimefZmin

    def extractfZmin_fZQuads(self,fZQuads):
        """ Extract the global fZminimum from fZQuads
        *This produces the same output as calcfZmin circa January 2019
            Args:
                fZQuads (list) - fZQuads has shape [sInds][Number fZmin][4]
            Returns:
                tuple:
                valfZmin (astropy Quantity array) - fZ minimum for the target
                absTimefZmin (astropy Time array) - Absolute time the fZmin occurs
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
        return np.asarray(valfZmin)/u.arcsec**2., Time(np.asarray(absTimefZmin),format='mjd',scale='tai')
