# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os, inspect
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, CubicSpline #interp1d, DELETE ME OLD INTERPOLANT
try:
    import cPickle as pickle
except:
    import pickle

class Stark(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.
    
    """

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
        # observatory distances (projected in ecliptic plane)
        r_obs_norm = np.linalg.norm(r_obs[:,0:2], axis=1)*r_obs.unit
        # observatory ecliptic longitudes
        r_obs_lon = np.sign(r_obs[:,1])*np.arccos(r_obs[:,0]/r_obs_norm).to('deg').value
        # longitude of the sun
        lon0 = (r_obs_lon + 180) % 360
        
        # target star positions vector in heliocentric true ecliptic frame
        r_targ = TL.starprop(sInds, currentTimeAbs, eclip=True)
        # target star positions vector wrt observatory in ecliptic frame
        r_targ_obs = (r_targ - r_obs).to('pc').value
        # tranform to astropy SkyCoordinates
        coord = SkyCoord(r_targ_obs[:,0], r_targ_obs[:,1], r_targ_obs[:,2],
                representation='cartesian').represent_as('spherical')
        # longitude and latitude absolute values for Leinert tables
        lon = coord.lon.to('deg').value - lon0
        lat = coord.lat.to('deg').value
        lon = abs((lon + 180) % 360 - 180)
        lat = abs(lat)
        
        # table 17 in Leinert et al. (1998)
        # Zodiacal Light brightness function of solar LON (rows) and LAT (columns)
        # values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
        path = os.path.split(inspect.getfile(self.__class__))[0]
        Izod = np.loadtxt(os.path.join(path, 'Leinert98_table17.txt'))*1e-8 # W/m2/sr/um
        # create data point coordinates
        lon_pts = np.array([0., 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90,
                105, 120, 135, 150, 165, 180]) # deg ecliptic longitude with 0 at sun
        lat_pts = np.array([0., 5, 10, 15, 20, 25, 30, 45, 60, 75, 90]) # deg ecliptic latitude with 0 at sun
        y_pts, x_pts = np.meshgrid(lat_pts, lon_pts)
        points = np.array(zip(np.concatenate(x_pts), np.concatenate(y_pts)))
        # create data values, normalized by (90,0) value
        z = Izod/Izod[12,0]
        values = z.reshape(z.size)
        # interpolates 2D
        fbeta = griddata(points, values, zip(lon, lat))
        
        # wavelength dependence, from Table 19 in Leinert et al 1998
        # interpolated w/ a quadratic in log-log space
        lam = mode['lam']
        zodi_lam = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5,
                4.8, 12, 25, 60, 100, 140]) # um
        zodi_Blam = np.array([2.5e-8, 5.3e-7, 2.2e-6, 2.6e-6, 2.0e-6, 1.3e-6,
                1.2e-6, 8.1e-7, 1.7e-7, 5.2e-8, 1.2e-7, 7.5e-7, 3.2e-7, 1.8e-8,
                3.2e-9, 6.9e-10]) # W/m2/sr/um
        x = np.log10(zodi_lam)
        y = np.log10(zodi_Blam)
        #logf = interp1d(x, y, kind='quadratic')#DELETE ME OLD INTERPOLANT
        logf = CubicSpline(x, zodi_Blam, bc_type='clamped')
        f = logf(np.log10(lam.to('um').value))*u.W/u.m**2/u.sr/u.um
        #f = 10.**(logf(np.log10(lam.to('um').value)))*u.W/u.m**2/u.sr/u.um#DELETE ME OLD INTERPOLANT
        h = const.h                             # Planck constant
        c = const.c                             # speed of light in vacuum
        ephoton = h*c/lam/u.ph                  # energy of a photon
        F0 = TL.OpticalSystem.F0(lam)           # zero-magnitude star (in ph/s/m2/nm)
        f_corr = f/ephoton/F0                   # color correction factor
        
        fZ = fbeta*f_corr.to('1/arcsec2')
        
        return fZ

    def calcfZmax(self, sInds, Obs, TL, currentTimeAbs, mode, hashname):
        """Finds the maximum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles
        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmax and fZmaxInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            currentTimeAbs (astropy Time array):
                current absolute time im MJD
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            fZmax[sInds] (astropy Quantity array):
                the maximum fZ
        """
        #Generate cache Name########################################################################
        cachefname = hashname + 'fZmax'

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZmax from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                tmpDat = pickle.load(f)
                fZmax = tmpDat[0,:]
                fZmaxInds = tmpDat[1,:]
            return fZmax#, fZmaxInds

        #IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        else:
            self.vprint("Calculating fZmax")
            if not hasattr(self,'fZ_startSaved'):
                self.fZ_startSaved = self.generate_fZ(Obs, TL, currentTimeAbs, mode, hashname)

            #DELETE fZ_startSaved = self.fZ_startSaved#fZ_startSaved[sInds,1000] - the fZ for each sInd for 1 year separated into 1000 timesegments
            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            
            #Generate Time array heritage from generate_fZ
            startTime = np.zeros(sInds.shape[0])*u.d + currentTimeAbs#Array of current times
            dt = 365.25/len(np.arange(1000))
            timeArray = [j*dt for j in range(1000)]
                
            #When are stars in KO regions
            kogoodStart = np.zeros([len(timeArray),sInds.shape[0]])#replaced self.schedule with sInds
            for i in np.arange(len(timeArray)):
                kogoodStart[i,:] = Obs.keepout(TL, sInds, TK.currentTimeAbs+timeArray[i]*u.d)#replaced self.schedule with sInds
                kogoodStart[i,:] = (np.zeros(kogoodStart[i,:].shape[0])+1)*kogoodStart[i,:]
            kogoodStart[kogoodStart==0] = nan

            #Filter Out fZ where star is in KO region

            #Find maximum fZ of each star
            fZmax = np.zeros(sInds.shape[0])
            fZmaxInds = np.zeros(sInds.shape[0])
            for i in xrange(len(sInds)):
                fZmax[i] = min(fZ_matrix[i,:])
                fZmaxInds[i] = np.argmax(fZ_matrix[i,:])

            tmpDat = np.zeros([2,fZmax.shape[0]])
            tmpDat[0,:] = fZmax
            tmpDat[1,:] = fZmaxInds
            with open(cachefname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                pickle.dump(tmpDat,fo)
                self.vprint("Saved cached fZmax to %s"%cachefname)
            return fZmax#, fZmaxInds

    def calcfZmin(self, sInds, Obs, TL, currentTimeAbs, mode, hashname):
        """Finds the minimum zodiacal light values for each star over an entire orbit of the sun not including keeoput angles
        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmin and fZminInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            currentTimeAbs (astropy Time array):
                current absolute time im MJD
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            fZmin[sInds] (astropy Quantity array):
                the minimum fZ
        """
        if not hasattr(self,'fZ_startSaved'):
            self.fZ_startSaved = self.generate_fZ(Obs, TL, currentTimeAbs, mode, hashname)

        #DELETE fZ_startSaved = self.fZ_startSaved#fZ_startSaved[sInds,1000] - the fZ for each sInd for 1 year separated into 1000 timesegments
        tmpfZ = np.asarray(self.fZ_startSaved)#convert into an array
        fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
        #Find minimum fZ of each star
        fZmin = np.zeros(sInds.shape[0])
        fZminInds = np.zeros(sInds.shape[0])
        for i in xrange(len(sInds)):
            fZmin[i] = min(fZ_matrix[i,:])
            fZminInds[i] = np.argmin(fZ_matrix[i,:])

        return fZmin/u.arcsec**2 #fZminInds
