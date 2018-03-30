# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os, inspect
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, CubicSpline #interp1d, DELETE ME OLD INTERPOLANT

class Stark(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.
    
    """

    def fZ(self, Obs, TL, sInds, currentTime, mode):
        """Returns surface brightness of local zodiacal light
        
        Args:
            Obs (Observatory module):
                Observatory class object
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            mode (dict):
                Selected observing mode
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        
        """
        
        # observatory positions vector in heliocentric ecliptic frame
        r_obs = Obs.orbit(currentTime, eclip=True)
        # observatory distances (projected in ecliptic plane)
        r_obs_norm = np.linalg.norm(r_obs[:,0:2], axis=1)*r_obs.unit
        # observatory ecliptic longitudes
        r_obs_lon = np.sign(r_obs[:,1])*np.arccos(r_obs[:,0]/r_obs_norm).to('deg').value
        # longitude of the sun
        lon0 = (r_obs_lon + 180) % 360
        
        # target star positions vector in heliocentric true ecliptic frame
        r_targ = TL.starprop(sInds, currentTime, eclip=True)
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
