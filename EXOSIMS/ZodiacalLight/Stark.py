# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os, inspect
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d, griddata

class Stark(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014."""

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
        
        # observatory coordinates wrt Sun
        rsc = SkyCoord(r_sc[:,0],r_sc[:,1],r_sc[:,2],representation='cartesian').heliocentrictrueecliptic
        # longitude of the sun
        lon0 = rsc.lon.value + 180
        
        # target coordinates wrt observatory
        x = TL.coords[sInds].represent_as('cartesian').x - r_sc[:,0]
        y = TL.coords[sInds].represent_as('cartesian').y - r_sc[:,1]
        z = TL.coords[sInds].represent_as('cartesian').z - r_sc[:,2]
        rt = SkyCoord(x,y,z,representation='cartesian').heliocentrictrueecliptic
        # longitudes and latitudes of targets
        lon = rt.lon.value - lon0
        lat = rt.lat.value
        lon = abs((lon+180)%360-180)
        lat = abs(lat)
        
        # Table 17 in Leinert et al. (1998)
        # Zodiacal light brightness function of solar longitudes (rows) and beta values (columns)
        # Values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
        path = os.path.split(inspect.getfile(self.__class__))[0]
        Izod = np.loadtxt(os.path.join(path, 'Leinert98_table17.txt'))*1e-8 # W/m2/sr/um
        # create data point coordinates
        lon_pts = np.array([0.,5,10,15,20,25,30,35,40,45,60,75,90,105,120,135,150,165,180]) # deg
        lat_pts = np.array([0.,5,10,15,20,25,30,45,60,75,90]) # deg
        y_pts,x_pts = np.meshgrid(lat_pts,lon_pts)
        points = np.array(zip(np.concatenate(x_pts),np.concatenate(y_pts)))
        # create data values, normalized by (90,0) value
        z = Izod/Izod[12,0]
        values = z.reshape(z.size)
        # interpolates 2D
        fbeta = griddata(points,values,zip(lon,lat))
        
        # wavelength dependence, from Table 19 in Leinert et al 1998
        # interpolated w/ a quadratic in log-log space
        zodi_lam = np.array([0.2,0.3,0.4,0.5,0.7,0.9,1.0,1.2,2.2,3.5,4.8,12,25,60,100,140]) # um
        zodi_Blam = np.array([2.5e-8,5.3e-7,2.2e-6,2.6e-6,2.0e-6,1.3e-6,1.2e-6,8.1e-7,\
                1.7e-7,5.2e-8,1.2e-7,7.5e-7,3.2e-7,1.8e-8,3.2e-9,6.9e-10]) # W/m2/sr/um
        x = np.log10(zodi_lam)
        y = np.log10(zodi_Blam)
        f_corr = 10.**(interp1d(x,y,kind='quadratic')(np.log10(lam.to('um').value)))\
                *u.W/u.m**2/u.sr/u.um
        h = const.h                             # Planck constant
        c = const.c                             # speed of light in vacuum
        ephoton = h*c/lam/u.ph                  # energy of a photon
        F0 = TL.OpticalSystem.F0(lam)           # zero-magnitude star (in ph/s/m2/nm)
        f_corr /= (ephoton * F0)                # color correction factor
        
        fZ = fbeta * f_corr.to('1/arcsec2')
        
        return fZ
