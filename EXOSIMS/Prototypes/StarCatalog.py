# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord
import numpy as np

class StarCatalog(object):
    """Star Catalog class template
     
    This class contains all variables and methods necessary to perform
    Star Catalog Module calculations in exoplanet mission simulation.
    
    Attributes:
        Name (ndarray):
            1D numpy ndarray of star names
        Type (ndarray):
            1D numpy ndarray of star types
        Spec (ndarray):
            1D numpy ndarray of spectral types
        parx (ndarray):
            1D numpy ndarray of parallax in milliarcseconds
        Umag (ndarray):
            1D numpy ndarray of U magnitude
        Bmag (ndarray):
            1D numpy ndarray of B magnitude
        Vmag (ndarray):
            1D numpy ndarray of V magnitude
        Rmag (ndarray):
            1D numpy ndarray of R magnitude
        Imag (ndarray):
            1D numpy ndarray of I magnitude
        Jmag (ndarray):
            1D numpy ndarray of J magnitude
        Hmag (ndarray):
            1D numpy ndarray of H magnitude
        Kmag (ndarray):
            1D numpy ndarray of K magnitude
        dist (ndarray):
            1D numpy ndarray of distance in parsecs to star
        BV (ndarray):
            1D numpy ndarray of B-V Johnson magnitude
        MV (ndarray):
            1D numpy ndarray of absolute V magnitude
        BC (ndarray):
            1D numpy ndarray of bolometric correction
        L (ndarray):
            1D numpy ndarray of stellar luminosity in Solar luminosities
        coords (SkyCoord):
            numpy ndarray of astropy SkyCoord objects containing right ascension
            and declination in degrees
        pmra (ndarray):
            1D numpy ndarray of proper motion in right ascension in
            milliarcseconds/year
        pmdec (ndarray):
            1D numpy ndarray of proper motion in declination in 
            milliarcseconds/year
        rv (ndarray):
            1D numpy ndarray of radial velocity in km/s
        Binary_Cut (ndarray):
            1D numpy ndarray of booleans where True is a star with a companion 
            closer than 10 arcsec
        
    """ 

    _modtype = 'StarCatalog'
    _outspec = {}

    def __init__(self,ntargs=0,**specs):
        
        self.Name = np.zeros(ntargs) # list of star names
        self.Type = np.zeros(ntargs) # list of star types
        self.Spec = np.zeros(ntargs) # list of spectral types
        self.parx = np.zeros(ntargs) # list of parallax in milliarcseconds
        self.Umag = np.zeros(ntargs) # list of U magnitude
        self.Bmag = np.zeros(ntargs) # list of B magnitude
        self.Vmag = np.zeros(ntargs) # list of V magnitude
        self.Rmag = np.zeros(ntargs) # list of R magnitude
        self.Imag = np.zeros(ntargs) # list of I magnitude
        self.Jmag = np.zeros(ntargs) # list of J magnitude
        self.Hmag = np.zeros(ntargs) # list of H magnitude
        self.Kmag = np.zeros(ntargs) # list of K magnitude
        self.dist = np.zeros(ntargs) # list of distance in parsecs (dist = 1000/parx)
        self.BV = np.zeros(ntargs) # list of B-V Johnson magnitude
        self.MV = np.zeros(ntargs) # list of absolute V magnitude (MV = -5 * log(1000/parx) + 5 - Vmag) 
        self.BC = np.zeros(ntargs) # list of bolometric correction
        self.L = np.zeros(ntargs) # list of stellar luminosity in Solar luminosities
        # list of astropy SkyCoord objects of right ascension and declination in degrees        
        self.coords = SkyCoord(ra=np.zeros(ntargs), dec=np.zeros(ntargs), unit='deg')
        self.pmra = np.zeros(ntargs) # list of proper motion in right ascension in milliarcseconds/year
        self.pmdec = np.zeros(ntargs) # list of proper motion in declination in milliarcseconds/year
        self.rv = np.zeros(ntargs) # list of radial velocity in kilometers/second
        self.Binary_Cut = np.zeros(ntargs,dtype=bool) # boolean list where True is companion closer than 10 arcsec

    def __str__(self):
        """String representation of the StarCatalog object
        
        When the command 'print' is used on the StarCatalog object, this method
        will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Star Catalog class object attributes'
