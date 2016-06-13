# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

class StarCatalog(object):
    """Star Catalog class template
     
    This class contains all variables and methods necessary to perform
    Star Catalog Module calculations in exoplanet mission simulation.
    
    Attributes:
        Name (string ndarray):
            Star names
        Spec (string ndarray):
            Star spectral types
        Umag (float ndarray):
            U magnitude
        Bmag (float ndarray):
            B magnitude
        Vmag (float ndarray):
            V magnitude
        Rmag (float ndarray):
            R magnitude
        Imag (float ndarray):
            I magnitude
        Jmag (float ndarray):
            J magnitude
        Hmag (float ndarray):
            H magnitude
        Kmag (float ndarray):
            K magnitude
        BV (float ndarray):
            B-V Johnson magnitude
        MV (float ndarray):
            Absolute V magnitude
        BC (float ndarray):
            Bolometric correction
        L (float ndarray):
            Stellar luminosity in Solar luminosities
        Binary_Cut (boolean ndarray):
            Booleans where True is a star with a companion closer than 10 arcsec
        dist (astropy Quantity array):
            Distance to star in units of pc
        parx (astropy Quantity array):
            Parallax in units of mas
        coords (astropy SkyCoord array):
            SkyCoord objects containing right ascension, declination, and 
            distance to star in units of deg, deg, and pc
        pmra (astropy Quantity array):
            Proper motion in right ascension in units of mas/year
        pmdec (astropy Quantity array):
            Proper motion in declination in units of mas/year
        rv (astropy Quantity array):
            Radial velocity in units of km/s
        
    """

    _modtype = 'StarCatalog'
    _outspec = {}

    def __init__(self, ntargs=0, **specs):
        
        # list of astropy attributes
        self.dist = np.ones(ntargs)*u.pc # distance in parsecs (dist = 1000/parx)
        self.parx = self.dist.to('mas',equivalencies=u.parallax()) # parallax in milliarcseconds
        self.coords = SkyCoord(ra=np.zeros(ntargs)*u.deg, dec=np.zeros(ntargs)*u.deg, \
                distance=self.dist) # right ascension and declination in degrees
        self.pmra = np.zeros(ntargs)*u.mas/u.yr # proper motion in right ascension in milliarcseconds/year
        self.pmdec = np.zeros(ntargs)*u.mas/u.yr # proper motion in declination in milliarcseconds/year
        self.rv = np.zeros(ntargs)*u.km/u.s # radial velocity in kilometers/second
        
        # list of non-astropy attributes
        self.Name = np.zeros(ntargs) # list of star names
        self.Spec = np.zeros(ntargs) # list of spectral types
        self.Umag = np.zeros(ntargs) # list of U magnitude
        self.Bmag = np.zeros(ntargs) # list of B magnitude
        self.Vmag = np.zeros(ntargs) # list of V magnitude
        self.Rmag = np.zeros(ntargs) # list of R magnitude
        self.Imag = np.zeros(ntargs) # list of I magnitude
        self.Jmag = np.zeros(ntargs) # list of J magnitude
        self.Hmag = np.zeros(ntargs) # list of H magnitude
        self.Kmag = np.zeros(ntargs) # list of K magnitude
        self.BV = np.zeros(ntargs) # list of B-V Johnson magnitude
        self.MV = np.zeros(ntargs) # list of absolute V magnitude (MV = -5 * log(1000/parx) + 5 - Vmag) 
        self.BC = np.zeros(ntargs) # list of bolometric correction
        self.L = np.zeros(ntargs) # list of stellar luminosity in Solar luminosities
        self.Binary_Cut = np.zeros(ntargs,dtype=bool) # boolean list where True is companion closer than 10 arcsec

    def __str__(self):
        """String representation of the StarCatalog object
        
        When the command 'print' is used on the StarCatalog object, this method
        will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Star Catalog class object attributes'
