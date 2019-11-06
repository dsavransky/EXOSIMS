# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

class StarCatalog(object):
    """Star Catalog class template
     
    This class contains all variables and methods necessary to perform
    Star Catalog Module calculations in exoplanet mission simulation.
    
    Attributes:
        ntargs (integer):
            Number of stars
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
            Stellar luminosity in ln(Solar luminosities)
        Binary_Cut (boolean ndarray):
            Boolean where True is a binary star with companion closer than 10 arcsec
        dist (astropy Quantity array):
            Distance to star in units of pc
        parx (astropy Quantity array):
            Parallax in units of mas
        coords (astropy SkyCoord array):
            SkyCoord object (ICRS frame) containing right ascension, declination, and 
            distance to star in units of deg, deg, and pc
        pmra (astropy Quantity array):
            Proper motion in right ascension in units of mas/year
        pmdec (astropy Quantity array):
            Proper motion in declination in units of mas/year
        rv (astropy Quantity array):
            Radial velocity in units of km/s
        cachedir (str):
            Path to cache directory
        
    """

    _modtype = 'StarCatalog'

    def __init__(self, ntargs=1, cachedir=None, **specs):

        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 

        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # ntargs must be an integer >= 1
        self.ntargs = max(int(ntargs), 1)
        
        # list of astropy attributes
        self.dist = np.ones(ntargs)*u.pc                        # distance
        self.parx = self.dist.to('mas', equivalencies=u.parallax()) # parallax
        self.coords = SkyCoord(ra=np.zeros(ntargs)*u.deg, dec=np.zeros(ntargs)*u.deg,
                distance=self.dist)                             # ICRS coordinates
        self.pmra = np.zeros(ntargs)*u.mas/u.yr                 # proper motion in RA
        self.pmdec = np.zeros(ntargs)*u.mas/u.yr                # proper motion in DEC
        self.rv = np.zeros(ntargs)*u.km/u.s                     # radial velocity
        
        # list of non-astropy attributes
        self.Name = np.array(['Prototype']*ntargs)              # star names
        self.Spec = np.array(['G']*ntargs)                      # spectral types
        self.Umag = np.zeros(ntargs)                            # U magnitude
        self.Bmag = np.zeros(ntargs)                            # B magnitude
        self.Vmag = np.zeros(ntargs)                            # V magnitude
        self.Rmag = np.zeros(ntargs)                            # R magnitude
        self.Imag = np.zeros(ntargs)                            # I magnitude
        self.Jmag = np.zeros(ntargs)                            # J magnitude
        self.Hmag = np.zeros(ntargs)                            # H magnitude
        self.Kmag = np.zeros(ntargs)                            # K magnitude
        self.BV = np.zeros(ntargs)                              # B-V Johnson magnitude
        self.MV = np.zeros(ntargs)                              # absolute V magnitude 
        self.BC = np.zeros(ntargs)                              # bolometric correction
        self.L = np.ones(ntargs)                               # stellar luminosity in ln(SolLum)
        self.Binary_Cut = np.zeros(ntargs, dtype=bool)          # binary closer than 10 arcsec
        
        # populate outspecs
        self._outspec['ntargs'] = self.ntargs

    def __str__(self):
        """String representation of the StarCatalog object
        
        When the command 'print' is used on the StarCatalog object, this method
        will return the values contained in the object
        
        """
        
        for att in self.__dict__:
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Star Catalog class object attributes'
