# -*- coding: utf-8 -*-
import os, inspect
import warnings
import numpy as np
import astropy
import astropy.units as u
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.StarCatalog import StarCatalog

class EXOCAT1(StarCatalog):
    """
    EXOCAT Catalog class
    
    This class populates the star catalog used in EXOSIMS from
    Margaret Turnbull's EXOCAT catalog, retrieved from the
    NASA Exoplanet Archive as a VOTABLE.
    
    Attributes:
        Only StarCatalog prototype attributes are used.
    
    """
    
    def __init__(self, catalogpath=None, **specs):
        """
        Constructor for EXOCAT1
        
        Args:
            catalogpath (string):
                Full path to catalog VOTABLE. Defaults to mission_exocat.votable
        
        """
       
        if catalogpath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = 'mission_exocat.votable'
            catalogpath = os.path.join(classpath, filename)
        
        if not os.path.exists(catalogpath):
            raise IOError('Catalog File %s Not Found.'%catalogpath)
        
        #read votable
        with warnings.catch_warnings():
            # warnings for IPAC votables are out of control 
            #   they are not moderated by pedantic=False
            #   they all have to do with units, which we handle independently anyway
            warnings.simplefilter('ignore', astropy.io.votable.exceptions.VOTableSpecWarning)
            warnings.simplefilter('ignore', astropy.io.votable.exceptions.VOTableChangeWarning)
            votable = parse(catalogpath)
        table = votable.get_first_table()
        data = table.array
        
        StarCatalog.__init__(self, ntargs=len(data), **specs)
        
        # list of astropy attributes
        self.dist = data['st_dist'].data*u.pc
        self.parx = self.dist.to('mas',equivalencies=u.parallax())
        self.coords = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg, distance=self.dist)
        self.pmra = data['st_pmra'].data*u.mas/u.yr
        self.pmdec = data['st_pmdec'].data*u.mas/u.yr
        
        # list of non-astropy attributes
        self.Name = data['hip_name']
        self.Spec = data['st_spttype']
        self.Vmag = data['st_vmag']
        self.Jmag = data['st_j2m']
        self.Hmag = data['st_h2m']
        self.BV = data['st_bmv']
        self.L = data['st_lbol']
        self.Bmag = self.Vmag + data['st_bmv']
        self.Kmag = self.Vmag - data['st_vmk']
        self.BC = -self.Vmag + data['st_mbol']
        self.MV = self.Vmag - 5*(np.log10(self.dist.value) - 1)
        self.Binary_Cut = ~data['wds_sep'].mask
