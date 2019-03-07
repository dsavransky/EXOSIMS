# -*- coding: utf-8 -*-
import os, inspect
import warnings
import numpy as np
import astropy
import astropy.units as u
from astropy.constants import R_sun
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.StarCatalog import StarCatalog

class EXOCAT1(StarCatalog):
    """
    EXOCAT Catalog class
    
    This class populates the star catalog used in EXOSIMS from
    Margaret Turnbull's EXOCAT catalog, retrieved from the
    NASA Exoplanet Archive as a VOTABLE.
    # documentation of fields available at https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01
    #https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#kep_names_stellar_timeseries
    #  link above then search mission_exocat
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
            warnings.simplefilter('ignore', 
                    astropy.io.votable.exceptions.VOTableSpecWarning)
            warnings.simplefilter('ignore', 
                    astropy.io.votable.exceptions.VOTableChangeWarning)
            votable = parse(catalogpath)
        table = votable.get_first_table()
        data = table.array
        
        StarCatalog.__init__(self, ntargs=len(data), **specs)
        
        # list of astropy attributes
        self.dist = data['st_dist'].data*u.pc #Distance to the planetary system in units of parsecs
        self.parx = self.dist.to('mas', equivalencies=u.parallax()) # parallactic angle in units of mas
        self.coords = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg,
                distance=self.dist) #Right Ascension of the planetary system in decimal degrees, Declination of the planetary system in decimal degrees
        self.pmra = data['st_pmra'].data*u.mas/u.yr #Angular change in right ascension over time as seen from the center of mass of the Solar System, units (mas/yr)
        self.pmdec = data['st_pmdec'].data*u.mas/u.yr #Angular change in declination over time as seen from the center of mass of the Solar System, units (mas/yr)
        self.L = data['st_lbol'].data #Amount of energy emitted by a star per unit time, measured in units of solar luminosities. The bolometric corrections are derived from V-K or B-V colors, units [log(solar)]
        
        # list of non-astropy attributes
        self.Name = data['hip_name'] #Name of the star as given by the Hipparcos Catalog.
        self.Spec = data['st_spttype'] #Classification of the star based on their spectral characteristics following the Morgan-Keenan system
        self.Vmag = data['st_vmag'] #Brightness of the host star as measured using the V band in units of magnitudes
        self.Jmag = data['st_j2m'] #Stellar J (2MASS) Magnitude Value
        self.Hmag = data['st_h2m'] #Stellar H (2MASS) Magnitude Value
        self.BV = data['st_bmv'] #Color of the star as measured by the difference between B and V bands, units of [mag]
        self.Bmag = self.Vmag + data['st_bmv'] #calculation of the B band
        #st_vmk Color of the star as measured by the difference between V and K bands. The source column is for the K magnitude, units of [mag]
        self.Kmag = self.Vmag - data['st_vmk'] #calculation of the K band
        #st_mbol Apparent magnitude of the star at a distance of 10 parsec units of [mag]
        self.BC = -self.Vmag + data['st_mbol'] # bolometric correction https://www.astro.princeton.edu/~gk/A403/constants.pdf
        self.MV = self.Vmag - 5.*(np.log10(self.dist.to('pc').value) - 1.) # apparent "visual" magnitude https://www.astro.princeton.edu/~gk/A403/constants.pdf
        self.stellar_diameters = data['st_rad']*2.*R_sun # stellar_diameters in solar diameters
        self.Binary_Cut = ~data['wds_sep'].mask #WDS (Washington Double Star) Catalog separation (arcsecs)