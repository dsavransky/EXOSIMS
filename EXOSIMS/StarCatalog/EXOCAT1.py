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
import re
import astropy.io.ascii

class EXOCAT1(StarCatalog):
    """
    EXOCAT Catalog class
    
    This class populates the star catalog used in EXOSIMS from
    Margaret Turnbull's EXOCAT catalog, retrieved from the
    NASA Exoplanet Archive as a VOTABLE.
    Documentation of fields available at https://exoplanetarchive.ipac.caltech.edu/docs/API_mission_stars.html
    Attributes:
        Only StarCatalog prototype attributes are used.
    
    """
    
    def __init__(self, catalogpath=None, wdsfilepath=None, **specs):
        """
        Constructor for EXOCAT1
        
        Args:
            catalogpath (string):
                Full path to catalog VOTABLE. If None (default) uses default catalogfile in
                EXOSIMS.StarCatalog directory.
            catalogfile (string):
                Catalog filename in EXOSIMS.StarCatalog directory to use. Ignored if catalogpath 
                is not None. Defaults to mission_exocat.votable
        
        """
       
        if catalogpath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = 'mission_exocat_2019.08.22_11.37.24.votable'
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
        self.Name = data['hip_name'].astype(str) #Name of the star as given by the Hipparcos Catalog.
        self.Spec = data['st_spttype'].astype(str) #Classification of the star based on their spectral characteristics following the Morgan-Keenan system
        self.Vmag = data['st_vmag'] # V mag
        self.Jmag = data['st_j2m'] #Stellar J (2MASS) Magnitude Value
        self.Hmag = data['st_h2m'] #Stellar H (2MASS) Magnitude Value
        self.BV = data['st_bmv'] #Color of the star as measured by the difference between B and V bands, units of [mag]
        self.Bmag = self.Vmag + data['st_bmv'] #B mag based on BV color
        self.Kmag = self.Vmag - data['st_vmk'] #K mag based on VK color
        #st_mbol Apparent magnitude of the star at a distance of 10 parsec units of [mag]
        self.BC = -self.Vmag + data['st_mbol'] # bolometric correction 
        self.MV = self.Vmag - 5.*(np.log10(self.dist.to('pc').value) - 1.) # absolute V mag
        self.stellar_diameters = data['st_rad']*2.*R_sun # stellar_diameters in solar diameters
        self.Binary_Cut = ~data['wds_sep'].mask #WDS (Washington Double Star) Catalog separation (arcsecs)

        #if given a WDS update file, ingest along with catalog
        if wdsfilepath is not None:
            assert os.path.exists(wdsfilepath), "WDS data file not found at %s"%wdsfilepath
            
            wdsdat = astropy.io.ascii.read(wdsfilepath)

            #get HIP numbers of catalog
            HIPnums = np.zeros(len(data),dtype=int) 
            for j,name in enumerate(data['hip_name'].astype(str)):
                tmp = re.match("HIP\s*(\d+)", name)
                if tmp:
                    HIPnums[j] = int(tmp.groups()[0])

            #find indices of wdsdata in catalog
            inds = np.zeros(len(wdsdat),dtype=int)
            for j,h in enumerate(wdsdat['HIP'].data):
                inds[j] = np.where(HIPnums == h)[0]

            #add attributes to catalog
            wdsatts = [ 'Close_Sep(")', 'Close(M2-M1)', 'Bright_Sep(")', 'Bright(M2-M1)' ]
            catatts = [ 'closesep', 'closedm', 'brightsep', 'brightdm']

            for wdsatt,catatt in zip(wdsatts,catatts):
                tmp = wdsdat[wdsatt].data
                tmp[tmp == '___'] = 'nan'
                tmp = tmp.astype(float)
                tmp2 = np.full(len(data),np.nan)
                tmp2[inds] = tmp
                setattr(self,catatt,tmp2)

        self.data = data
