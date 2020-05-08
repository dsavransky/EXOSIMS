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
import pkg_resources

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
v = Vizier(columns=['Plx'],catalog="I/311/hip2")
#Simbad.reset_votable_fields()
Simbad.add_votable_fields('typed_id', #queries value (i.e. HP)
                              'flux(V)', #m_V
                              'flux(B)', #m_B
                              'flux(R)', #m_R
                              'flux(I)', #m_I
                              'flux(H)', #m_K
                              'flux(J)', #m_K
                              'flux(K)', #m_K
                              'distance', #parsecs
                              'flux_bibcode(V)',#flux citation
                              'flux_error(V)', #v-band uncertainty
                              'sp'#spectral type
                              )



class HIP(StarCatalog):
    """
    Catalog generator class
    
    """
    
    def __init__(self, HIP, **specs):
        """
        
        Args:
            HIP (list or string):
                List of Hipparcos identifiers (HIP numbers)
        """
        if isinstance(HIP,str):
            if HIP[-3:]=='csv':
                HIP=np.loadtxt("hip.csv",delimiter=",",dtype="str")
            else:
                raise ValueError("Expected CSV file containing HIP values")

            
        #catalogpath = pkg_resources.resource_filename('EXOSIMS.StarCatalog',catalogfile)
        #
        #if not os.path.exists(catalogpath):
        #    raise IOError('Catalog File %s Not Found.'%catalogpath)
        
        if HIP[0][:3] != "HIP":
            raise ValueError("First value in list not an HIP Identifier")

        
        StarCatalog.__init__(self, ntargs=len(HIP), **specs)
        HIP_names=[HIP[i] for i in range(len(HIP))]
        simbad_list= Simbad.query_objects(HIP_names)


        #fill in distances
        for i, targ in enumerate(simbad_list["Distance_distance"]):
            if targ>0:
                continue
            else:
            #print(simbad_list["TYPED_ID"][i].decode('ascii'))
                result = v.query_object(simbad_list["TYPED_ID"][i].decode('ascii'))['I/311/hip2']
                d=1000/result['Plx']
                print(d)
                simbad_list["Distance_distance"][i]=d.data.data[0]
                simbad_list["Distance_method"][i]="hip2"
        data=simbad_list
        self.dist = simbad_list["Distance_distance"].data.data*u.pc #Distance to the planetary system in units of parsecs
        print(simbad_list['RA'].data.data)
        self.coords = SkyCoord(ra=simbad_list['RA'].data.data,
                                   dec=simbad_list['DEC'].data.data,
                                   #distance=self.dist,
                                    unit=(u.hourangle, u.deg,u.arcsec)) #Right Ascension of the planetary system in decimal degrees, Declination of the planetary system in decimal degrees
        #self.pmra = data['st_pmra'].data*u.mas/u.yr #Angular change in right ascension over time as seen from the center of mass of the Solar System, units (mas/yr)
        #self.pmdec = data['st_pmdec'].data*u.mas/u.yr #Angular change in declination over time as seen from the center of mass of the Solar System, units (mas/yr)
        #self.L = data['st_lbol'].data #Amount of energy emitted by a star per unit time, measured in units of solar luminosities. The bolometric corrections are derived from V-K or B-V colors, units [log(solar)]
        
        # list of non-astropy attributes
        self.Name = HIP_names #Name of the star as given by the Hipparcos Catalog.
        self.Spec = data['SP_TYPE'].astype(str) #Classification of the star based on their spectral characteristics following the Morgan-Keenan system
        self.Vmag = data['FLUX_V'].data.data # V mag
        self.Jmag = data['FLUX_J'] #Stellar J Magnitude Value
        self.Hmag = data['FLUX_H'] #Stellar H  Magnitude Value
        self.Hmag = data['FLUX_I'] #Stellar I Magnitude Value
        self.Bmag = data['FLUX_B'].data.data
        self.Kmag = data['FLUX_K'].data.data
        self.BV = data['FLUX_B'].data.data - data['FLUX_V'].data.data #Color of the star as measured by the difference between B and V bands, units of [mag]
        self.MV = self.Vmag - 5.*(np.log10(self.dist.to('pc').value) - 1.) # absolute V mag
        #self.Teff =  data['st_teff']
        #st_mbol Apparent magnitude of the star at a distance of 10 parsec units of [mag]
        #self.BC = -self.Vmag + data['st_mbol'] # bolometric correction 
        #self.stellar_diameters = data['st_rad']*2.*R_sun # stellar_diameters in solar diameters
        #self.Binary_Cut = ~data['wds_sep'].mask #WDS (Washington Double Star) Catalog separation (arcsecs)
        #save original data
        self.data = data
