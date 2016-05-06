from EXOSIMS.Prototypes.TargetList import TargetList
import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.coordinates import SkyCoord


class KnownRVPlanetsTargetList(TargetList):
    """
    Target list based on population of known RV planets from IPAC.  
    Intended for use with the KnownRVPlanets family of modules.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
                
    
    Notes:  
    
    """

    def __init__(self, **specs):
        
        #define mapping between attributes we need and the IPAC data
        #table loaded in the Planet Population module
        self.atts_mapping = {'Name':'pl_hostname',
                             'Type':'st_spstr', 
                             'Spec':'st_spstr',
                             'parx':'st_plx',
                             'Umag':'st_uj',
                             'Bmag':'st_bj',
                             'Vmag':'st_vj',
                             'Rmag':'st_rc',
                             'Imag':'st_ic',
                             'Jmag':'st_j',
                             'Hmag':'st_h',
                             'Kmag':'st_k',
                             'dist':'st_dist',
                             'BV':'st_bmvj',
                             'L':'st_lum', #log(Lsun)
                             'pmra':'st_pmra', #mas/year
                             'pmdec':'st_pmdec', #mas/year
                             'rv': 'st_radv'}
        
        TargetList.__init__(self, **specs)

    def populate_target_list(self, **specs):
        
        PPop = self.PlanetPopulation
        OS = self.OpticalSystem
        
        tmp = PPop.allplanetdata[:]
        # filter out targets with planets outside of WA range 
        dist = tmp['st_dist'].filled()*u.pc
        mask = ~tmp['st_dist'].mask & \
                (np.arctan(PPop.sma*(1+PPop.eccen)/dist) > OS.IWA) & \
                (np.arctan(PPop.sma*(1-PPop.eccen)/dist) < OS.OWA)
        tmp = tmp[mask]
        # filter out redundant targets
        tmp = tmp[np.unique(tmp['pl_hostname'].data,return_index=True)[1]]
        # filter missing Vmag and BV , for OS.calc_maxintTime
        tmp = tmp[~tmp['st_vj'].mask]
        tmp = tmp[~tmp['st_bmvj'].mask]
        
        self.nStars = len(tmp)
        assert self.nStars, "Target list is empty: nStars = %r"%self.nStars
        
        for att in self.atts_mapping.keys():
            setattr(self, att, tmp[self.atts_mapping[att]].data)
        self.dist = self.dist*u.pc
        
        self.BC =  -2.5*self.L - 26.832 - self.Vmag
        self.L = 10.**self.L
        self.MV = self.Vmag  - 5*(np.log10(self.dist.value) - 1)
        self.coords = SkyCoord(ra=tmp['ra'].data, dec=tmp['dec'].data, unit='deg')
        self.Binary_Cut = np.zeros(self.nStars,dtype=bool)
        
        # populate completness values
        self.comp0 = self.Completeness.target_completeness(self)
        # populate maximum integration time
        self.maxintTime = OS.calc_maxintTime(self)
        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()
        
        # include new attributes to the target list catalog attributes
        self.catalog_atts.append('comp0')
        self.catalog_atts.append('maxintTime')

    def filter_target_list(self, **specs):
        """ Filtering is done as part of populating the table, so this helper function
        is just a dummy."""
        
        pass
