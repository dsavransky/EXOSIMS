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

        tmp = self.PlanetPopulation.allplanetdata[:]
        #throw away things outside of WA range:
        inds = (self.PlanetPopulation.sma*(1+self.PlanetPopulation.eccentricity) > \
                np.tan(self.OpticalSystem.IWA)*tmp['st_dist']*u.pc) & \
               (self.PlanetPopulation.sma*(1-self.PlanetPopulation.eccentricity) < \
                np.tan(self.OpticalSystem.OWA)*tmp['st_dist']*u.pc)
        tmp = tmp[inds]

        tmp = tmp[np.unique(tmp['pl_hostname'].data,return_index=True)[1]]

        for key in self.atts_mapping.keys():
            setattr(self, key, tmp[self.atts_mapping[key]].data)
        self.dist = self.dist*u.pc
   
        self.BC =  -2.5*self.L - 26.832 - self.Vmag
        self.L = 10.**self.L
        self.MV = self.Vmag  - 5*(np.log10(self.dist.value) - 1)

        self.coords = SkyCoord(ra=tmp['ra'].data, dec=tmp['dec'].data, unit='deg')

        self.nStars = len(tmp)
        assert self.nStars, "Target list is empty: nStars = %r"%self.nStars
        self.Binary_Cut = np.zeros(self.nStars,dtype=bool)

        # populate completness values
        self.comp0 = self.Completeness.target_completeness(self)
        # include completeness now that it is set
        self.catalog_atts.append('comp0')
        # populate maximum integration time
        self.maxintTime = self.OpticalSystem.calc_maxintTime(self)
        # include integration time now that it is set
        self.catalog_atts.append('maxintTime')
        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()


    def filter_target_list(self, **specs):
        """ Filtering is done as part of populating the table, so this helper function
        is just a dummy."""

        pass
