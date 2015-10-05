from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import os,inspect
from astropy.io.votable import parse
import numpy as np
from astropy.coordinates import SkyCoord

class EXOCAT1(StarCatalog):
    """
    EXOCAT Catalog class
    
    This class populates the star catalog used in EXOSIMS from
    Margaret Trunbull's EXOCAT catalog, retrieved from the
    NASA Exoplanet Archive as a VOTABLE.

    Attributes:
        stardata (ndarray):
            All VOTABLE data

    All other attributes as in StarCatalog

    """
    
    def __init__(self,catalogpath=None,**specs):
        """
        Constructor for EXOCAT1

        Args:
            catalogpath (str):
                Full path to catalog VOTABLE.  Defaults to mission_exocat.votable

        Notes:
            All votable data is loaded to self.stardata

        """
       
        if catalogpath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = 'mission_exocat.votable'
            catalogpath = os.path.join(classpath, filename)
        
        if not os.path.exists(catalogpath):
                raise IOError('Catalog File %s Not Found.'%catalogpath)

        StarCatalog.__init__(self)

        #read votable
        votable = parse(catalogpath)
        table = votable.get_first_table()
        data = table.array

        keyword_table = {'Name':'hip_name',
                        'Spec':'st_spttype',
                        'Vmag':'st_vmag', 
                        'Jmag':'st_j2m', 
                        'Hmag':'st_h2m',
                        'dist':'st_dist', 
                        'BV':'st_bmv',
                        'L':'st_lbol',
                        'pmra':'st_pmra',
                        'pmdec':'st_pmdec'}

                
        for key in keyword_table.keys():
            setattr(self,key, data[keyword_table[key]])

        self.Bmag = data['st_bmv'] + self.Vmag
        self.Kmag = self.Vmag - data['st_vmk']
        self.BC = data['st_mbol'] - self.Vmag
        self.Binary_Cut = ~data['wds_sep'].mask
        self.MV = self.Vmag  - 5*(np.log10(self.dist) - 1)
        self.coords = SkyCoord(ra=data['ra'], dec=data['dec'], unit='deg')

        self.stardata = data
        

