import os, inspect
import numpy as np
import astropy
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from EXOSIMS.util.get_dirs import get_downloads_dir


class GaiaCat1(StarCatalog):
    """
    Gaia-derived Catalog class
    
    This class populates the star catalog used in EXOSIMS from
    a dump of Gaia DR2 data.

    Attributes:
        Only StarCatalog prototype attributes are used.
    
    """
    
    def __init__(self, catalogfile='Glt15Dlt200DSNgt4-result.fits', **specs):
        """
        Constructor 

        Args:
            catalogfile (string):
                Name of catalog FITS Table. Defaults to Glt15Dlt200DSNgt4-result.fits
                Assumed to be in downloads directory.
        
        """
       
        downloadsdir = get_downloads_dir()
        catalogpath = os.path.join(downloadsdir, catalogfile)
        
        if not os.path.exists(catalogpath):
            raise IOError('Catalog File %s Not Found.'%catalogpath)
        

        data = fits.open(catalogpath)[1].data

        StarCatalog.__init__(self, ntargs=len(data), **specs)
        
        self.parx = data['parallax']*u.mas
        self.dist = self.parx.to('pc', equivalencies=u.parallax())

        self.coords = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg,
                distance=self.dist)
        
        self.Name = data['source_id']
        self.Teff = data['teff_val']
        self.Gmag = data['phot_g_mean_mag']
        self.BPmag = data['phot_bp_mean_mag']
        self.RPmag = data['phot_rp_mean_mag']
        self.RAerr = data['ra_error']
        self.DECerr = data['dec_error']
        self.parxerr = data['parallax_error']
        self.astrometric_matched_observations = data['astrometric_matched_observations']
        self.visibility_periods_used = data['visibility_periods_used']

        #photometric fit relationships from: 
        #https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T8

        self.Vmag = self.Gmag - (-0.01760 - 0.006860*(self.BPmag - self.RPmag)\
                - 0.1732*(self.BPmag - self.RPmag)**2)
        self.Rmag = self.Gmag - (-0.003226 + 0.3833*(self.BPmag - self.RPmag)\
                - 0.1345*(self.BPmag - self.RPmag)**2)
        self.Imag = self.Gmag - (0.02085 + 0.7419*(self.BPmag - self.RPmag)\
                - 0.09631*(self.BPmag - self.RPmag)**2)


#        GV = self.Gmag - self.Vmag
#        self.BV = np.full(GV.size,np.nan)
#        for j,gv in enumerate(GV):
#            if ~np.isnan(gv):
#                tmp = np.roots([-0.001768,-0.2297,-0.02385,-0.02907 - gv])
#                self.BV[j] = np.max(tmp)

