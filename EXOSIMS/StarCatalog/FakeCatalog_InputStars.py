from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

class FakeCatalog_InputStars(StarCatalog):
    
    def __init__(self, star_dist=1, ra_input = np.array([]), dec_input = np.array([]), Vmag = 5, Bmag = 5,**specs):
        """
        lon_sep and lat_sep should be in deg
        """
        
        StarCatalog.__init__(self,**specs)
        
        # stars we wish to add
        ra_Array = ra_input*u.deg
        dec_Array = dec_input*u.deg
        
        # ntargs must be an integer >= 1
        self.ntargs = max(int(ra_Array.size), 1)
        
        # star distances in pc
        dists = star_dist*np.ones(self.ntargs) *u.pc
        
        # reference star should be first on the list
        coords = SkyCoord(ra_Array,dec_Array,dists,frame='icrs')
        
        # list of astropy attributes
        self.coords = coords     # barycentric true ecliptic coordinates
        self.dist = star_dist*np.ones(self.ntargs)*u.pc              # distance
        self.parx = self.dist.to('mas', equivalencies=u.parallax())  # parallax
        self.pmra = np.zeros(self.ntargs)*u.mas/u.yr                 # proper motion in RA
        self.pmdec = np.zeros(self.ntargs)*u.mas/u.yr                # proper motion in DEC
        self.rv = np.zeros(self.ntargs)*u.km/u.s                     # radial velocity
        
        # list of non-astropy attributes to pass target list filters
        self.Name = np.array([str(x) for x in range(self.ntargs)])   # star names
        self.Spec = np.array(['G']*self.ntargs)                      # spectral types
        self.Umag = np.zeros(self.ntargs)                            # U magnitude
        self.Bmag = Bmag*np.ones(self.ntargs)                            # B magnitude
        self.Vmag = Vmag*np.ones(self.ntargs)                           # V magnitude
        self.Rmag = np.zeros(self.ntargs)                            # R magnitude
        self.Imag = np.zeros(self.ntargs)                            # I magnitude
        self.Jmag = np.zeros(self.ntargs)                            # J magnitude
        self.Hmag = np.zeros(self.ntargs)                            # H magnitude
        self.Kmag = np.zeros(self.ntargs)                            # K magnitude
        self.BV = np.zeros(self.ntargs)                              # B-V Johnson magnitude
        self.MV   = self.Vmag - 5*( np.log10(star_dist) - 1 )   # absolute V magnitude 
        self.BC   = -0.10*np.ones(self.ntargs)                       # bolometric correction
        
        BM        = self.MV + self.BC
        L0        = 3.0128e28
        BMsun     = 4.74
        self.L    = L0*10**(0.4*(BMsun-BM))                     # stellar luminosity in ln(SolLum)
        self.Binary_Cut = np.zeros(self.ntargs, dtype=bool)          # binary closer than 10 arcsec
        
        # populate outspecs
        self._outspec['ntargs'] = self.ntargs