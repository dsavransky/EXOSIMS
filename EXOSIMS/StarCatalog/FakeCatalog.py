from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

class FakeCatalog(StarCatalog):
    
    def __init__(self, ntargs=1000, star_dist=5, **specs):
        
        StarCatalog.__init__(self,**specs)
        
        # ntargs must be an integer >= 1
        self.ntargs = max(int(ntargs), 1)
        
        # list of astropy attributes
        self.coords = self.inverse_method(self.ntargs,star_dist)     # ICRS coordinates
        self.ntargs = int(len(self.coords.ra))
        self.dist = star_dist*np.ones(self.ntargs)*u.pc               # distance
        self.parx = self.dist.to('mas', equivalencies=u.parallax())   # parallax
        self.pmra = np.zeros(self.ntargs)*u.mas/u.yr                 # proper motion in RA
        self.pmdec = np.zeros(self.ntargs)*u.mas/u.yr                # proper motion in DEC
        self.rv = np.zeros(self.ntargs)*u.km/u.s                     # radial velocity
        
        # list of non-astropy attributes to pass target list filters
        self.Name = np.array([str(x) for x in range(self.ntargs)])   # star names
        self.Spec = np.array(['G']*self.ntargs)                      # spectral types
        self.Umag = np.zeros(self.ntargs)                            # U magnitude
        self.Bmag = np.zeros(self.ntargs)                            # B magnitude
        self.Vmag = 5*np.ones(self.ntargs)                           # V magnitude
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
        
    def inverse_method(self,N,d):
        
        t = np.linspace(1e-3,0.999,N)
        f = np.log( t / (1 - t) )
        f = f/f[0]
        
        psi= np.pi*f
        cosPsi = np.cos(psi)
        sinTheta = ( np.abs(cosPsi) + (1-np.abs(cosPsi))*np.random.rand(len(cosPsi)))
        
        theta = np.arcsin(sinTheta)
        theta = np.pi-theta + (2*theta - np.pi)*np.round(np.random.rand(len(t)))
        cosPhi = cosPsi/sinTheta
        phi = np.arccos(cosPhi)*(-1)**np.round(np.random.rand(len(t)))
        
        coords = SkyCoord(phi*u.rad,(np.pi/2-theta)*u.rad,d*np.ones(len(phi))*u.pc)

        return coords
