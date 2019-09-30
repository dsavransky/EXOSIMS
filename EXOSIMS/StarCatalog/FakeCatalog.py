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
        
        # getting sizes of the two angular sep distributions
        nP = int(np.floor(N/2.))     # half of stars in positive branch
        nN = nP + 1 if N % 2 else nP # checks to see if N is odd
        
        # creating output of logistic function (positive and negative branch)
        tP = np.linspace(0.5,0.99,nP)
        tN = np.linspace(0.5,0.01,nN)[1:] # not using the same reference star twice
        
        # getting inverse of logistic function as distribution of separations
        fP = np.log( tP / (1 - tP) )
        fP = fP/np.abs(fP[-1])
        
        fN = np.log( tN / (1 - tN) )
        fN = fN/np.abs(fN[-1])
        
        # getting angular distributions of stars for two branches
        raP,decP,distsP = self.get_angularDistributions(fP,d,pos=True)
        raN,decN,distsN = self.get_angularDistributions(fN,d,pos=False)
        
        # putting it all together
        ra    = np.hstack([ raP , raN ]) * u.rad
        dec   = np.hstack([ decP , decN ]) * u.rad
        dists = np.hstack([ distsP , distsN ]) *u.pc
        
        # reference star should be first on the list
        coords = SkyCoord(ra,dec,dists)

        return coords


    def get_angularDistributions(self,f,d,pos=True):
        
        n = int( len(f) )
        
        # angular separations from reference star
        psi    = np.pi * f
        cosPsi = np.cos(psi)
        
        # calculating phi angle (i.e. DEC)
        sinPhi = ( np.abs(cosPsi) + ( 1-np.abs(cosPsi))*np.random.rand(n) )
        phi    = np.arcsin( sinPhi ) # only returns angles from 0 to pi/2
        
        # calculating phi angle (i.e. RA)
        cosTheta    = cosPsi/sinPhi
        theta       = np.arccos(cosTheta)
        
        # moving stars to southern hemisphere
        if pos:
            phi = np.pi - phi 
            theta = 2*np.pi - theta
        
        # final transforms
        dec   = (np.pi/2. - phi)
        dists = d*np.ones(n)
        
        return theta, dec, dists