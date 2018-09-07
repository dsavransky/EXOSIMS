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
        
        
    def partitionSphere(self,N,d):
        Atot = 4*np.pi  # total surface area for 2D sphere
        Ar   = Atot/N   # area of each partition
        
        #spherical caps have area Ar, begin at certain colatitude
        PHIc = np.arccos( 1 - Atot/(2*np.pi*N) )
        
        # COLLAR ANGLES (LATITUDE)
        dI = np.sqrt(Ar)          #ideal collar angles measured as colatitudes
        nI = (np.pi - 2*PHIc)/dI   #ideal # of collars
        
        #actual collar angles
        nF = int(np.max([1,np.round(nI)]))    #actual # of collars
        dF = (np.pi - 2*PHIc)/nF               #'fitting' angle for each collar
        
        collars = np.zeros(nF+1)
        COLAT = np.zeros(nF+2)                #colatitudes of each collar
        COLAT[1] = PHIc + dF/2
        COLAT[-1] = np.pi
        
        #colatitudes of each collar
        for i in range(nF+1): collars[i] = PHIc + i*dF
        
        #colatitudes of each star
        for i in np.arange(2,nF+1): COLAT[i] = COLAT[i-1] + dF
            
        # ZONE PARTITIONS (LONGITUDE)
        Acap = Ar
        Az   = np.zeros(nF+2)
        Az[0]  = Acap
        Az[-1] = Acap 
        for i in range(nF): Az[i+1] = 2*np.pi*( np.cos(collars[i]) - np.cos(collars[i+1]) )
        
        yJ = Az/Ar   #ideal zone partitions
        mJ = np.ones(nF+2)
        aJ = np.zeros(nF+2)
        
        for j in np.arange(1,nF+2):
            mJ[j] = round(yJ[j]+aJ[j-1])
            aJ[j] = aJ[j-1] + (yJ[j] - mJ[j])
        
        # putting it all together
        theta = np.zeros(N)
        phi   = np.zeros(N)
        S = 0
        for i in range(nF+2):
            K   = int(mJ[i])
            LONG = np.zeros(K)
            for k in range(K): LONG[k] = (np.pi/K)*(1+2*k)
            S += K
            for s,k in zip(np.arange(S-K,S),range(K)):
                phi[s]   = np.pi/2 - COLAT[i]
                theta[s] = LONG[k]        
        
        theta_0 = theta[0]
        phi_0   = phi[0]
        
        psi = np.arccos( np.cos(phi_0)*np.cos(phi)*np.cos(theta_0 - theta) + np.sin(phi_0)*np.sin(phi))
        psi = np.sort(psi)
        theta_,phi_ = self.add_caps(np.max(np.abs(np.diff(psi))))
    
        theta = np.append(theta,theta_)
        phi   = np.append(phi,  phi_  )
    
        coords = SkyCoord(theta*u.rad,phi*u.rad,d*np.ones(len(phi))*u.pc)
        return coords
        
    def add_caps(self,psi):
    
        As = 2*np.pi*(1-np.cos(psi)) # area of polar cap specified by psi
        At = 4*np.pi                 # total area
        f  = As/At                   # fraction of stars in cap
        N  = int(round( 1/f )*2)       #number of stars in the cap
        
        Ar = As/N            #area of each individual area
        dI = np.sqrt(Ar)     #ideal collar angle
        
        nI = psi/dI             #ideal number of collars
        n  = int(np.max([1,np.round(nI)]))   #actual number of collars
        dF = (nI/n)*dI          #fitted collar angle
        
        COLAT   = np.zeros(n-1)
        
        #colatitudes of each star
        for i in np.arange(1,n-1): COLAT[i] = COLAT[i-1] + dF
            
        # ZONE PARTITIONS (LONGITUDE)
        Az   = np.zeros(n-1)
        for i in range(n-2): Az[i+1] = 2*np.pi*( np.cos(COLAT[i]) - np.cos(COLAT[i+1]) )
            
        yJ = Az/Ar   #ideal zone partitions
        mJ = np.ones(n-1)
        aJ = np.zeros(n-1)
        
        for j in np.arange(1,n-1):
            mJ[j] = round(yJ[j]+aJ[j-1])
            aJ[j] = aJ[j-1] + (yJ[j] - mJ[j])
    
        # putting it all together
        theta = np.zeros(N)
        phi   = np.zeros(N)
        S = 0
        for i in range(n-1):
            K   = int(mJ[i])
            LONG = np.zeros(K)
            for k in range(K): LONG[k] = (np.pi/K)*(1+2*k)
            S += K
            for s,k in zip(np.arange(S-K,S),range(K)):
                phi[s] = np.pi/2 - COLAT[i]  #dec - phi
                theta[s]   = LONG[k]         #ra  - theta
        
        goodIdx = np.where( (phi>0) & (phi is not np.pi/2) )
        theta   = theta[goodIdx]
        phi     = phi[goodIdx]
        
        theta = np.append(theta,theta)
        phi   = np.append(phi,-phi)

        RAsort = np.argsort(theta)

        return theta[RAsort],phi[RAsort]
