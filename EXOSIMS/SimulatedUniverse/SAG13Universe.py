from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time

class SAG13Universe(SimulatedUniverse):
    """
    Simulated Universe module based on SAG13 occurrence rates.
    
    This is the current working model based on averaging multiple studies. 
    These do not yet represent official scientific values.
    """

    def __init__(self, SAG13coeffs=[[.38, -.19, .26, 0],[.73, -1.18, .59, 3.4]],
            Rprange=[2/3., 17.0859375], arange=[0.06346941, 3.14678393], **specs):
        
        # load SAG13 coefficients
        self.SAG13coeffs = np.array(SAG13coeffs, dtype=float)
        assert self.SAG13coeffs.ndim <= 2, "SAG13coeffs array dimension must be <= 2."
        
        if self.SAG13coeffs.ndim == 1:
            self.SAG13coeffs = np.array(np.append(self.SAG13coeffs[:3], 0.), ndmin=2)
        if len(self.SAG13coeffs) != 4:
            self.SAG13coeffs = self.SAG13coeffs.T
        assert len(self.SAG13coeffs) == 4, "SAG13coeffs array length must be 4."
        
        # default sma range = [0.06346941, 3.14678393] AU
        # corresponds to period range = [10, 640] day
        # and EXOCAT stelar mass range = [0.34101181, 10.14651506] solMass
        specs['Rprange'] = Rprange
        specs['arange'] = arange
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generating universe based on planet radius and period sampling.
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        
        # SAG13 coeffs
        Gamma = self.SAG13coeffs[0,:]
        alpha = self.SAG13coeffs[1,:]
        beta = self.SAG13coeffs[2,:]
        Rplim = np.append(self.SAG13coeffs[3,:], np.inf)
        
        # generate period range
        Msrange = [min(TL.MsTrue).value, max(TL.MsTrue).value]*TL.MsTrue.unit
        PPop.Trange = 2*np.pi*np.sqrt(PPop.arange**3/(const.G*Msrange)).to('year')
        
        # check if generated period range is compatible with arange and Msrange
        assert PPop.Trange[1] > PPop.Trange[0], \
                "SAG13 period range not compatible with sma range and/or Ms range."
        
        # SAG13 planet radius ad period sampling
        nx = 6
        x = np.log2(PPop.Trange.to('year').value)
        T = 2**np.linspace(x[0], x[1], nx+1)
        ny = 8
        y = np.log(PPop.Rprange.to('earthRad').value)/np.log(1.5)
        Rp = 1.5**np.linspace(y[0], y[1], ny+1)
        
        # loop over all (T, Rp) regions
        plan2star = []
        period = []
        radius = []
        self.eta = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                # discretize each region for integration
                npts = 1e3
                dx = np.linspace(np.log(T[i]), np.log(T[i+1]), npts)
                dy = np.linspace(np.log(Rp[j]), np.log(Rp[j+1]), npts)
                dT, dRp = np.meshgrid(np.exp(dx), np.exp(dy))
                dN = np.zeros(dRp.shape)
                for k in range(len(Rplim)-1):
                    mask = (dRp >= Rplim[k]) & (dRp < Rplim[k+1])
                    dN[mask] = Gamma[k] * dRp[mask]**alpha[k] * dT[mask]**beta[k]
                dxstep = dx[1] - dx[0]
                dystep = dy[1] - dy[0]
                self.eta[i,j] = np.trapz(np.trapz(dN*dxstep*dystep))
                
                # treat eta as the rate paramter of a Poisson distribution
                targetSystems = np.random.poisson(lam=self.eta[i,j],size=TL.nStars)
                for m,n in enumerate(targetSystems):
                    plan2star = np.hstack((plan2star,[m]*n))
                    period = np.hstack((period,np.exp(np.random.uniform(low=np.log(T[i]),
                            high=np.log(T[i+1]),size=n)).tolist()))
                    radius = np.hstack((radius,np.exp(np.random.uniform(low=np.log(Rp[j]),
                            high=np.log(Rp[j+1]),size=n)).tolist()))
        
        # must generate at least one planet
        if plan2star.size == 0:
            plan2star = np.array([0])
            period = np.array([1.])
            radius = np.array([1.])
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        # sample all of the orbital and physical parameters
        self.T = period*u.year                              # period
        self.Rp = radius*u.earthRad                         # radius
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)      # mass from radius
        mu = const.G*TL.MsTrue[self.plan2star]
        self.a = ((mu*(self.T/(2*np.pi))**2)**(1/3.)).to('AU')# semi-major axis
        self.e = PPop.gen_eccen_from_sma(self.nPlans,self.a) if PPop.constrainOrbits \
                else PPop.gen_eccen(self.nPlans)            # eccentricity
        self.I = PPop.gen_I(self.nPlans)                    # inclination
        self.O = PPop.gen_O(self.nPlans)                    # longitude of ascending node
        self.w = PPop.gen_w(self.nPlans)                    # argument of periapsis
        self.M0 = np.random.uniform(360,size=self.nPlans)*u.deg # initial mean anomaly
        self.p = PPop.gen_albedo(self.nPlans)               # albedo
