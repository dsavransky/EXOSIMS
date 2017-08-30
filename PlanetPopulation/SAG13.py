from EXOSIMS.PlanetPopulation.KeplerLike2 import KeplerLike2
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate

class SAG13(KeplerLike2):
    """Planet Population module based on SAG13 occurrence rates.
    
    This is the current working model based on averaging multiple studies. 
    These do not yet represent official scientific values.
    
    Attributes: 
        SAG13coeffs (float 4x2 ndarray):
            Coefficients used by the SAG13 broken power law. The 4 lines
            correspond to Gamma, alpha, beta, and the minimum radius.
        SAG13starMass (astropy Quantity):
            Assumed stellar mass corresponding to the given set of coefficients.
        eta2D (float ndarray):
            2D array of planet occurrence rate per star, binned by planetary
            radius and period.
        lnRp (float ndarray):
            Logarithm of the eta2D grid radius values in units of Earth radius. 
        lnT (float ndarray):
            Logarithm of the eta2D grid period values in units of year. 
        Trange (astropy Quantity 1x2 array):
            Period range in units of year.
        Tknee (float):
            Location (in year) of period decay point (knee).
            Not an astropy quantity.
    
    """

    def __init__(self, SAG13coeffs=[[.38, -.19, .26, 0.],[.73, -1.18, .59, 3.4]],
            SAG13starMass=1., Rprange=[2/3., 17.0859375],
            arange=[0.09084645, 1.45354324], **specs):
        
        # first initialize with KerplerLike constructor
        specs['Rprange'] = Rprange
        specs['arange'] = arange
        KeplerLike2.__init__(self, **specs)
        
        # load SAG13 star mass in solMass: 1.3 (F), 1 (G), 0.70 (K), 0.35 (M)
        self.SAG13starMass = float(SAG13starMass)*u.solMass
        
        # generate period range from sma range, for given SAG13starMass
        # default sma range [0.09-1.45 AU] corresponds to period range [10-640 day] @ 1solMass
        mu = const.G*self.SAG13starMass
        self.Trange = 2.*np.pi*np.sqrt(self.arange**3/mu).to('year')
        
        # period value corresponding to KeplerLike smaknee (in year, not a quantity)
        self.Tknee = 2.*np.pi*np.sqrt((self.smaknee*u.AU)**3/mu).to('year').value
        
        # load SAG13 coefficients (Gamma, alpha, beta, Rplim)
        self.SAG13coeffs = np.array(SAG13coeffs, dtype=float)
        assert self.SAG13coeffs.ndim <= 2, "SAG13coeffs array dimension must be <= 2."
        # if only one row of coefficients, make sure the forth element
        # (minimum radius) is set to zero
        if self.SAG13coeffs.ndim == 1:
            self.SAG13coeffs = np.array(np.append(self.SAG13coeffs[:3], 0.), ndmin=2)
        # make sure the array is of shape (4, n) where the forth row
        # contains the minimum radius values (broken power law)
        if len(self.SAG13coeffs) != 4:
            self.SAG13coeffs = self.SAG13coeffs.T
        assert len(self.SAG13coeffs) == 4, "SAG13coeffs array must have 4 rows."
        # sort by minimum radius
        self.SAG13coeffs = self.SAG13coeffs[:,np.argsort(self.SAG13coeffs[3,:])]
        
        # create grid of radii and periods
        # SAG13 sampling uses a log base 1.5 for radius, and a log base 2 for period
        Rplogbase = 1.5
        Tlogbase = 2.
        xlim = np.log(self.Rprange.to('earthRad').value)/np.log(Rplogbase)
        ylim = np.log(self.Trange.to('year').value)/np.log(Tlogbase)
        nx = np.maximum(int(round(np.diff(xlim))), 1)
        ny = np.maximum(int(round(np.diff(ylim))), 1)
        Rp = np.logspace(xlim[0], xlim[1], num=nx+1, base=Rplogbase)
        T = np.logspace(ylim[0], ylim[1], num=ny+1, base=Tlogbase)
        self.lnRp = np.log(Rp)
        self.lnT = np.log(T)
        
        # loop over all log values of radii and periods, and generate the eta 2D array
        self.eta2D = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                ranges = [self.lnRp[i:i+2], self.lnT[j:j+2]]
                self.eta2D[i,j] = integrate.nquad(self.dist_lnradius_lnperiod, ranges)[0]
        
        # eta is the sum of the eta 2D array
        self.eta = np.sum(self.eta2D)
        self._outspec['eta'] = self.eta
        
        # populate _outspec with SAG13 specific attributes
        self._outspec['SAG13starMass'] = self.SAG13starMass.to('solMass').value
        self._outspec['SAG13coeffs'] = self.SAG13coeffs
        self._outspec['eta2D'] = self.eta2D
        self._outspec['lnRp'] = self.lnRp
        self._outspec['lnT'] = self.lnT
        self._outspec['Trange'] = self.Trange.to('year').value
        self._outspec['Tknee'] = self.Tknee

    def gen_radius_sma(self, n):
        """Generate radius values in earth radius and semi-major axis values in AU.
        
        This method is called by gen_radius and gen_sma.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
            a (astropy Quantity array):
                Semi-major axis values in units of AU
        
        """
        # get number of samples per bin
        nsamp = np.ceil(n*self.eta2D/self.eta).astype(int)
        
        # generate random radii and period in each bin
        radius = []
        period = []
        for i in range(len(self.lnRp)-1):
            for j in range(len(self.lnT)-1):
                radius = np.hstack((radius,np.exp(np.random.uniform(low=self.lnRp[i],
                        high=self.lnRp[i+1], size=nsamp[i,j])).tolist()))
                period = np.hstack((period,np.exp(np.random.uniform(low=self.lnT[j],
                        high=self.lnT[j+1], size=nsamp[i,j])).tolist()))
        
        # select exactly n radom planets
        ind = np.random.choice(len(radius), size=n, replace=len(radius)<n)
        Rp = radius[ind]*u.earthRad
        T = period[ind]*u.year
        
        # convert periods to sma values
        mu = const.G*self.SAG13starMass
        a = ((mu*(T/(2*np.pi))**2)**(1/3.)).to('AU')
        
        return Rp, a
    
    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)
        
        Semi-major axis and planetary radius are jointly distributed. 
        Eccentricity is a Rayleigh distribution. Albedo is dependent on the 
        PlanetPhysicalModel but is calculated such that it is independent of 
        other parameters.
        
        Args:
            n (integer):
                Number of samples to generate
        
        Returns:
            a (astropy Quantity array):
                Semi-major axis in units of AU
            e (float ndarray):
                Eccentricity
            p (float ndarray):
                Geometric albedo
            Rp (astropy Quantity array):
                Planetary radius in units of earthRad
        
        """
        n = self.gen_input_check(n)
        # generate semi-major axis and planetary radius samples
        Rp, a = self.gen_radius_sma(n)
        
        # check for constrainOrbits == True for eccentricity samples
        # constants
        C1 = np.exp(-self.erange[0]**2/(2.*self.esigma**2))
        ar = self.arange.to('AU').value
        if self.constrainOrbits:
            # clip sma values to sma range
            sma = np.clip(a.to('AU').value, ar[0], ar[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1. - ar[0]/sma[sma <= amean]
            elim[sma > amean] = ar[1]/sma[sma>amean] - 1.
            # additional constant
            C2 = C1 - np.exp(-elim**2/(2.*self.esigma**2))
        else:
            C2 = self.enorm
        e = self.esigma*np.sqrt(-2.*np.log(C1 - C2*np.random.uniform(size=n)))
        # generate albedo (independent)
        _, sma = self.gen_radius_sma(n)
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(sma)
        
        return a, e, p, Rp

    def dist_lnradius_lnperiod(self, lnRp, lnT):
        """Probability density function for logarithm of planetary radius 
        in Earth radius, and orbital period in year.
        
        This method implements the SAG13 broken power law. It returns the joint 
        distribution of the logarithm of planetary radius and orbital period 
        values (d2N / (dlnRp * dlnT).
        
        It also integrates the decay point (smaknee/Tknee) defined 
        in the KeplerLike module.
        
        Args:
            lnRp (float ndarray):
                Logarithm of planetary radius value(s) in Earth radius. 
                Not an astropy quantity.
            lnT (float ndarray):
                Logarithm of orbital period value(s) in year. Not an astropy quantity.
                
        Returns:
            f (ndarray):
                Joint (radius and period) probability density matrix
                of shape (len(x),len(y))
        
        """
        
        # SAG13 coeffs
        Gamma = self.SAG13coeffs[0,:]
        alpha = self.SAG13coeffs[1,:]
        beta = self.SAG13coeffs[2,:]
        Rplim = np.append(self.SAG13coeffs[3,:], np.inf)
        
        # get radius and period values
        Rp = np.exp(lnRp)
        T = np.exp(lnT)
        
        # decay exponential
        decay = np.exp(-(T/self.Tknee)**2)
        
        # scalar case
        if np.size(Rp)*np.size(T) == 1:
            for k in range(len(Rplim) - 1):
                if (Rp >= Rplim[k]) & (Rp < Rplim[k+1]):
                    f = Gamma[k] * Rp**alpha[k] * T**beta[k] 
            # apply exponential decay
            f *= decay
        
        # array case
        else:
            # create an (Rp, T) coordinate matrix
            Rps, Ts = np.meshgrid(Rp, T)
            # generate the probability density matrix
            f = np.zeros(Rps.shape)
            for k in range(len(Rplim) - 1):
                mask = (Rps[0,:] >= Rplim[k]) & (Rps[0,:] < Rplim[k+1])
                f[:,mask] = Gamma[k] * Rps[:,mask]**alpha[k] * Ts[:,mask]**beta[k] 
            # apply exponential decay
            f = (f.T*decay).T
        
        return f
