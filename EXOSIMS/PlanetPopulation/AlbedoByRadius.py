from EXOSIMS.PlanetPopulation.SAG13 import SAG13
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.integrate as integrate

class AlbedoByRadius(SAG13):
    """Planet Population module based on SAG13 occurrence rates.
    
    NOTE: This assigns constant albedo based on radius range.
    
    Attributes: 
        SAG13coeffs (float 4x2 ndarray):
            Coefficients used by the SAG13 broken power law. The 4 lines
            correspond to Gamma, alpha, beta, and the minimum radius.
        Gamma (float ndarray):
            Gamma coefficients used by SAG13 broken power law.
        alpha (float ndarray):
            Alpha coefficients used by SAG13 broken power law.
        beta (float ndarray):
            Beta coefficients used by SAG13 broken power law.
        Rplim (float ndarray):
            Minimum radius used by SAG13 broken power law.
        SAG13starMass (astropy Quantity):
            Assumed stellar mass corresponding to the given set of coefficients.
        mu (astropy Quantity):
            Gravitational parameter associated with SAG13starMass.
        Ca (float 2x1 ndarray):
            Constants used for sampling.
        ps (float nx1 ndarray):
            Constant geometric albedo values.
        Rb (float (n-1)x1 ndarray):
            Planetary radius break points for albedos in earthRad.
    
    """
    
    def __init__(self, SAG13coeffs=[[.38, -.19, .26, 0.],[.73, -1.18, .59, 3.4]],
            SAG13starMass=1., Rprange=[2/3., 17.0859375],
            arange=[0.09084645, 1.45354324], ps=[0.2,0.5], Rb=[1.4], **specs):
        
        SAG13.__init__(self, SAG13coeffs=SAG13coeffs, SAG13starMass=SAG13starMass,
                       Rprange=Rprange, arange=arange, **specs)
        
        self.ps = np.array(ps, copy=False)
        self.Rb = np.array(Rb, copy=False)
        # check to ensure proper inputs
        assert len(self.ps) - len(self.Rb) == 1, \
            'input albedos must have one more element than break radii'
        
        # populate _outspec with new specific attributes
        self._outspec['ps'] = self.ps
        self._outspec['Rb'] = self.Rb
        
    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)
        
        Semi-major axis and planetary radius are jointly distributed. 
        Eccentricity is a Rayleigh distribution. Albedo is a constant value
        based on planetary radius.
        
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
            # restrict semi-major axis limits
            arcon = np.array([ar[0]/(1.-self.erange[0]), ar[1]/(1.+self.erange[0])])
            # clip sma values to sma range
            sma = np.clip(a.to('AU').value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(arcon)
            elim[sma <= amean] = 1. - ar[0]/sma[sma <= amean]
            elim[sma > amean] = ar[1]/sma[sma>amean] - 1.
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # additional constant
            C2 = C1 - np.exp(-elim**2/(2.*self.esigma**2))
            a = sma*u.AU
        else:
            C2 = self.enorm
        e = self.esigma*np.sqrt(-2.*np.log(C1 - C2*np.random.uniform(size=n)))
        # generate albedo from planetary radius
        Rs = np.hstack((0.,self.Rb,np.inf))
        p = np.zeros((n,))
        for i in xrange(len(Rs)-1):
            mask = np.where((Rp.to('earthRad').value>=Rs[i])&(Rp.to('earthRad').value<Rs[i+1]))
            p[mask] = self.ps[i]
        
        return a, e, p, Rp