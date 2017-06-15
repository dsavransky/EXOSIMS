# -*- coding: utf-8 -*-

from EXOSIMS.Completeness.BrownCompleteness import BrownCompleteness

import numpy as np
import os, inspect
import hashlib
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import astropy.units as u
try:
    import cPickle as pickle
except ImportError:
    import pickle
from EXOSIMS.util.memoize import memoize

class GarrettCompleteness(BrownCompleteness):
    """Analytical Completeness class
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations based on Garrett and Savransky 2016
    in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        updates (nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, **specs):
        # bring in inherited Completeness prototype __init__ values
        BrownCompleteness.__init__(self, **specs)
        # get path to completeness interpolant stored in a pickled .comp file
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        self.filename = specs['modules']['PlanetPopulation'] + specs['modules']['PlanetPhysicalModel']
        # get unitless values of population parameters
        self.amin = float(self.PlanetPopulation.arange.min().value)
        self.amax = float(self.PlanetPopulation.arange.max().value)
        self.emin = float(self.PlanetPopulation.erange.min())
        self.emax = float(self.PlanetPopulation.erange.max())
        self.pmin = float(self.PlanetPopulation.prange.min())
        self.pmax = float(self.PlanetPopulation.prange.max())
        self.Rmin = float(self.PlanetPopulation.Rprange.min().to('km').value)
        self.Rmax = float(self.PlanetPopulation.Rprange.max().to('km').value)
        if self.PlanetPopulation.constrainOrbits:
            self.rmin = self.amin
            self.rmax = self.amax
        else:
            self.rmin = self.amin*(1.0 - self.emax)
            self.rmax = self.amax*(1.0 + self.emax)
        self.zmin = self.pmin*self.Rmin**2
        self.zmax = self.pmax*self.Rmax**2
        # conversion factor
        self.x = float(u.km.to('AU'))
        # distributions needed
        self.dist_sma = self.PlanetPopulation.adist
        self.dist_eccen = self.PlanetPopulation.edist
        self.dist_eccen_con = self.PlanetPopulation.edist_from_sma
        self.dist_albedo = self.PlanetPopulation.pdist
        self.dist_radius = self.PlanetPopulation.Rpdist
        # are any of a, e, p, Rp constant?
        self.aconst = self.amin == self.amax
        self.econst = self.emin == self.emax
        self.pconst = self.pmin == self.pmax
        self.Rconst = self.Rmin == self.Rmax
        # solve for bstar
        beta = np.linspace(0.0,np.pi,1000)*u.rad
        Phis = self.PlanetPhysicalModel.calc_Phi(beta).value
        # Interpolant for phase function which removes astropy Quantity
        self.Phi = interpolate.InterpolatedUnivariateSpline(beta.value,Phis,k=1,ext=1)
        self.Phiinv = interpolate.InterpolatedUnivariateSpline(Phis[::-1],beta.value[::-1],k=1,ext=1)
        # get numerical derivative of phase function
        dPhis = np.zeros(beta.shape)
        db = beta[1].value - beta[0].value
        dPhis[0:1] = (-25.0*Phis[0:1]+48.0*Phis[1:2]-36.0*Phis[2:3]+16.0*Phis[3:4]-3.0*Phis[4:5])/(12.0*db)
        dPhis[-2:-1] = (25.0*Phis[-2:-1]-48.0*Phis[-3:-2]+36.0*Phis[-4:-3]-16.0*Phis[-5:-4]+3.0*Phis[-6:-5])/(12.0*db)
        dPhis[2:-2] = (Phis[0:-4]-8.0*Phis[1:-3]+8.0*Phis[3:-1]-Phis[4:])/(12.0*db)
        self.dPhi = interpolate.InterpolatedUnivariateSpline(beta.value,dPhis,k=1,ext=1)
        # solve for bstar        
        f = lambda b: 2.0*np.sin(b)*np.cos(b)*self.Phi(b) + np.sin(b)**2*self.dPhi(b)
        self.bstar = float(optimize.root(f,np.pi/3.0).x)
        # helpful constants
        self.cdmin1 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmin)**2)
        self.cdmin2 = -2.5*np.log10(self.pmax*(self.Rmax*self.x*np.sin(self.bstar))**2*self.Phi(self.bstar))
        self.cdmin3 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmax)**2)
        self.cdmax = -2.5*np.log10(self.pmin*(self.Rmin*self.x/self.rmax)**2)
        self.val = np.sin(self.bstar)**2*self.Phi(self.bstar)
        self.d1 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmin)**2)
        self.d2 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmin)**2*self.Phi(self.bstar))
        self.d3 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmax)**2*self.Phi(self.bstar))
        self.d4 = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmax)**2*self.Phi(np.pi/2.0))
        self.d5 = -2.5*np.log10(self.pmin*(self.Rmin*self.x/self.rmax)**2*self.Phi(np.pi/2.0))
        # vectorize scalar methods
        self.rgrand2v = np.vectorize(self.rgrand2)
        self.f_dmagsv = np.vectorize(self.f_dmags)
        self.f_sdmagv = np.vectorize(self.f_sdmag)
        self.f_dmagv = np.vectorize(self.f_dmag)
        self.mindmagv = np.vectorize(self.mindmag)
        # inverse functions for phase angle
        b1 = np.linspace(0.0, self.bstar, 1000)
        # b < bstar
        self.binv1 = interpolate.InterpolatedUnivariateSpline(np.sin(b1)**2*self.Phi(b1), b1, k=1, ext=1)
        b2 = np.linspace(self.bstar, np.pi, 1000)
        b2val = np.sin(b2)**2*self.Phi(b2)
        # b > bstar
        self.binv2 = interpolate.InterpolatedUnivariateSpline(b2val[::-1], b2[::-1], k=1, ext=1)
        # get pdf of r
        print 'Generating pdf of orbital radius'
        r = np.linspace(self.rmin, self.rmax, 1000)
        fr = np.zeros(r.shape)
        for i in xrange(len(r)):
            fr[i] = self.f_r(r[i])
        self.dist_r = interpolate.InterpolatedUnivariateSpline(r, fr, k=1, ext=1)
        print 'Finished pdf of orbital radius'
        # get pdf of p*R**2
        print 'Generating pdf of albedo times planetary radius squared'
        z = np.linspace(self.zmin, self.zmax, 1000)
        fz = np.zeros(z.shape)
        for i in xrange(len(z)):
            fz[i] = self.f_z(z[i])
        self.dist_z = interpolate.InterpolatedUnivariateSpline(z, fz, k=1, ext=1)
        print 'Finished pdf of albedo times planetary radius squared'
                
    def target_completeness(self, TL):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList module): 
                TargetList class object
            
        Returns:
            comp0 (ndarray): 
                1D numpy array of completeness values for each target star
        
        """
        
        # important PlanetPopulation attributes
        atts = ['arange','erange','prange','Rprange','Mprange','scaleOrbits','constrainOrbits']
        extstr = ''
        for att in atts:
            extstr += '%s: ' % att + str(getattr(self.PlanetPopulation, att)) + ' '
        # include dMagLim
        extstr += '%s: ' % 'dMagLim' + str(getattr(TL.OpticalSystem, 'dMagLim')) + ' '
        ext = hashlib.md5(extstr).hexdigest()
        self.filename += ext
        Cpath = os.path.join(self.classpath, self.filename+'.acomp')
        
        dist_s = self.genComp(Cpath, TL)

        dist_sv = np.vectorize(dist_s.integral)
        
        # calculate separations based on IWA
        smin = (np.tan(TL.OpticalSystem.IWA)*TL.dist).to('AU').value
        if np.isinf(TL.OpticalSystem.OWA):
            smax = self.rmax
        else:
            smax = (np.tan(TL.OpticalSystem.OWA)*TL.dist).to('AU').value

        # calculate dMags based on limiting dMag
        dMagmax = TL.OpticalSystem.dMagLim
        dMagmin = self.mindmagv(smin)
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L>0, TL.L, 1e-10) #take care of zero/negative values
            smin = smin/np.sqrt(L)
            smax = smax/np.sqrt(L)
            dMagmin -= 2.5*np.log10(L)
            dMagmax -= 2.5*np.log10(L)
        
        comp0 = dist_sv(smin, smax)

        return comp0
        
    def genComp(self, Cpath, TL):
        """Generates function to get completeness values
        
        Args:
            Cpath (str):
                Path to pickled dictionary containing interpolant function
            TL (TargetList module): 
                TargetList class object
                
        Returns:
            dist_s (callable(s)):
                Marginalized to dMagLim probability density function for 
                projected separation
        
        """
        
        if os.path.exists(Cpath):
            # dist_s interpolant already exists for parameters
            print 'Loading cached completeness file from %s' % Cpath
            H = pickle.load(open(Cpath, 'rb'))
            print 'Completeness loaded from cache.'
            dist_s = H['dist_s']
        else:
            # generate dist_s interpolant and pickle it
            print 'Cached completeness file not found at "%s".' % Cpath
            print 'Generating completeness.'
            print 'Marginalizing joint pdf of separation and dMag up to dMagLim'
            # get pdf of s up to dmaglim
            s = np.linspace(0.0,self.rmax,1000)
            fs = np.zeros(s.shape)
            for i in xrange(len(s)):
                fs[i] = self.f_s(s[i],TL.OpticalSystem.dMagLim)
            dist_s = interpolate.InterpolatedUnivariateSpline(s, fs, k=1, ext=1)
            print 'Finished marginalization'
            H = {'dist_s': dist_s}
            pickle.dump(H, open(Cpath, 'wb'))
            print 'Completeness data stored in %s' % Cpath
            
        return dist_s
            
    @memoize    
    def f_s(self, s, dmaglim):
        """Calculates probability density of projected separation marginalized
        up to dmaglim
        
        Args:
            s (float):
                Value of projected separation
            dmaglim (float):
                Value of limiting dMag
                
        Returns:
            f (float):
                Probability density
        
        """
        
        if (s == 0.0) or (s == self.rmax):
            f = 0.0
        else:
            d1 = self.mindmag(s)
            d2 = self.maxdmag(s)
            if d2 > dmaglim:
                d2 = dmaglim
            if d1 > d2:
                f = 0.0
            else:
                f = integrate.fixed_quad(self.f_dmagsv, d1, d2, args=(s,), n=31)[0]
        
        return f
    
    @memoize    
    def f_dmags(self, dmag, s):
        """Calculates the joint probability density of dMag and projected
        separation
        
        Args:
            dmag (float):
                Value of dMag
            s (float):
                Value of projected separation (AU)
        
        Returns:
            f (float):
                Value of joint probability density
        
        """
        
        if (dmag < self.mindmag(s)) or (dmag > self.maxdmag(s)):
            f = 0.0
        else:
            ztest = (s/self.x)**2*10.**(-0.4*dmag)/self.val
            if ztest >= self.zmax:
                f = 0.0
            elif (self.pconst & self.Rconst):
                f = self.f_dmagsz(self.zmin,dmag,s)
            else:
                if ztest < self.zmin:
                    f = integrate.fixed_quad(self.f_dmagsz, self.zmin, self.zmax, args=(dmag, s), n=61)[0]
                else:
                    f = integrate.fixed_quad(self.f_dmagsz, ztest, self.zmax, args=(dmag, s), n=61)[0]
        return f
    
    def f_dmagsz(self, z, dmag, s):
        """Calculates the joint probability density of albedo times planetary 
        radius squared, dMag, and projected separation
        
        Args:
            z (ndarray):
                Values of albedo times planetary radius squared
            dmag (float):
                Value of dMag
            s (float):
                Value of projected separation
        
        Returns:
            f (ndarray):
                Values of joint probability density
                
        """
        if not isinstance(z,np.ndarray):
            z = np.array(z, ndmin=1, copy=False)

        vals = (s/self.x)**2*10.**(-0.4*dmag)/z
        
        f = np.zeros(z.shape)
        fa = f[vals<self.val]
        za = z[vals<self.val]
        valsa = vals[vals<self.val]
        b1 = self.binv1(valsa)
        b2 = self.binv2(valsa)
        r1 = s/np.sin(b1)
        r2 = s/np.sin(b2)
        good1 = ((r1>self.rmin)&(r1<self.rmax))
        good2 = ((r2>self.rmin)&(r2<self.rmax))
        if (self.pconst & self.Rconst):
            fa[good1] = np.sin(b1[good1])/2.0*self.dist_r(r1[good1])/np.abs(self.Jac(b1[good1]))
            fa[good2] += np.sin(b2[good2])/2.0*self.dist_r(r2[good2])/np.abs(self.Jac(b2[good2]))
        else:
            fa[good1] = self.dist_z(za[good1])*np.sin(b1[good1])/2.0*self.dist_r(r1[good1])/np.abs(self.Jac(b1[good1]))
            fa[good2] += self.dist_z(za[good2])*np.sin(b2[good2])/2.0*self.dist_r(r2[good2])/np.abs(self.Jac(b2[good2]))
            
        f[vals<self.val] = fa
        
        return f
    
    def mindmag(self, s):
        """Calculates the minimum value of dMag for projected separation
        
        Args:
            s (float):
                Projected separations (AU)
        
        Returns:
            mindmag (float):
                Minimum value of dMag
                
        """
        if s == 0.0:
            mindmag = self.cdmin1
        elif s <= self.rmin*np.sin(self.bstar):
            mindmag = self.cdmin1-2.5*np.log10(self.Phi(np.arcsin(s/self.rmin)))
        elif s <= self.rmax*np.sin(self.bstar):
            mindmag = self.cdmin2+5.0*np.log10(s)
        elif s <= self.rmax:
            mindmag = self.cdmin3-2.5*np.log10(self.Phi(np.arcsin(s/self.rmax)))
        else:
            mindmag = np.inf
        
        return mindmag
    
    def maxdmag(self, s):
        """Calculates the maximum value of dMag for projected separation
        
        Args:
            s (ndarray):
                Projected separations (AU)
        
        Returns:
            maxdmag (ndarray):
                Maximum value of dMag
        
        """
        
        maxdmag = self.cdmax-2.5*np.log10(self.Phi(np.pi - np.arcsin(s/self.rmax)))

        return maxdmag

    def Jac(self, b):
        """Calculates determinant of the Jacobian transformation matrix to get
        the joint probability density of dMag and s
        
        Args:
            b (ndarray):
                Phase angles
                
        Returns:
            f (ndarray):
                Determinant of Jacobian transformation matrix
        
        """
        
        f = -2.5/(self.Phi(b)*np.log(10.0))*self.dPhi(b)*np.sin(b) - 5./np.log(10.0)*np.cos(b)
        
        return f
    
    def rgrand1(self, e, a, r):
        """Calculates first integrand for determinining probability density of
        orbital radius
        
        Args:
            e (ndarray):
                Values of eccentricity
            a (float):
                Values of semi-major axis in AU
            r (float):
                Values of orbital radius in AU
        
        Returns:
            f (ndarray):
                Values of first integrand
        
        """
        if self.PlanetPopulation.constrainOrbits:
            f = 1.0/(np.sqrt((a*e)**2-(a-r)**2))*self.dist_eccen_con(e,a)
        else:
            f = 1.0/(np.sqrt((a*e)**2-(a-r)**2))*self.dist_eccen(e)
        
        return f
    
    def rgrand2(self, a, r):
        """Calculates second integrand for determining probability density of
        orbital radius
        
        Args:
            a (float):
                Value of semi-major axis in AU
            r (float):
                Value of orbital radius in AU
                
        Returns:
            f (float):
                Value of second integrand
        
        """
        emin1 = np.abs(1.0 - r/a)
        emin1 *= (1.0+1e-3)
        if emin1 < self.emin:
            emin1 = self.emin
    
        if emin1 >= self.emax:
            f = 0.0
        else:
            if self.PlanetPopulation.constrainOrbits:
                if a <= 0.5*(self.amin+self.amax):
                    elim = 1.0-self.amin/a
                else:
                    elim = self.amax/a - 1.0
                if emin1 > elim:
                    f = 0.0
                else:
                    f = self.dist_sma(a)/a*integrate.fixed_quad(self.rgrand1,emin1,elim, args=(a,r), n=50)[0]
            else:
                f = self.dist_sma(a)/a*integrate.fixed_quad(self.rgrand1, emin1, self.emax, args=(a,r), n=50)[0]

        return f
        
    def rgrandac(self, e, a, r):
        """Calculates integrand for determining probability density of orbital
        radius when semi-major axis is constant
        
        Args:
            e (ndarray):
                Values of eccentricity
            a (float):
                Value of semi-major axis in AU
            r (float):
                Value of orbital radius in AU
        
        Returns:
            f (ndarray):
                Value of integrand
        
        """
        if self.PlanetPopulation.constrainOrbits:
            f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.dist_eccen_con(e,a)
        else:
            f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.dist_eccen(e)
        
        return f
        
    def rgrandec(self, a, e, r):
        """Calculates integrand for determining probability density of orbital
        radius when eccentricity is constant
        
        Args:
            a (ndarray):
                Values of semi-major axis in AU
            e (float):
                Value of eccentricity
            r (float):
                Value of orbital radius in AU
        
        Returns:
            f (float):
                Value of integrand
        """
        
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.dist_sma(a)
        
        return f

    def f_r(self, r):
        """Calculates the probability density of orbital radius
        
        Args:
            r (float):
                Value of semi-major axis in AU
                
        Returns:
            f (float):
                Value of probability density
        
        """
        # takes scalar input
        if (r == self.rmin) or (r == self.rmax):
            f = 0.0
        else:
            if (self.aconst & self.econst):
                if self.emin == 0.0:
                    f = self.dist_sma(r)
                else:
                    if r > self.amin*(1.0-self.emin):
                        f = (r/(np.pi*self.amin*np.sqrt((self.amin*self.emin)**2-(self.amin-r)**2)))
                    else:
                        f = 0.0
            elif self.aconst:
                etest1 = 1.0 - r/self.amin
                etest2 = r/self.amin - 1.0
                if self.emax < etest1:
                    f = 0.0
                else:
                    if r < self.amin:
                        if self.emin > etest1:
                            low = self.emin
                        else:
                            low = etest1
                    else:
                        if self.emin > etest2:
                            low = self.emin
                        else:
                            low = etest2
                    f = integrate.fixed_quad(self.rgrandac, low, self.emax, args=(self.amin,r), n=40)[0]
            elif self.econst:
                if self.emin == 0.0:
                    f = self.dist_sma(r)
                else:
                    atest1 = r/(1.0-self.emin)
                    atest2 = r/(1.0+self.emin)
                    if self.amax < atest1:
                        high = self.amax
                    else:
                        high = atest1
                    if self.amin < atest2:
                        low = atest2
                    else:
                        low = self.amin
                    f = integrate.fixed_quad(self.rgrandec, low, high, args=(self.emin,r), n=40)[0]
            else:
                if self.PlanetPopulation.constrainOrbits:
                    a1 = 0.5*(self.amin+r)
                    a2 = 0.5*(self.amax+r)
                else:
                    a1 = r/(1.0+self.emax)
                    a2 = r/(1.0-self.emax)
                    if a1 < self.amin:
                        a1 = self.amin
                    if a2 > self.amax:
                        a2 = self.amax
                f = r/np.pi*integrate.fixed_quad(self.rgrand2v, a1, a2, args=(r,), n=40)[0]
    
        return f
        
    def pgrand(self, p, z):
        """Calculates integrand for determining probability density of albedo
        times planetary radius squared
        
        Args:
            p (ndarray):
                Values of albedo
            z (float):
                Value of albedo times planetary radius squared
        
        Returns:
            f (ndarray):
                Values of integrand
        
        """
        
        f = 1.0/(2.0*np.sqrt(z*p))*self.dist_radius(np.sqrt(z/p))*self.dist_albedo(p)
        
        return f
  
    def f_z(self, z):
        """Calculates probability density of albedo times planetary radius 
        squared
        
        Args:
            z (float):
                Value of albedo times planetary radius squared
        
        Returns:
            f (float):
                Probability density
        
        """
        
        # takes scalar input
        if (z < self.pmin*self.Rmin**2) or (z > self.pmax*self.Rmax**2):
            f = 0.0
        else:
            if (self.pconst & self.Rconst):
                f = 1.0
            elif self.pconst:
                f = 1.0/(2.0*np.sqrt(self.pmin*z))*self.dist_radius(np.sqrt(z/self.pmin))
            elif self.Rconst:
                f = 1.0/self.Rmin**2*self.dist_albedo(z/self.Rmin**2)
            else:
                p1 = z/self.Rmax**2
                p2 = z/self.Rmin**2
                if p1 < self.pmin:
                    p1 = self.pmin
                if p2 > self.pmax:
                    p2 = self.pmax
                f = integrate.fixed_quad(self.pgrand,p1,p2,args=(z,),n=200)[0]
                
        return f
    
    def s_bound(self, dmag, smax):
        """Calculates the bounding value of projected separation for dMag
        
        Args:
            dmag (float):
                dMag value
            smax (float):
                maximum projected separation (AU)
        
        Returns:
            sb (float):
                boundary value of projected separation (AU)
        """
        
        if dmag < self.d1:
            s = 0.0
        elif (dmag > self.d1) and (dmag <= self.d2):
            s = self.rmin*np.sin(self.Phiinv(self.rmin**2*10.0**(-0.4*dmag)/(self.pmax*(self.Rmax*self.x)**2)))
        elif (dmag > self.d2) and (dmag <= self.d3):
            s = np.sin(self.bstar)*np.sqrt(self.pmax*(self.Rmax*self.x)**2*self.Phi(self.bstar)/10.0**(-0.4*dmag))
        elif (dmag > self.d3) and (dmag <= self.d4):
            s = self.rmax*np.sin(self.Phiinv(self.rmax**2*10.0**(-0.4*dmag)/(self.pmax*(self.Rmax*self.x)**2)))
        elif (dmag > self.d4) and (dmag <= self.d5):
            s = smax
        else:
            s = self.rmax*np.sin(np.pi - self.Phiinv(10.0**(-0.4*dmag)*self.rmax**2/(self.pmin*(self.Rmin*self.x)**2)))
    
        return s
    
    def f_sdmag(self, s, dmag):
        """Calculates the joint probability density of projected separation and
        dMag by flipping the order of f_dmags
        
        Args:
            s (float):
                Value of projected separation (AU)
            dmag (float):
                Value of dMag
        
        Returns:
            f (float):
                Value of joint probability density
        
        """
        return self.f_dmags(dmag, s)
    
    @memoize
    def f_dmag(self, dmag, smin, smax):
        """Calculates probability density of dMag by integrating over projected
        separation
        
        Args:
            dmag (float):
                Value of dMag
            smin (float):
                Value of minimum projected separation (AU) from instrument
            smax (float):
                Value of maximum projected separation (AU) from instrument
        
        Returns:
            f (float):
                Value of probability density
        
        """
        if dmag < self.mindmag(smin):
            f = 0.0
        else:
            su = self.s_bound(dmag, smax)
            if su > smax:
                su = smax
            if su < smin:
                f = 0.0
            else:
                f = integrate.fixed_quad(self.f_sdmagv, smin, su, args=(dmag,), n=20)[0]
        
        return f
    
    def comp_dmag(self, smin, smax, dmaglim):
        """Calculates completeness by first integrating over projected 
        separation and then dMag.
        
        Args:
            smin (ndarray):
                Values of minimum projected separation (AU) from instrument
            smax (ndarray):
                Value of maximum projected separation (AU) from instrument
            dmaglim (ndarray):
                dMaglim from instrument
        
        Returns:
            comp (float):
                Completeness value
        
        """
        # cast to arrays
        smin = np.array(smin, ndmin=1)
        smax = np.array(smax, ndmin=1)
        dmaglim = np.array(dmaglim, ndmin=1)
        
        comp = np.zeros(smin.shape)
        for i in xrange(len(smin)):
            d1 = self.mindmag(smin[i])
            if d1 > dmaglim[i]:
                comp[i] = 0.0
            else:
                comp[i] = integrate.fixed_quad(self.f_dmagv, d1, dmaglim[i], args=(smin[i],smax[i]), n=31)[0]
        
        return comp
    
    def comp_per_intTime(self, t_int, TL, sInds, fZ, fEZ, WA, mode):
        """Calculates completeness for integration time
        
        Args:
            t_int (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
                
        Returns:
            comp (array):
                Completeness values
        
        """
        
        # cast inputs to arrays and check
        t_int = np.array(t_int.value, ndmin=1)*t_int.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(t_int) == len(sInds), "t_int and sInds must be same length"
        assert len(t_int) == len(fZ) or len(fZ) == 1, "fZ must be constant or have same length as t_int"
        assert len(t_int) == len(fEZ) or len(fEZ) == 1, "fEZ must be constant or have same length as t_int"
        assert len(WA) == 1, "WA must be constant"
        
        dMag = TL.OpticalSystem.calc_dMag_per_intTime(t_int, TL, sInds, fZ, fEZ, WA, mode).reshape((len(t_int),))
        smin = (np.tan(TL.OpticalSystem.IWA)*TL.dist[sInds]).to('AU').value
        smax = (np.tan(TL.OpticalSystem.OWA)*TL.dist[sInds]).to('AU').value
        comp = self.comp_dmag(smin, smax, dMag)
        
        return comp
        
    def dcomp_dt(self, t_int, TL, sInds, fZ, fEZ, WA, mode):
        """Calculates derivative of completeness with respect to integration time
        
        Args:
            t_int (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
                
        Returns:
            dcomp (array):
                Derivative of completeness with respect to integration time
        
        """
        
        # cast inputs to arrays and check
        t_int = np.array(t_int.value, ndmin=1)*t_int.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(t_int) == len(sInds), "t_int and sInds must be same length"
        assert len(t_int) == len(fZ) or len(fZ) == 1, "fZ must be constant or have same length as t_int"
        assert len(t_int) == len(fEZ) or len(fEZ) == 1, "fEZ must be constant or have same length as t_int"
        assert len(WA) == 1, "WA must be constant"
        
        dMag = TL.OpticalSystem.calc_dMag_per_intTime(t_int, TL, sInds, fZ, fEZ, WA, mode).reshape((len(t_int),))
        smin = (np.tan(TL.OpticalSystem.IWA)*TL.dist[sInds]).to('AU').value
        smax = (np.tan(TL.OpticalSystem.OWA)*TL.dist[sInds]).to('AU').value
        fdmag = np.zeros(t_int.shape)
        for i in xrange(len(t_int)):
            fdmag[i] = self.f_dmagv(dMag[i], smin[i], smax[i])
        ddMag = TL.OpticalSystem.ddMag_dt(t_int, TL, sInds, fZ, fEZ, WA, mode).reshape((len(fdmag),))
        dcomp = fdmag*ddMag
        
        return dcomp