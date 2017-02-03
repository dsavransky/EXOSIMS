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

class GarrettCompleteness(BrownCompleteness):
    """Analytical Completeness class
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations based on Garrett and Savransky 2016
    in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        visits (ndarray):
            Number of observations corresponding to each star in the target list
            (initialized in gen_update)
        updates (nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, **specs):
        # bring in inherited Completeness prototype __init__ values
        BrownCompleteness.__init__(self, **specs)
        # get path to completeness interpolant stored in a pickled .comp file
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        self.filename = specs['modules']['PlanetPopulation']
        # get unitless values of population parameters
        self.amin = self.PlanetPopulation.arange.min().value
        self.amax = self.PlanetPopulation.arange.max().value
        self.emin = self.PlanetPopulation.erange.min()
        self.emax = self.PlanetPopulation.erange.max()
        self.pmin = self.PlanetPopulation.prange.min()
        self.pmax = self.PlanetPopulation.prange.max()
        self.Rmin = self.PlanetPopulation.Rprange.min().value
        self.Rmax = self.PlanetPopulation.Rprange.max().value
        self.rmin = self.amin*(1.0 - self.emax)
        self.rmax = self.amax*(1.0 + self.emax)
        self.zmin = self.pmin*self.Rmin**2
        self.zmax = self.pmax*self.Rmax**2
        # conversion factor
        self.x = self.PlanetPopulation.Rprange.unit.to(self.PlanetPopulation.arange.unit)
        # distributions needed
        self.dist_sma = self.PlanetPopulation.dist_sma
        self.dist_eccen = self.PlanetPopulation.dist_eccen
        self.dist_albedo = self.PlanetPopulation.dist_albedo
        self.dist_radius = self.PlanetPopulation.dist_radius
        # are any of a, e, p, Rp constant?
        self.aconst = self.amin == self.amax
        self.econst = self.emin == self.emax
        self.pconst = self.pmin == self.pmax
        self.Rconst = self.Rmin == self.Rmax
        # solve for bstar
        beta = np.linspace(0.0,np.pi,1000)*u.rad
        Phis = self.PlanetPhysicalModel.calc_Phi(beta).value
        # Interpolant for phase function which removes astropy Quantity
        self.Phi = interpolate.InterpolatedUnivariateSpline(beta.value,Phis,k=3,ext=1)
        # get numerical derivative of phase function
        dPhis = np.zeros(beta.shape)
        db = beta[1].value - beta[0].value
        dPhis[0:1] = (-25.0*Phis[0:1]+48.0*Phis[1:2]-36.0*Phis[2:3]+16.0*Phis[3:4]-3.0*Phis[4:5])/(12.0*db)
        dPhis[-2:-1] = (25.0*Phis[-2:-1]-48.0*Phis[-3:-2]+36.0*Phis[-4:-3]-16.0*Phis[-5:-4]+3.0*Phis[-6:-5])/(12.0*db)
        dPhis[2:-2] = (Phis[0:-4]-8.0*Phis[1:-3]+8.0*Phis[3:-1]-Phis[4:])/(12.0*db)
        self.dPhi = interpolate.InterpolatedUnivariateSpline(beta.value,dPhis,k=3,ext=1)
        # solve for bstar        
        f = lambda b: 2.0*np.sin(b)*np.cos(b)*self.Phi(b) + np.sin(b)**2*self.dPhi(b)
        self.bstar = optimize.root(f,np.pi/3.0).x
                
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
        dMagmax = TL.OpticalSystem.dMagLim #np.array([targlist.OpticalSystem.dMagLim]*targlist.nStars)
        dMagmin = self.mindmag(smin)
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
            print 'Creating preliminary functions.'
            # vectorize integrand
            self.rgrand2v = np.vectorize(self.rgrand2)
            # threshold value used later
            self.val = np.sin(self.bstar)**2*self.Phi(self.bstar)
            # inverse functions for phase angle
            b1 = np.linspace(0.0, self.bstar, 25000)
            # b < bstar
            self.binv1 = interpolate.InterpolatedUnivariateSpline(np.sin(b1)**2*self.Phi(b1), b1, k=3, ext=1)
            b2 = np.linspace(self.bstar, np.pi, 25000)
            b2val = np.sin(b2)**2*self.Phi(b2)
            # b > bstar
            self.binv2 = interpolate.InterpolatedUnivariateSpline(b2val[::-1], b2[::-1], k=3, ext=1)
            print 'Generating pdf of orbital radius'            
            # get pdf of r
            r = np.linspace(self.rmin, self.rmax, 1000)
            fr = np.zeros(r.shape)
            for i in xrange(len(r)):
                fr[i] = self.f_r(r[i])
            self.dist_r = interpolate.InterpolatedUnivariateSpline(r, fr, k=3, ext=1)
            print 'Finished pdf of orbital radius'
            print 'Generating pdf of albedo times planetary radius squared'
            # get pdf of p*R**2
            z = np.linspace(self.pmin*self.Rmin**2, self.pmax*self.Rmax**2, 1000)
            fz = np.zeros(z.shape)
            for i in xrange(len(z)):
                fz[i] = self.f_z(z[i])
            self.dist_z = interpolate.InterpolatedUnivariateSpline(z, fz, k=3, ext=1)
            print 'Finished pdf of albedo times planetary radius squared'
            self.f_dmagsv = np.vectorize(self.f_dmags)
            print 'Marginalizing joint pdf of separation and dMag up to dMagLim'
            # get pdf of s up to dmaglim
            s = np.linspace(0.0,self.rmax,1000)
            fs = np.zeros(s.shape)
            for i in xrange(len(s)):
                fs[i] = self.f_s(s[i],TL.OpticalSystem.dMagLim)
            dist_s = interpolate.InterpolatedUnivariateSpline(s, fs, k=3, ext=1)
            print 'Finished marginalization'
            H = {'dist_s': dist_s}
            pickle.dump(H, open(Cpath, 'wb'))
            print 'Completeness data stored in %s' % Cpath
            
        return dist_s
            
        
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
        
        z = np.array(z, ndmin=1, copy=False)

        vals = (s/self.x)**2*10.**(-0.4*dmag)/z
        
        if vals.max() > self.val:
            f = np.zeros(z.shape)
            b1 = self.binv1(vals[vals<self.val])
            b2 = self.binv2(vals[vals<self.val])
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            if (self.pconst & self.Rconst):
                f[vals<self.val] = self.dist_b(b1)*self.dist_r(r1)/np.abs(self.Jac(b1))
                f[vals<self.val] += self.dist_b(b2)*self.dist_r(r2)/np.abs(self.Jac(b2))
            else:
                f[vals<self.val] = self.dist_z(z[vals<self.val])*self.dist_b(b1)*self.dist_r(r1)/np.abs(self.Jac(b1))
                f[vals<self.val] += self.dist_z(z[vals<self.val])*self.dist_b(b2)*self.dist_r(r2)/np.abs(self.Jac(b2))
        else:
            f = np.zeros(z.shape)
            b1 = self.binv1(vals)
            b2 = self.binv2(vals)
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            if (self.pconst & self.Rconst):
                f = self.dist_b(b1)*self.dist_r(r1)/np.abs(self.Jac(b1))
                f+= self.dist_b(b2)*self.dist_r(r2)/np.abs(self.Jac(b2))
            else:
                f = self.dist_z(z)*self.dist_b(b1)*self.dist_r(r1)/np.abs(self.Jac(b1))
                f += self.dist_z(z)*self.dist_b(b2)*self.dist_r(r2)/np.abs(self.Jac(b2))
    
        return f
        
    def mindmag(self, s):
        """Calculates the minimum value of dMag for projected separation
        
        Args:
            s (ndarray):
                Projected separations (AU)
        
        Returns:
            mindmag (ndarray):
                Minimum value of dMag
                
        """
        
        s = np.array(s, ndmin=1, copy=False)

        mindmag = -2.5*np.log10(self.pmax*(self.Rmax*self.x*np.sin(self.bstar)/s)**2*self.Phi(self.bstar))
        mindmag[s < self.rmin*np.sin(self.bstar)] = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmin)**2*self.Phi(np.arcsin(s[s < self.rmin*np.sin(self.bstar)]/self.rmin)))
        mindmag[s > self.rmax*np.sin(self.bstar)] = -2.5*np.log10(self.pmax*(self.Rmax*self.x/self.rmax)**2*self.Phi(np.arcsin(s[s > self.rmax*np.sin(self.bstar)]/self.rmax)))
        
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
        
        maxdmag = -2.5*np.log10(self.pmin*(self.Rmin*self.x/self.rmax)**2*self.Phi(np.pi - np.arcsin(s/self.rmax)))

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
        if np.isnan(f).any():
            f = np.where(~np.isnan(f), f, np.inf)
        
        return f

    def rgrand1(self, e, a, r):
        """Calculates first integrand for determinining probability density of
        orbital radius
        
        Args:
            e (ndarray):
                Values of eccentricity
            a (ndarray):
                Values of semi-major axis in AU
            r (ndarray):
                Values of orbital radius in AU
        
        Returns:
            f (ndarray):
                Values of first integrand
        
        """
        
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.dist_eccen(e)*self.dist_sma(a)
        
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
        if emin1 < self.emin:
            emin1 = self.emin
    
        if emin1 > self.emax:
            f = 0.0
        else:
            f = integrate.fixed_quad(self.rgrand1, emin1, self.emax, args=(a,r), n=100)[0]

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
                    f = self.PlanetPopulation.dist_sma(r)
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
                    f = integrate.fixed_quad(self.rgrandac, low, self.emax, args=(self.amin,r), n=200)[0]
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
                    f = integrate.fixed_quad(self.rgrandec, low, high, args=(self.emin,r), n=200)[0]
            else:
                a1 = r/(1.0+self.emax)
                a2 = r/(1.0-self.emax)
                if a1 < self.amin:
                    a1 = self.amin
                if a2 > self.amax:
                    a2 = self.amax
                f = integrate.fixed_quad(self.rgrand2v, a1, a2, args=(r,), n=200)[0]
    
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
    
    def dist_b(self, b):
        """Calculates the probability density of phase angle
        
        The phase angle is sinusoidally distributed.
        
        Args:
            b (ndarray/float):
                Value(s) of phase angle (rad)
        
        Returns:
            f (ndarray/float):
                Value(s) of probability density
        
        """
            
        b = np.array(b, ndmin=1, copy=False)
            
        f = ((b >= 0.0) & (b <= np.pi)).astype(int)*np.sin(b)/2.0    
        
        return f