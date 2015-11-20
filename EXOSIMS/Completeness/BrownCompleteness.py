# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os, inspect, copy
import cPickle as pickle
from EXOSIMS.Prototypes.Completeness import Completeness
import EXOSIMS.util.statsFun as statsFun 


class BrownCompleteness(Completeness):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        minComp (float): 
            minimum completeness level for detection
        Nplanets (float):
            number of planets for initial completeness Monte Carlo simulation
        EVPOC (interp2d):
            scipy interp2d object to find initial completeness values
        obsinds (ndarray):
            1D numpy ndarray containing target list star indices of stars
            which have been observed during simulation
        initialobstime (Quantity):
            1D numpy ndarray with units of time attached containing the 
            initial observation time of each star in obsinds
        accumulated (ndarray):
            accumulated completeness value for observed stars in obsinds
        tcomp (Quantity):
            1D numpy ndarray with units of day containing time to compare 
        non (ndarray):
            1D numpy ndarray containing completeness value for planets which
            are not observable at any time
        
    """
    
    def __init__(self, **specs):
        
        # bring in inherited Completeness prototype __init__ values
        Completeness.__init__(self, **specs)
        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        xedges = np.linspace(0., self.PlanetPopulation.arange.max().value*(1.+self.PlanetPopulation.erange.max()), bins)*self.PlanetPopulation.arange.unit
        xedges = xedges.to(u.AU).value

        # yedges is array of delta magnitude values for interpolant
        ymin = np.round((-2.5*np.log10(self.PlanetPopulation.prange.max()*(self.PlanetPopulation.Rrange.max()/(self.PlanetPopulation.arange.min()*(1.-self.PlanetPopulation.erange.max()))).decompose().value**2)))
        ymax = np.round((-2.5*np.log10(self.PlanetPopulation.prange.min()*(self.PlanetPopulation.Rrange.min()/(self.PlanetPopulation.arange.max()*(1.+self.PlanetPopulation.erange.max()))).decompose().value**2*1e-11)))
        yedges = np.linspace(ymin, ymax, bins)
        # generate completeness values for interpolant
        # total number of planets for completeness calcs
        self.Nplanets = 1e8
        # number of planets for each Monte Carlo simulation
        nplan = 1e6
        # number of simulations to perform (must be integer)
        steps = int(self.Nplanets/nplan)
        
        # get path to completeness interpolant stored in a pickled .comp file
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        filename = specs['modules']['PlanetPopulation']
#        if 'popname' in specs:
#            # if specific Planet Population module is used get its name
#            filename = specs['popname']
#        else:
#            # use the prototype
#            filename = 'PlanetPopulation'
        
        # path to 2D completeness array for interpolation
        Cpath = os.path.join(classpath, filename+'.comp')
        
        C, xedges2, yedges2 = self.genC(Cpath, nplan, xedges, yedges, steps)

        self.EVPOC = interpolate.interp2d(xedges, yedges, C)

        # attributes needed for completeness updates
        # indices of observed target stars
        self.obsinds = np.array([], dtype=int)
        # initial observation time of observed target stars
        self.initialobstime = np.array([])*u.day
        # accumulated completeness
        self.accumulated = np.array([])
        # time comparison value
        self.tcomp = np.array([])*u.day
        # non-observable completeness value
        self.non = np.array([])
    
    def __str__(self):
        """String representation of Completeness object
        
        When the command 'print' is used on the Completeness object, this 
        method will return the values contained in the object
        
        """

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Completeness class object attributes'
        
    def target_completeness(self, targlist):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            targlist (TargetList): 
                TargetList class object which, in addition to TargetList class
                object attributes, has available:
                    targlist.OpticalSystem: 
                        OpticalSystem class object
                    targlist.rules: 
                        Rules class object
                    targlist.ZodiacalLight: 
                        ZodiacalLight class object
            
        Returns:
            comp0 (ndarray): 
                1D numpy array of completeness values for each target star
        
        """
        
        # calculate separations based on IWA
        s = np.tan(targlist.OpticalSystem.IWA)*targlist.dist*u.pc
        # calculate dMags based on limiting dMag
        dMag = np.array([0.]*len(targlist.Name)) + targlist.OpticalSystem.dMagLim
        if self.PlanetPopulation.scaleOrbits:
            s = s/np.sqrt(targlist.L)
            dMag = dMag - 2.5*np.log10(targlist.L)
            
        comp0 = np.zeros((len(s),))
        # interpolation returns a 2D array, but comp0 should be 1D
        # for now do a for loop
        for i in xrange(len(s)):
            comp0[i] = self.EVPOC(s[i].to(u.AU).value, dMag[i])[0]/self.Nplanets
            
        return comp0
        
    def completeness_update(self, s_ind, targlist, obsbegin, obsend, nexttime):
        """Updates completeness value for stars previously observed
        
        Args:
            s_ind (int):
                index of star just observed
            targlist (TargetList):
                TargetList module
            obsbegin (Quantity):
                time of observation begin
            obsend (Quantity):
                time of observation end
            nexttime (Quantity):
                time of next observational period
        
        Returns:
            comp0 (ndarray):
                1D numpy ndarray of completeness values for each star in the 
                target list
        
        """
        comp0 = copy.copy(targlist.comp0)
        
        if s_ind not in self.obsinds:
            # unique observation of star, add to obsinds
            self.obsinds = np.hstack((self.obsinds, s_ind))
            # add observation begin time to initialobstime
            self.initialobstime = np.hstack((self.initialobstime.to(u.day).value, obsbegin.to(u.day).value))*u.day
            # add accumulated completeness to accumulated
            self.accumulated = np.hstack((self.accumulated, comp0[s_ind]))
            # store time comparison value
            s = np.tan(targlist.OpticalSystem.IWA)*targlist.dist[s_ind]*u.pc
            mu = const.G*(targlist.MsTrue[s_ind]*const.M_sun + targlist.PlanetPopulation.Mprange.max())
            t = (2.*np.abs(np.arcsin(s/targlist.PlanetPopulation.arange.max()).decompose().value)*np.sqrt(targlist.PlanetPopulation.arange.max()**3/mu)).to(u.day)
            self.tcomp = np.hstack((self.tcomp.to(u.day).value, t.to(u.day).value))*u.day
            # store non-observable completeness (rmax < s)
            a = statsFun.simpSample(targlist.PlanetPopulation.semi_axis, 20000, targlist.PlanetPopulation.arange.min().value, targlist.PlanetPopulation.arange.max().value)*targlist.PlanetPopulation.arange.unit
            e = statsFun.simpSample(targlist.PlanetPopulation.eccentricity, 20000, targlist.PlanetPopulation.erange.min(), targlist.PlanetPopulation.erange.max())
            rmax = a*(1. + e)
            inds = np.where(rmax < s)[0]
            no = len(inds)/20000.
            self.non = np.hstack((self.non, no))
        else:
            # update the accumulated value
            ind = np.where(self.obsinds == s_ind)[0]
            self.accumulated[ind] += comp0[s_ind]
            
        # find time delta
        tnext = nexttime - self.initialobstime
        # calculate dynamic completeness
        inds1 = np.where(tnext < self.tcomp)[0]
        inds2 = np.where(tnext >= self.tcomp)[0]
        dynamic = np.zeros(self.tcomp.shape)
        if inds1.shape[0] != 0:
            dynamic[inds1] = 8.51625*(1. - self.accumulated[inds1] - self.non[inds1])*tnext[inds1]/self.tcomp[inds1]
        if inds2.shape[0] != 0:
            dynamic[inds2] = (1. - self.accumulated[inds2] - self.non[inds2])
        
        comp0[self.obsinds] = dynamic
            
        return comp0
            
        
    def genC(self, Cpath, nplan, xedges, yedges, steps):
        """Gets completeness interpolant for initial completeness
        
        This function either loads a completeness .comp file based on specified
        Planet Population module or performs Monte Carlo simulations to get
        the 2D completeness values needed for interpolation.
        
        Args:
            Cpath (str):
                path to 2D completeness value array
            nplan (float):
                number of planets used in each simulation
            xedges (ndarray):
                1D numpy ndarray of x edge of 2d histogram (separation)
            yedges (ndarray):
                1D numpy ndarray of y edge of 2d histogram (dMag)
            steps (int):
                number of simulations to perform
                
        Returns:
            C (ndarray):
                2D numpy ndarray of completeness values
        
        """
        
        # if the 2D completeness array exists as a .comp file load it
        if os.path.exists(Cpath):
            C = pickle.load(open(Cpath, 'rb'))
            print 'Loaded completeness .comp file'
            h, xedges, yedges = self.hist(nplan, xedges, yedges)
        else:
            # run Monte Carlo simulation and pickle the resulting array
            print 'Beginning Monte Carlo completeness calculations'
            
            
            for i in xrange(steps):
                print 'iteration: %r / %r' % (i+1, steps)
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
        
            # initialize completeness
            C = np.zeros(H.shape)
            mult = np.tril(np.ones(H.shape))
            C[0,:] = np.dot(H[0,:], mult)
            for i in xrange(len(C) - 1):
                C[i+1,:] = C[i,:] + np.dot(H[i+1,:], mult)
            
            # store 2D completeness array as .comp file
            pickle.dump(C, open(Cpath, 'wb'))
            print 'Monte Carlo completeness calculations finished'
            print '2D completeness array stored in %r' % Cpath
        
        return C, xedges, yedges
        
    def hist(self, nplan, xedges, yedges):
        """Returns completeness histogram for Monte Carlo simulation
        
        This function uses the inherited Planet Population module.
        
        Args:
            nplan (float):
                number of planets used
            xedges (ndarray):
                1D numpy ndarray of x edge of 2d histogram (separation)
            yedges (ndarray):
                1D numpy ndarray of y edge of 2d histogram (dMag)
        
        Returns:
            h (ndarray):
                2D numpy ndarray containing completeness histogram
        
        """
        
        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to(u.AU).value, bins=1000, range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
        return h, xedges, yedges
        
    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation
        
        Args:
            nplan (int):
                number of planets
                
        Returns:
            s, dMag, a (Quantity, ndarray, Quantity):
                planet separations (in AU), difference in brightness
        
        """
        
        # sample uniform distribution of mean anomaly
        M = 2.*np.pi*np.random.rand(nplan)
        # sample semi-major axis        
        a = statsFun.simpSample(self.PlanetPopulation.semi_axis, nplan, self.PlanetPopulation.arange.min().value, self.PlanetPopulation.arange.max().value)*self.PlanetPopulation.arange.unit
        # sample other necessary orbital parameters
        if self.PlanetPopulation.erange.sum() == 0:
            # all circular orbits
            nu = M
            r = a
            e = 0.
            E = M
        else:
            # sample eccentricity            
            e = statsFun.simpSample(self.PlanetPopulation.eccentricity, nplan, self.PlanetPopulation.erange.min(), self.PlanetPopulation.erange.max())
            # Newton-Raphson to find E
            counter = 0
            err = 1
            E = M + e
            minus1 = np.where((M < 0) & (M > -np.pi))[0]
            E[minus1] -= 2.*e[minus1]
            minus2 = np.where(M > np.pi)[0]
            E[minus2] -= 2.*e[minus2]
            r = a
            while err > 1e-8 and counter < 1000:
                Enew = E + (M - E + e*np.sin(E))/(1. - e*np.cos(E))
                err = np.sum(np.abs(Enew - E))
                counter += 1
                E = Enew
                
            
        # RAAN, arument of perigee, inclination sampling
        Omega = np.radians(statsFun.simpSample(self.PlanetPopulation.RAAN, nplan, self.PlanetPopulation.Orange.min(), self.PlanetPopulation.Orange.max()))
        omega = np.radians(statsFun.simpSample(self.PlanetPopulation.arg_perigee, nplan, self.PlanetPopulation.wrange.min(), self.PlanetPopulation.wrange.max()))
        I = np.radians(statsFun.simpSample(self.PlanetPopulation.inclination, nplan, self.PlanetPopulation.Irange.min(), self.PlanetPopulation.Irange.max()))
        
        r1 = (r*(np.cos(E) - e)).to(u.AU).value
        r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
        r2 = (r*np.sin(E)*np.sqrt(1. -  e**2)).to(u.AU).value
        r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        
        a1 = np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(I)
        a2 = np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(I)
        a3 = np.sin(omega)*np.sin(I)
        
        A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))

        b1 = -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(I)
        b2 = -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(I)
        b3 = np.cos(omega)*np.sin(I)

        B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))

        rps = (A*r1 + B*r2)*u.AU
        
        r = np.sqrt(rps[:,0]**2 + rps[:,1]**2 + rps[:,2]**2)
               
        # compute Lambert phase
        beta = np.arccos(rps[:,2]/r).value
        Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi
        
        # sample albedo and planetary radius
        p = statsFun.simpSample(self.PlanetPopulation.albedo, nplan, self.PlanetPopulation.prange.min(), self.PlanetPopulation.prange.max())
        R = statsFun.simpSample(self.PlanetPopulation.radius, nplan, self.PlanetPopulation.Rrange.min().value, self.PlanetPopulation.Rrange.max().value)*self.PlanetPopulation.Rrange.unit
        
        # calculate dMag
        dMag = -2.5*np.log10((p*(R/r)**2*Phi).decompose().value)
        # calculate apparent separation
        s = np.sqrt(rps[:,0]**2 + rps[:,1]**2)
        
        return s, dMag