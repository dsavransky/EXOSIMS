# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os, inspect
try:
    import cPickle as pickle
except:
    import pickle
import hashlib
from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag

class BrownCompleteness(Completeness):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        Nplanets (integer):
            Number of planets for initial completeness Monte Carlo simulation
        classpath (string):
            Path on disk to Brown Completeness
        filename (string):
            Name of file where completeness interpolant is stored
        visits (integer ndarray):
            Number of observations corresponding to each star in the target list
            (initialized in gen_update)
        updates (float nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, Nplanets=1e8, **specs):
        
        # bring in inherited Completeness prototype __init__ values
        Completeness.__init__(self, **specs)
        
        # Number of planets to sample
        self.Nplanets = int(Nplanets)
        
        # get path to completeness interpolant stored in a pickled .comp file
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        self.filename = specs['modules']['PlanetPopulation']
        atts = ['arange','erange','prange','Rprange','Mprange','scaleOrbits','constrainOrbits']
        
        extstr = ''
        for att in atts:
            extstr += '%s: ' % att + str(getattr(self.PlanetPopulation, att)) + ' '
        ext = hashlib.md5(extstr).hexdigest()
        self.filename += ext

    def target_completeness(self, TL):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList module):
                TargetList class object
            
        Returns:
            comp0 (float ndarray): 
                Completeness values for each target star
        
        """
        
        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        xedges = np.linspace(0., self.PlanetPopulation.rrange[1].value, bins)*\
                self.PlanetPopulation.arange.unit
        xedges = xedges.to('AU').value
        
        # yedges is array of delta magnitude values for interpolant
        ymin = np.round((-2.5*np.log10(self.PlanetPopulation.prange[1]*\
                (self.PlanetPopulation.Rprange[1]/(self.PlanetPopulation.rrange[0]))\
                .decompose().value**2)))
        ymax = np.round((-2.5*np.log10(self.PlanetPopulation.prange[0]*\
                (self.PlanetPopulation.Rprange[0]/(self.PlanetPopulation.rrange[1]))\
                .decompose().value**2*1e-11)))
        yedges = np.linspace(ymin, ymax, bins)
        
        # number of planets for each Monte Carlo simulation
        nplan = int(np.min([1e6,self.Nplanets]))
        # number of simulations to perform (must be integer)
        steps = int(self.Nplanets/nplan)
        
        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(self.classpath, self.filename+'.comp')
        Cpdf, xedges2, yedges2 = self.genC(Cpath, nplan, xedges, yedges, steps)
        
        EVPOCpdf = interpolate.RectBivariateSpline(xedges, yedges, Cpdf.T)
        EVPOC = np.vectorize(EVPOCpdf.integral)
            
        # calculate separations based on IWA
        OS = TL.OpticalSystem
        smin = np.tan(OS.IWA)*TL.dist
        if np.isinf(OS.OWA):
            smax = xedges[-1]*u.AU
        else:
            smax = np.tan(OS.OWA)*TL.dist
        
        # calculate dMags based on limiting dMag
        dMagmax = OS.dMagLim #np.array([OS.dMagLim]*TL.nStars)
        dMagmin = ymin
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L>0, TL.L, 1e-10) #take care of zero/negative values
            smin = smin/np.sqrt(L)
            smax = smax/np.sqrt(L)
            dMagmin -= 2.5*np.log10(L)
            dMagmax -= 2.5*np.log10(L)
            
        comp0 = EVPOC(smin.to('AU').value, smax.to('AU').value, dMagmin, dMagmax)
        
        return comp0

    def gen_update(self, TL):
        """Generates dynamic completeness values for multiple visits of each 
        star in the target list
        
        Args:
            TL (TargetList module):
                TargetList class object
        
        """
        
        OS = TL.OpticalSystem
        PPop = self.PlanetPopulation
        
        print 'Beginning completeness update calculations'
        # initialize number of visits
        self.visits = np.array([0]*TL.nStars)
        # dynamic completeness values: rows are stars, columns are number of visits
        self.updates = np.zeros((TL.nStars, 5))
        # number of planets to simulate
        nplan = int(2e4)
        # normalization time
        dt = 1e9*u.day
        # sample quantities which do not change in time
        a = PPop.gen_sma(nplan) # AU
        e = PPop.gen_eccen(nplan)
        I = PPop.gen_I(nplan) # deg
        O = PPop.gen_O(nplan) # deg
        w = PPop.gen_w(nplan) # deg
        p = PPop.gen_albedo(nplan)
        Rp = PPop.gen_radius(nplan) # km
        Mp = PPop.gen_mass(nplan) # kg
        rmax = a*(1.+e)
        # sample quantity which will be updated
        M = np.random.uniform(high=2.*np.pi,size=nplan)
        newM = np.zeros((nplan,))
        # population values
        smin = (np.tan(OS.IWA)*TL.dist).to('AU')
        if np.isfinite(OS.OWA):
            smax = (np.tan(OS.OWA)*TL.dist).to('AU')
        else:
            smax = np.array([np.max(PPop.arange.to('AU').value)*\
                    (1.+np.max(PPop.erange))]*TL.nStars)*u.AU
        # fill dynamic completeness values
        for sInd in xrange(TL.nStars):
            Mstar = TL.MsTrue[sInd]*const.M_sun
            # remove rmax < smin 
            pInds = np.where(rmax > smin[sInd])[0]
            # calculate for 5 successive observations
            for num in xrange(5):
                if num == 0:
                    self.updates[sInd, num] = TL.comp0[sInd]
                if not pInds.any():
                    break
                # find Eccentric anomaly
                if num == 0:
                    E = eccanom(M[pInds],e[pInds])
                    newM[pInds] = M[pInds]
                else:
                    E = eccanom(newM[pInds],e[pInds])
                
                r1 = a[pInds]*(np.cos(E) - e[pInds])
                r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
                r2 = (a[pInds]*np.sin(E)*np.sqrt(1. -  e[pInds]**2))
                r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
                
                a1 = np.cos(O[pInds])*np.cos(w[pInds]) - np.sin(O[pInds])*np.sin(w[pInds])*np.cos(I[pInds])
                a2 = np.sin(O[pInds])*np.cos(w[pInds]) + np.cos(O[pInds])*np.sin(w[pInds])*np.cos(I[pInds])
                a3 = np.sin(w[pInds])*np.sin(I[pInds])
                A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))
                
                b1 = -np.cos(O[pInds])*np.sin(w[pInds]) - np.sin(O[pInds])*np.cos(w[pInds])*np.cos(I[pInds])
                b2 = -np.sin(O[pInds])*np.sin(w[pInds]) + np.cos(O[pInds])*np.cos(w[pInds])*np.cos(I[pInds])
                b3 = np.cos(w[pInds])*np.sin(I[pInds])
                B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))
                
                # planet position, planet-star distance, apparent separation
                r = (A*r1 + B*r2)*u.AU # position vector
                d = np.sqrt(np.sum(r**2, axis=1)) # planet-star distance
                s = np.sqrt(np.sum(r[:,0:2]**2, axis=1)) # apparent separation
                beta = np.arccos(r[:,2]/d) # phase angle
                Phi = self.PlanetPhysicalModel.calc_Phi(beta) # phase function
                dMag = deltaMag(p[pInds],Rp[pInds],d,Phi) # difference in magnitude
                
                toremoves = np.where((s > smin[sInd]) & (s < smax[sInd]))[0]
                toremovedmag = np.where(dMag < OS.dMagLim)[0]
                toremove = np.intersect1d(toremoves, toremovedmag)
                
                pInds = np.delete(pInds, toremove)
                
                if num == 0:
                    self.updates[sInd, num] = TL.comp0[sInd]
                else:
                    self.updates[sInd, num] = float(len(toremove))/nplan
                
                # update M
                mu = const.G*(Mstar+Mp[pInds])
                n = np.sqrt(mu/a[pInds]**3)
                newM[pInds] = (newM[pInds] + n*dt)/(2*np.pi) % 1 * 2.*np.pi
                            
            if (sInd+1) % 50 == 0:
                print 'stars: %r / %r' % (sInd+1,TL.nStars)
        
        print 'Completeness update calculations finished'

    def completeness_update(self, TL, sInds, dt):
        """Updates completeness value for stars previously observed by selecting
        the appropriate value from the updates array
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            dt (astropy Quantity):
                Time since initial completeness
        
        Returns:
            comp0 (float ndarray):
                Completeness values for each star
        
        """
        
        # number of visits for each star
        cols = self.visits[sInds]
        # if visited more than five times, return 5th stored dynamic 
        # completeness value
        cols[cols>4] = 4
        # return value from the updates array
        
        return self.updates[sInds, cols]

    def genC(self, Cpath, nplan, xedges, yedges, steps):
        """Gets completeness interpolant for initial completeness
        
        This function either loads a completeness .comp file based on specified
        Planet Population module or performs Monte Carlo simulations to get
        the 2D completeness values needed for interpolation.
        
        Args:
            Cpath (string):
                path to 2D completeness value array
            nplan (float):
                number of planets used in each simulation
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
            steps (integer):
                number of simulations to perform
                
        Returns:
            H (float ndarray):
                2D numpy ndarray containing completeness probability density values
        
        """
        
        # if the 2D completeness pdf array exists as a .comp file load it
        if os.path.exists(Cpath):
            print 'Loading cached completeness file from "%s".' % Cpath
            H = pickle.load(open(Cpath, 'rb'))
            print 'Completeness loaded from cache.'
            #h, xedges, yedges = self.hist(nplan, xedges, yedges)
        else:
            # run Monte Carlo simulation and pickle the resulting array
            print 'Cached completeness file not found at "%s".' % Cpath
            print 'Beginning Monte Carlo completeness calculations.'
            
            t0, t1 = None, None # keep track of per-iteration time
            for i in xrange(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = '' # no message
                else:
                    delta_t_msg = '[%.3f s/iteration]' % (t1 - t0)
                print 'Completeness iteration: %5d / %5d %s' % (i+1, steps, delta_t_msg)
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
            
            H = H/(self.Nplanets*(xedges[1]-xedges[0])*(yedges[1]-yedges[0]))            
                        
            # store 2D completeness pdf array as .comp file
            pickle.dump(H, open(Cpath, 'wb'))
            print 'Monte Carlo completeness calculations finished'
            print '2D completeness array stored in %r' % Cpath
        
        return H, xedges, yedges

    def hist(self, nplan, xedges, yedges):
        """Returns completeness histogram for Monte Carlo simulation
        
        This function uses the inherited Planet Population module.
        
        Args:
            nplan (float):
                number of planets used
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
        
        Returns:
            h (ndarray):
                2D numpy ndarray containing completeness histogram
        
        """
        
        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=1000, \
                range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
        return h, xedges, yedges

    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation
        
        Args:
            nplan (integer):
                Number of planets
                
        Returns:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness
        
        """
        
        PPop = self.PlanetPopulation
        
        nplan = int(nplan)
        
        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.*np.pi,size=nplan)
        # sample semi-major axis
        a = PPop.gen_sma(nplan).to('AU').value
        
        # sample other necessary orbital parameters
        if np.sum(PPop.erange) == 0:
            # all circular orbits
            r = a
            e = 0.
            E = M
        else:
            # sample eccentricity
            if PPop.constrainOrbits:
                e = PPop.gen_eccen_from_sma(nplan,a*u.AU)
            else:
                e = PPop.gen_eccen(nplan)   
            # Newton-Raphson to find E
            E = eccanom(M,e)
            # orbital radius
            r = a*(1-e*np.cos(E))
        
        # orbit angle sampling
        O = PPop.gen_O(nplan).to('rad').value
        w = PPop.gen_w(nplan).to('rad').value
        I = PPop.gen_I(nplan).to('rad').value
        
        r1 = a*(np.cos(E) - e)
        r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
        r2 = a*np.sin(E)*np.sqrt(1. -  e**2)
        r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        
        a1 = np.cos(O)*np.cos(w) - np.sin(O)*np.sin(w)*np.cos(I)
        a2 = np.sin(O)*np.cos(w) + np.cos(O)*np.sin(w)*np.cos(I)
        a3 = np.sin(w)*np.sin(I)
        A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))
        
        b1 = -np.cos(O)*np.sin(w) - np.sin(O)*np.cos(w)*np.cos(I)
        b2 = -np.sin(O)*np.sin(w) + np.cos(O)*np.cos(w)*np.cos(I)
        b3 = np.cos(w)*np.sin(I)
        B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))
        
        # planet position, planet-star distance, apparent separation
        r = (A*r1 + B*r2)*u.AU
        d = np.sqrt(np.sum(r**2, axis=1))
        s = np.sqrt(np.sum(r[:,0:2]**2, axis=1))
        
        # sample albedo, planetary radius, phase function
        p = PPop.gen_albedo(nplan)
        Rp = PPop.gen_radius(nplan)
        beta = np.arccos(r[:,2]/d)
        Phi = self.PlanetPhysicalModel.calc_Phi(beta)
        
        # calculate dMag
        dMag = deltaMag(p,Rp,d,Phi)
        
        return s, dMag
