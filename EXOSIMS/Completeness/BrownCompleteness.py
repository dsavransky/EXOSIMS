# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os, inspect
try:
    import cPickle as pickle
except:
    import pickle
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
        minComp (float): 
            minimum completeness level for detection
        Nplanets (float):
            number of planets for initial completeness Monte Carlo simulation
        classpath (str):
            path on disk to Brown Completeness
        filename (str):
            name of file where completeness interpolant is stored
        visits (list):
            number of observations corresponding to each star in the target list
            (initialized in gen_update)
        updates (list):
            completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, Nplanets=1e8, **specs):
        
        # bring in inherited Completeness prototype __init__ values
        Completeness.__init__(self, **specs)

        # Number of planets to sample
        self.Nplanets = Nplanets
        
        # get path to completeness interpolant stored in a pickled .comp file
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        self.filename = specs['modules']['PlanetPopulation']

    def target_completeness(self, targlist):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            targlist (TargetList): 
                TargetList class object
            
        Returns:
            comp0 (ndarray): 
                1D numpy array of completeness values for each target star
        
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
        
        # path to 2D completeness array for interpolation
        Cpath = os.path.join(self.classpath, self.filename+'.comp')
        C, xedges2, yedges2 = self.genC(Cpath, nplan, xedges, yedges, steps)
        C = C/self.Nplanets
        EVPOC = interpolate.RectBivariateSpline(xedges, yedges, C.T)
        EVPOC = np.vectorize(EVPOC)
            
        # calculate separations based on IWA
        s = np.tan(targlist.OpticalSystem.IWA)*targlist.dist
        # calculate dMags based on limiting dMag
        dMag = np.array([targlist.OpticalSystem.dMagLim]*targlist.nStars)
        if self.PlanetPopulation.scaleOrbits:
            s = s/np.sqrt(targlist.L)
            dMag = dMag - 2.5*np.log10(targlist.L)
            
        comp0 = EVPOC(s.to('AU').value, dMag)
            
        return comp0
        
    def gen_update(self, targlist):
        """Generates dynamic completeness values for multiple visits of each 
        star in the target list
        
        Args:
            targlist (TargetList):
                TargetList module
        
        """
        
        print 'Beginning completeness update calculations'
        self.visits = [0]*targlist.nStars
        self.updates = []
        # constants needed
        Msun = const.M_sun
        # number of planets to simulate
        nplan = int(2e4)
        # normalization time
        dt = 1e9*u.day
        # sample quantities which do not change in time
        a = self.PlanetPopulation.gen_sma(nplan) # AU
        e = self.PlanetPopulation.gen_eccen(nplan)
        Omega = self.PlanetPopulation.gen_O(nplan) # deg
        omega = self.PlanetPopulation.gen_w(nplan) # deg
        I = self.PlanetPopulation.gen_I(nplan) # deg
        p = self.PlanetPopulation.gen_albedo(nplan)
        Rp = self.PlanetPopulation.gen_radius(nplan) # m
        Mp = self.PlanetPopulation.gen_mass(nplan) # kg
        rmax = a*(1.+e)
        rmin = a*(1.-e)
        # sample quantity which will be updated
        M = np.random.uniform(high=2.*np.pi,size=nplan)
        # population values
        smin = (np.tan(targlist.OpticalSystem.IWA)*targlist.dist).to('AU')
        if np.isfinite(targlist.OpticalSystem.OWA):
            smax = (np.tan(targlist.OpticalSystem.OWA)*targlist.dist).to('AU')
        else:
            smax = np.array([np.max(self.PlanetPopulation.arange)*(1.+np.max(self.PlanetPopulation.erange))]*targlist.nStars)
        # fill dynamic completeness values
        for sInd in xrange(targlist.nStars):
            # remove rmax < smin and rmin > smax
            inside = np.where(rmax > smin[sInd])[0]
            outside = np.where(rmin < smax[sInd])[0]
            new_plan_inds = np.intersect1d(inside,outside)
            dynamic = []
            # calculate for 5 successive observations
            for num in xrange(5):
                # find Eccentric anomaly
                E = eccanom(M[new_plan_inds],e[new_plan_inds])
                r = a[new_plan_inds]*(1.-e[new_plan_inds]*np.cos(E))
                r1 = r*(np.cos(E) - e[new_plan_inds])
                r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
                r2 = (r*np.sin(E)*np.sqrt(1. -  e[new_plan_inds]**2))
                r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        
                a1 = np.cos(Omega[new_plan_inds])*np.cos(omega[new_plan_inds]) - np.sin(Omega[new_plan_inds])*np.sin(omega[new_plan_inds])*np.cos(I[new_plan_inds])
                a2 = np.sin(Omega[new_plan_inds])*np.cos(omega[new_plan_inds]) + np.cos(Omega[new_plan_inds])*np.sin(omega[new_plan_inds])*np.cos(I[new_plan_inds])
                a3 = np.sin(omega[new_plan_inds])*np.sin(I[new_plan_inds])
                A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))

                b1 = -np.cos(Omega[new_plan_inds])*np.sin(omega[new_plan_inds]) - np.sin(Omega[new_plan_inds])*np.cos(omega[new_plan_inds])*np.cos(I[new_plan_inds])
                b2 = -np.sin(Omega[new_plan_inds])*np.sin(omega[new_plan_inds]) + np.cos(Omega[new_plan_inds])*np.cos(omega[new_plan_inds])*np.cos(I[new_plan_inds])
                b3 = np.cos(omega[new_plan_inds])*np.sin(I[new_plan_inds])
                B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))

                # planet position, planet-star distance, apparent separation
                r = (A*r1 + B*r2)*u.AU # position vector
                d = np.sqrt(np.sum(r**2, axis=1)) # planet-star distance
                s = np.sqrt(np.sum(r[:,0:2]**2, axis=1)) # apparent separation
                beta = np.arccos(r[:,2]/d) # phase angle
                Phi = self.PlanetPhysicalModel.calc_Phi(beta) # phase function
                dMag = deltaMag(p[new_plan_inds],Rp[new_plan_inds],d,Phi) # difference in magnitude

                toremoves = np.where((s > smin[sInd]) & (s < smax[sInd]))[0]
                toremovedmag = np.where(dMag < targlist.OpticalSystem.dMagLim)[0]
                toremove = np.intersect1d(toremoves, toremovedmag)
                
                new_plan_inds = np.delete(new_plan_inds, toremove)    
                if num == 0:
                    dynamic.append(targlist.comp0[sInd])
                else:
                    dynamic.append(len(toremove)/nplan)

                # update M
                mu = const.G*(Msun+Mp[new_plan_inds])
                n = np.sqrt(mu/a[new_plan_inds]**3)
                M[new_plan_inds] = (M[new_plan_inds] + n*dt)/(2*np.pi) % 1 * 2.*np.pi

            self.updates.append(dynamic)
            if (sInd+1) % 50 == 0:
                print 'stars: %r / %r' % (sInd+1,targlist.nStars)
        print 'Completeness update calculations finished'
        
    def completeness_update(self, sInd, targlist, obsbegin, obsend, nexttime):
        """Updates completeness value for stars previously observed
        
        Args:
            sInd (int):
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
        
        self.visits[sInd] += 1
        
        if self.visits[sInd] > len(self.updates[sInd])-1:
            targlist.comp0[sInd] = self.updates[sInd][-1]
        else:
            targlist.comp0[sInd] = self.updates[sInd][self.visits[sInd]]
            
        
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
            print 'Loading completeness file from %s'%Cpath
            C = pickle.load(open(Cpath, 'rb'))
            print 'Completeness Loaded'
            #h, xedges, yedges = self.hist(nplan, xedges, yedges)
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
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=1000, \
                range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
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
        nplan = int(nplan)

        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.*np.pi,size=nplan)
        # sample semi-major axis        
        a = self.PlanetPopulation.gen_sma(nplan).to('AU').value

        # sample other necessary orbital parameters
        if np.sum(self.PlanetPopulation.erange) == 0:
            # all circular orbits
            r = a
            e = 0.
            E = M
        else:
            # sample eccentricity
            if self.PlanetPopulation.constrainOrbits:
                e = self.PlanetPopulation.gen_eccen_from_sma(nplan,a*u.AU)
            else:
                e = self.PlanetPopulation.gen_eccen(nplan)   
            # Newton-Raphson to find E
            E = eccanom(M,e)
            # orbital radius
            r = a*(1-e*np.cos(E))

        # orbit angle sampling
        Omega = self.PlanetPopulation.gen_O(nplan).to('rad').value
        omega = self.PlanetPopulation.gen_w(nplan).to('rad').value
        I = self.PlanetPopulation.gen_I(nplan).to('rad').value
        
        r1 = (r*(np.cos(E) - e))
        r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
        r2 = (r*np.sin(E)*np.sqrt(1. -  e**2))
        r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        
        a1 = np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(I)
        a2 = np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(I)
        a3 = np.sin(omega)*np.sin(I)
        A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))

        b1 = -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(I)
        b2 = -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(I)
        b3 = np.cos(omega)*np.sin(I)
        B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))

        # planet position, planet-star distance, apparent separation
        r = (A*r1 + B*r2)*u.AU
        d = np.sqrt(np.sum(r**2, axis=1))
        s = np.sqrt(np.sum(r[:,0:2]**2, axis=1))

        # sample albedo, planetary radius, phase function
        p = self.PlanetPopulation.gen_albedo(nplan)
        Rp = self.PlanetPopulation.gen_radius(nplan)

        beta = np.arccos(r[:,2]/d)

        Phi = self.PlanetPhysicalModel.calc_Phi(beta)

        # calculate dMag
        dMag = deltaMag(p,Rp,d,Phi)

        return s, dMag
