# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os
try:
    import cPickle as pickle
except:
    import pickle
import hashlib
from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag
import sys
from keplertools import fun
import matplotlib.pyplot as plt

# Python 3 compatibility:
if sys.version_info[0] > 2:
    xrange = range

class BrownCompleteness2(Completeness):
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
        updates (float nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, Nplanets=1e5, **specs):
        
        # bring in inherited Completeness prototype __init__ values
        Completeness.__init__(self, **specs)
        
        # Number of planets to sample
        self.Nplanets = int(Nplanets)
        
        # get path to completeness interpolant stored in a pickled .comp file
        self.filename = self.PlanetPopulation.__class__.__name__ + self.PlanetPhysicalModel.__class__.__name__
        
        # get path to dynamic completeness array in a pickled .dcomp file
        self.dfilename = self.PlanetPopulation.__class__.__name__ + \
                         self.PlanetPhysicalModel.__class__.__name__ +\
                         specs['modules']['OpticalSystem'] + \
                         specs['modules']['StarCatalog'] + \
                         specs['modules']['TargetList']
        
        # Create the values to be used later for dynamic completeness calculations
        self.a_vals, self.e_vals, self.p_vals, self.Rp_vals = self.PlanetPopulation.gen_plan_params(Nplanets)
        self.I_vals, self.O_vals, self.w_vals = self.PlanetPopulation.gen_angles(Nplanets)
        self.M0_vals = np.random.uniform(0, 2*np.pi, int(Nplanets))*u.rad
        self.mu = self.PlanetPopulation.mu
        
        # Dictionary that keeps track of simulated planets not eliminated from a star
        # self.vprint('Resting the dynamic completeness DICTIONARY ALSKJD:ALKSJJJJJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        self.dc_dict = {}
        
        atts = list(self.PlanetPopulation.__dict__)
        self.extstr = ''
        for att in sorted(atts, key=str.lower):
            if not callable(getattr(self.PlanetPopulation, att)) and att != 'PlanetPhysicalModel':
                self.extstr += '%s: ' % att + str(getattr(self.PlanetPopulation, att)) + ' '
        ext = hashlib.md5(self.extstr.encode("utf-8")).hexdigest()
        self.filename += ext

    def target_completeness(self, TL, calc_char_comp0=False):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList module):
                TargetList class object
            
        Returns:
            float ndarray: 
                Completeness values for each target star
        
        """
        
        
        
        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        if self.PlanetPopulation.constrainOrbits:
            xedges = np.linspace(0.0, self.PlanetPopulation.arange[1].to('AU').value, bins+1)
        else:
            xedges = np.linspace(0.0, self.PlanetPopulation.rrange[1].to('AU').value, bins+1)
        
        # yedges is array of delta magnitude values for interpolant
        ymin = -2.5*np.log10(float(self.PlanetPopulation.prange[1]*\
                (self.PlanetPopulation.Rprange[1]/self.PlanetPopulation.rrange[0])**2))
        ymax = -2.5*np.log10(float(self.PlanetPopulation.prange[0]*\
                (self.PlanetPopulation.Rprange[0]/self.PlanetPopulation.rrange[1])**2)*1e-11)
        yedges = np.linspace(ymin, ymax, bins+1)
        # number of planets for each Monte Carlo simulation
        nplan = int(np.min([1e6,self.Nplanets]))
        # number of simulations to perform (must be integer)
        steps = int(self.Nplanets/nplan)
        
        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(self.cachedir, self.filename+'.comp')
        Cpdf, xedges2, yedges2 = self.genC(Cpath, nplan, xedges, yedges, steps)

        xcent = 0.5*(xedges2[1:]+xedges2[:-1])
        ycent = 0.5*(yedges2[1:]+yedges2[:-1])
        xnew = np.hstack((0.0,xcent,self.PlanetPopulation.rrange[1].to('AU').value))
        ynew = np.hstack((ymin,ycent,ymax))
        Cpdf = np.pad(Cpdf,1,mode='constant')

        #save interpolant to object
        self.Cpdf = Cpdf
        self.EVPOCpdf = interpolate.RectBivariateSpline(xnew, ynew, Cpdf.T)
        self.EVPOC = np.vectorize(self.EVPOCpdf.integral, otypes=[np.float64])
        self.xnew = xnew
        self.ynew = ynew  
        
        # calculate separations based on IWA and OWA
        OS = TL.OpticalSystem
        if calc_char_comp0:
            mode = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))[0]
        else:
            mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
        IWA = mode['IWA']
        OWA = mode['OWA']
        smin = np.tan(IWA)*TL.dist
        if np.isinf(OWA):
            smax = np.array([xedges[-1]]*len(smin))*u.AU
        else:
            smax = np.tan(OWA)*TL.dist
            smax[smax>self.PlanetPopulation.rrange[1]] = self.PlanetPopulation.rrange[1]
        
        # limiting planet delta magnitude for completeness
        dMagMax = self.dMagLim
        
        comp0 = np.zeros(smin.shape)
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L>0, TL.L, 1e-10) #take care of zero/negative values
            smin = smin/np.sqrt(L)
            smax = smax/np.sqrt(L)
            dMagMax -= 2.5*np.log10(L)
            mask = (dMagMax>ymin) & (smin<self.PlanetPopulation.rrange[1])
            comp0[mask] = self.EVPOC(smin[mask].to('AU').value, \
                    smax[mask].to('AU').value, 0.0, dMagMax[mask])
        else:
            mask = smin<self.PlanetPopulation.rrange[1]
            comp0[mask] = self.EVPOC(smin[mask].to('AU').value, smax[mask].to('AU').value, 0.0, dMagMax)
        # remove small values
        comp0[comp0<1e-6] = 0.0
        # ensure that completeness is between 0 and 1
        comp0 = np.clip(comp0, 0., 1.)
        
        return comp0
    
        
    def dc_vals_at_time(self, TL, TK, sInd, t):
        """
        Method to calculate the dynamic completeness value at a time

        Args:
            TL (TargetList):
                TargetList class object
            TK (TimeKeeping):
                TimeKeeping class object
            sInd (integer):
                Integer index of the star of interest
            t (astropy quantity):
                Time to evaluate dynamic completeness at
        Returns:
            cij (float):
                Completeness value at the time
            full_visible_planets (ndarray):
                Array about which of the planets simulated for completeness
                are visible at the time
        """
        OS = TL.OpticalSystem
        
        if sInd not in self.dc_dict:
            # If dynamic completeness for the star then generate a list of 
            # true values as the potential planets, where each True represents
            # the fact that the planet hasn't been eliminated from search
            # The potential planets are those defined in instantiation of this
            # completeness module as a_vals, e_vals, etc
            
            potential_planets = np.ones(self.Nplanets, dtype=bool)
        else:
            # If it's been checked already then get the list containing the 
            # array with which planets have been eliminated already
            potential_planets = self.dc_dict[sInd]
        
        
        # Get indices for which potential planets so that they can be properly listed
        # as visible or not visible 
        planet_indices = np.linspace(0, self.Nplanets-1, self.Nplanets).astype(int)
        potential_planet_indices = planet_indices[potential_planets]
        
        # limiting planet delta magnitude for completeness
        dMagMax = self.dMagLim
        
        # get name for stored dynamic completeness updates array
        # inner and outer working angles for detection mode
        mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
        IWA = mode['IWA']
        OWA = mode['OWA']
        
        # Remove planets that have already been eliminated
        a_p = self.a_vals[potential_planets].to(u.m)
        e_p = self.e_vals[potential_planets]
        M0_p = self.M0_vals[potential_planets].to(u.rad)
        I_p = self.I_vals[potential_planets].to(u.rad)
        w_p = self.w_vals[potential_planets].to(u.rad)
        Rp_p = self.Rp_vals[potential_planets]
        p_p = self.p_vals[potential_planets]
        
        # Get distance to target star
        d = TL.dist[sInd]
        
        t_0 = TK.missionStart
        start = (TK.currentTimeAbs-t_0).value # The astropy Time module has units of days for this delta
        
        # Narrow down the potential planets based on which were visible at time
        # of detection (Should probably add a propagation method)
        M1 = np.sqrt(self.mu.to(u.m**3/(u.s**2))/a_p**3)*(start*u.d).to(u.s)*u.rad
        M2 = M0_p
        M  = (M1+M2)%(2*np.pi*u.rad)
        # Calculate the anomalies for each planet after the time period passes
        E = fun.eccanom(M.value, e_p)
        nu = fun.trueanom(E, e_p)
        # Find apparent separation
        theta = nu + w_p.value
        r = a_p.to(u.AU)*(1-e_p**2)/(1+e_p*np.cos(nu))
        s = (r.value/4)*np.sqrt(4*np.cos(2*I_p.value) + 4*np.cos(2*theta)-2*np.cos(2*I_p.value-2*theta) \
             - 2*np.cos(2*I_p.value+2*theta) + 12)
            
        # Find delta mag
        beta = np.arccos(-np.sin(I_p)*np.sin(theta))
        phi = (np.sin(beta.value) + (np.pi - beta.value)*np.cos(beta.value))/np.pi
        dMag = deltaMag(p_p, Rp_p.to(u.km), r.to(u.AU), phi)
    
        # Convert to min and max separation
        min_separation = IWA.to(u.arcsec).value*d.to(u.pc).value
        max_separation = OWA.to(u.arcsec).value*d.to(u.pc).value
        
        # Determine which are visible
        visible_planets = (s > min_separation) & (s < max_separation) & (dMag < dMagMax)
        # Create an array with all of the visible planets at their original location
        # in the list of planets
        visible_planet_indices = potential_planet_indices[visible_planets]
        full_visible_planets = np.zeros(self.Nplanets, dtype=bool)
        full_visible_planets[visible_planet_indices] = True
        
        # Calculate the completeness
        cij = np.sum(visible_planets)/float(np.sum(potential_planets))
        
        return cij, full_visible_planets
        
    def dc_dict_update(self, TL, TK, sInd, detected):
        """Updates the dictionary containing all of the potential planets for a
        star
        
        Args:
            TL (TargetList):
                TargetList class object
            TK (TimeKeeping)
                TimeKeeping class object
            sInd (integer):
                Integer index of the star of interest
            detected (list of integers):
                Values about whether the planets around the target were detected
        """
        
        
        if sInd not in self.dc_dict:
            # If dynamic completeness hasn't been checked for the star then generate a list of 
            # true values as the potential planets, where each True represents
            # the fact that the planet hasn't been eliminated from search
            # The potential planets are those defined in instantiation of this
            # completeness module as a_vals, e_vals, etc
            self.dc_dict[sInd] = np.ones(self.Nplanets, dtype=bool)
        
        potential_planets = self.dc_dict[sInd]
        t = TK.currentTimeNorm
        _, visible_planets = self.dc_vals_at_time(TL, TK, sInd, t)
        
        
        
        if 1 in detected:
            # If a planet is observed then keep all potential planets that are
            # visible at the time of the observation and if not remove the others
            potential_planets = np.logical_and(potential_planets, visible_planets)
            # Update the dictionary for later visits
            self.dc_dict[sInd] = potential_planets
        else:
            # If a planet is not observed then eliminate all visible planets from 
            # the list of potential planets
            potential_planets = np.logical_and(potential_planets, np.logical_not(visible_planets))
            self.dc_dict[sInd] = potential_planets 
    
    
    def completeness_update(self, TL, TK, sInds, visits, dt):
        """Updates completeness value for stars previously observed by selecting
        the appropriate value from the updates array
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            visits (integer array):
                Number of visits for each star
            dt (astropy Quantity array):
                Time since previous observation
        
        Returns:
            float ndarray:
                Completeness values for each star
        
        """
        # if visited more than five times, return 5th stored dynamic 
        # completeness value
        # visits[visits > 4] = 4
        # dcomp = self.updates[sInds, visits]
        
        for sInd in sInds:
            
            if sInd not in self.dc_dict:
                # if the star has not been visited then just use it's comp0 val
                dcomp = TL.comp0[sInd]
            else:
                # Calculate it's dynamic completeness
                dcomp, _ = self.dc_vals_at_time(TL, TK, sInd, dt)
                
                
        return dcomp

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
            float ndarray:
                2D numpy ndarray containing completeness probability density values
        
        """
        
        # if the 2D completeness pdf array exists as a .comp file load it
        if os.path.exists(Cpath):
            self.vprint('Loading cached completeness file from "%s".' % Cpath)
            try:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff)
            except UnicodeDecodeError:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff,encoding='latin1')
            self.vprint('Completeness loaded from cache.')
        else:
            # run Monte Carlo simulation and pickle the resulting array
            self.vprint('Cached completeness file not found at "%s".' % Cpath)
            self.vprint('Beginning Monte Carlo completeness calculations.')
            
            t0, t1 = None, None # keep track of per-iteration time
            for i in xrange(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = '' # no message
                else:
                    delta_t_msg = '[%.3f s/iteration]' % (t1 - t0)
                self.vprint('Completeness iteration: %5d / %5d %s' % (i+1, steps, delta_t_msg))
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
            
            H = H/(self.Nplanets*(xedges[1]-xedges[0])*(yedges[1]-yedges[0]))
                        
            # store 2D completeness pdf array as .comp file
            with open(Cpath, 'wb') as ff:
                pickle.dump(H, ff)
            self.vprint('Monte Carlo completeness calculations finished')
            self.vprint('2D completeness array stored in %r' % Cpath)
        
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
            float ndarray:
                2D numpy ndarray containing completeness frequencies
        
        """
        
        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=1000,
                range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
        return h, xedges, yedges

    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation
        
        Args:
            nplan (integer):
                Number of planets
                
        Returns:
            tuple:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness
        
        """
        
        PPop = self.PlanetPopulation
        
        nplan = int(nplan)
        
        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.0*np.pi,size=nplan)
        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(nplan)
        # check if circular orbits
        if np.sum(PPop.erange) == 0:
            r = a
            e = 0.0
            E = M
        else:
            E = eccanom(M,e)
            # orbital radius
            r = a*(1.0-e*np.cos(E))

        beta = np.arccos(1.0-2.0*np.random.uniform(size=nplan))*u.rad
        s = r*np.sin(beta)
        # phase function
        Phi = self.PlanetPhysicalModel.calc_Phi(beta)
        # calculate dMag
        dMag = deltaMag(p,Rp,r,Phi)
        
        return s, dMag

    def comp_per_intTime(self, intTimes, TL, TK, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None):
        """Calculates completeness for integration time
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            TK (TimeKeeping module):
                TimeKeeping class object
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
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
                
        Returns:
            flat ndarray:
                Completeness values
        
        """
        intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = self.comps_input_reshape(intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp)
        
        comp = np.zeros(len(sInds))
        for i in range(len(sInds)):
            sInd = sInds[i]
            if sInd not in self.dc_dict:
                # if the star has not been visited then just use it's comp0 val
                comp[i] = TL.comp0[sInd]
            else:
                # Calculate it's dynamic completeness
                
                comp[i], _ = self.dc_vals_at_time(TL, TK, sInd, intTimes[i])
                # print('Considering star: ' + str(sInd) + ' with completeness: ' + str(round(comp[i],3)))
        return comp
    
    def comp_calc(self, smin, smax, dMag):
        """Calculates completeness for given minimum and maximum separations
        and dMag
        
        Note: this method assumes scaling orbits when scaleOrbits == True has
        already occurred for smin, smax, dMag inputs
        
        Args:
            smin (float ndarray):
                Minimum separation(s) in AU
            smax (float ndarray):
                Maximum separation(s) in AU
            dMag (float ndarray):
                Difference in brightness magnitude
        
        Returns:
            float ndarray:
                Completeness values
        
        """
        
        comp = self.EVPOC(smin, smax, 0., dMag)
        # remove small values
        comp[comp<1e-6] = 0.
        
        return comp

    def dcomp_dt(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None):
        """Calculates derivative of completeness with respect to integration time
        
        Args:
            intTimes (astropy Quantity array):
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
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)                
                
        Returns:
            astropy Quantity array:
                Derivative of completeness with respect to integration time (units 1/time)
        
        """
        intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = self.comps_input_reshape(intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp)
        
        ddMag = TL.OpticalSystem.ddMag_dt(intTimes, TL, sInds, fZ, fEZ, WA, mode).reshape((len(intTimes),))
        dcomp = self.calc_fdmag(dMag, smin, smax)
        mask = smin>self.PlanetPopulation.rrange[1].to('AU').value
        dcomp[mask] = 0.
        
        return dcomp*ddMag
    
    def comps_input_reshape(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None):
        """
        Reshapes inputs for comp_per_intTime and dcomp_dt as necessary
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface bright ness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s (optional)
        
        Returns:
            tuple: 
            intTimes (astropy Quantity array):
                Integration times
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            smin (ndarray):
                Minimum projected separations in AU
            smax (ndarray):
                Maximum projected separations in AU
            dMag (ndarray):
                Difference in brightness magnitude
        """
        
        # cast inputs to arrays and check
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) in [1, len(sInds)], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [1, len(sInds)], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [1, len(sInds)], "fEZ must be constant or have same length as sInds"
        assert len(WA) in [1, len(sInds)], "WA must be constant or have same length as sInds"
        # make constants arrays of same length as sInds if len(sInds) != 1
        if len(sInds) != 1:
            if len(intTimes) == 1:
                intTimes = np.repeat(intTimes.value, len(sInds))*intTimes.unit
            if len(fZ) == 1:
                fZ = np.repeat(fZ.value, len(sInds))*fZ.unit
            if len(fEZ) == 1:
                fEZ = np.repeat(fEZ.value, len(sInds))*fEZ.unit
            if len(WA) == 1:
                WA = np.repeat(WA.value, len(sInds))*WA.unit

        dMag = TL.OpticalSystem.calc_dMag_per_intTime(intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp).reshape((len(intTimes),))
        # calculate separations based on IWA and OWA
        IWA = mode['IWA']
        OWA = mode['OWA']
        smin = (np.tan(IWA)*TL.dist[sInds]).to('AU').value
        if np.isinf(OWA):
            smax = np.array([self.PlanetPopulation.rrange[1].to('AU').value]*len(smin))
        else:
            smax = (np.tan(OWA)*TL.dist[sInds]).to('AU').value
            smax[smax>self.PlanetPopulation.rrange[1].to('AU').value] = self.PlanetPopulation.rrange[1].to('AU').value
        smin[smin>smax] = smax[smin>smax]
        
        # take care of scaleOrbits == True
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L[sInds]>0, TL.L[sInds], 1e-10) #take care of zero/negative values
            smin = smin/np.sqrt(L)
            smax = smax/np.sqrt(L)
            dMag -= 2.5*np.log10(L)
        
        return intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag            
    
    def calc_fdmag(self, dMag, smin, smax):
        """Calculates probability density of dMag by integrating over projected
        separation
        
        Args:
            dMag (float ndarray):
                Planet delta magnitude(s)
            smin (float ndarray):
                Value of minimum projected separation (AU) from instrument
            smax (float ndarray):
                Value of maximum projected separation (AU) from instrument
        
        Returns:
            float:
                Value of probability density
        
        """
        
        f = np.zeros(len(smin))
        for k, dm in enumerate(dMag):
            f[k] = interpolate.InterpolatedUnivariateSpline(self.xnew,self.EVPOCpdf(self.xnew,dm),ext=1).integral(smin[k],smax[k])
            
        return f
