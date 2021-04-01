# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os
from EXOSIMS.Completeness.SubtypeCompleteness import SubtypeCompleteness
import sys
from exodetbox.projectedEllipse import *

class IntegrationTimeAdjustedCompleteness(SubtypeCompleteness):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        specs: 
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
        
        print('Num Planets BEFORE SubtypeComp declaration: ' + str(Nplanets))
        # bring in inherited SubtypeCompleteness prototype __init__ values
        SubtypeCompleteness.__init__(self, **specs)
        print('Num Planets AFTER SubtypeComp declaration: ' + str(Nplanets))
        #Note: This calls target completeness which calculates TL.comp0, a term used for filtering targets based on low completeness values
        #This executes with 10^8 planets, the default for SubtypeCompleteness. self.Nplanets is updated later here
        
        # Number of planets to sample
        self.Nplanets = int(Nplanets)
       
        # get path to completeness interpolant stored in a pickled .comp file
        self.filename = self.PlanetPopulation.__class__.__name__ + self.PlanetPhysicalModel.__class__.__name__ + self.__class__.__name__ + str(self.Nplanets) + self.PlanetPhysicalModel.whichPlanetPhaseFunction

        # get path to dynamic completeness array in a pickled .dcomp file
        self.dfilename = self.PlanetPopulation.__class__.__name__ + \
                         self.PlanetPhysicalModel.__class__.__name__ +\
                         specs['modules']['OpticalSystem'] + \
                         specs['modules']['StarCatalog'] + \
                         specs['modules']['TargetList']
        atts = list(self.PlanetPopulation.__dict__)
        self.extstr = ''
        for att in sorted(atts, key=str.lower):
            if not callable(getattr(self.PlanetPopulation, att)) and att != 'PlanetPhysicalModel':
                self.extstr += '%s: ' % att + str(getattr(self.PlanetPopulation, att)) + ' '
        ext = hashlib.md5(self.extstr.encode("utf-8")).hexdigest()
        self.filename += ext
        self.filename.replace(" ","") #Remove spaces from string (in the case of prototype use)


        #### STANDARD PLANET POPLATION GENERATION
        #Calculate and create a set of planets
        PPop = self.PlanetPopulation
        self.inc, self.W, self.w = PPop.gen_angles(Nplanets,None)
        self.sma, self.e, self.p, self.Rp = PPop.gen_plan_params(Nplanets)
        self.inc, self.W, self.w = self.inc.to('rad').value, self.W.to('rad').value, self.w.to('rad').value
        self.sma = self.sma.to('AU').value

        #Pass in as TL object?
        #starMass #set as default of 1 M_sun
        starMass = const.M_sun
        plotBool = False #need to remove eventually
        self.periods = (2.*np.pi*np.sqrt((self.sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value#need to pass in

    def comp_calc(self, smin, smax, dMag, subpop=-2, tmax=0.,starMass=const.M_sun, IACbool=False):
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
            subpop (int):
                planet subtype to use for calculation of comp0
                -2 - planet population
                -1 - earthLike population
                (i,j) - kopparapu planet subtypes
            tmax (float):
                the integration time of the observation
            starMass (float):
                star mass in units of M_sun
            IACbool (boolean):
                a boolean indicating whether to use integration timeadjusted completeness or normal brown completeness
                if False, tmax does nothing
        Returns:
            ndarray:
                comp, SubtypeCompleteness Completeness values (brown's method mixed with classification) or integration time adjusted completeness totalCompleteness_maxIntTimeCorrected
        
        """
        
        if IACbool:
            print(len(self.sma))
            sma = self.sma
            e = self.e
            W = self.W
            w = self.w
            inc = self.inc
            p = self.p
            Rp = self.Rp

            #Pass in as TL object?
            #starMass #set as default of 1 M_sun
            plotBool = False #need to remove eventually
            periods = self.periods #need to pass in

            #inputs
            s_inner = smin
            s_outer = smax
            dmag_upper = dMag
            #input tmax
            totalCompleteness_maxIntTimeCorrected = integrationTimeAdjustedCompletness(sma,e,W,w,inc,p,Rp,starMass,plotBool,periods, s_inner, s_outer, dmag_upper, tmax)

            return totalCompleteness_maxIntTimeCorrected
        else:
            if subpop == -2:
                comp = self.EVPOC_pop(smin, smax, 0., dMag)
            elif subpop == -1:
                comp = self.EVPOC_earthlike(smin, smax, 0., dMag)
            else:
                comp = self.EVPOC_hs[subpop[0],subpop[1]](smin, smax, 0., dMag)
            # remove small values
            comp[comp<1e-6] = 0.
            
            return comp
