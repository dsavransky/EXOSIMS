# -*- coding: utf-8 -*-
import numpy as np
import sys
import astropy.units as u
import astropy.constants as const
import copy
from EXOSIMS.util.get_module import get_module

class SurveySimulation(object):
    """Survey Simulation class template
    
    This class contains all variables and methods necessary to perform
    Survey Simulation Module calculations in exoplanet mission simulation.
    
    It inherits the following class objects which are defined in __init__:
    Simulated Universe, Observatory, TimeKeeping, PostProcessing
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        SimulatedUniverse (SimulatedUniverse):
            SimulatedUniverse class object
        Observatory (Observatory):
            Observatory class object
        TimeKeeping (TimeKeeping):
            TimeKeeping class object
        PostProcessing (PostProcessing):
            PostProcessing class object
        TargetList (TargetList):
            TargetList class object
        PlanetPhysicalModel (PlanetPhysicalModel):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        Completeness (Completeness):
            Completeness class object
        DRM (list):
            list containing results of survey simulation
        
    """

    _modtype = 'SurveySimulation'
    
    def __init__(self,scriptfile=None,**specs):
        """Initializes Survey Simulation with default values
        
        Input: 
            scriptfile:
                JSON script file.  If not set, assumes that 
                dictionary has been passed through specs 
            specs: 
                Dictionary containing user specification values and 
                a dictionary of modules.  The module dictionary can contain
                string values, in which case the objects will be instantiated,
                or object references.
        """

        # if a script file is provided read it in
        if scriptfile is not None:
            import json
            import os.path
        
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile

            try:
                script = open(scriptfile).read()
                specs = json.loads(script)
            except ValueError:
                print "%s improperly formatted."%scriptfile
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

            # modules array must be present
            if 'modules' not in specs.keys():
                raise ValueError("No modules field found in script.")

        #if any of the modules is a string, assume that they are all strings and we need to initalize
        if isinstance(specs['modules'].itervalues().next(),basestring):

            # get desired module names (prototype or specific)
            # import simulated universe class
            SimUni = get_module(specs['modules']['SimulatedUniverse'], 'SimulatedUniverse')
            # import observatory class
            Obs = get_module(specs['modules']['Observatory'], 'Observatory')
            # import timekeeping class
            TK = get_module(specs['modules']['TimeKeeping'], 'TimeKeeping')
            # import postprocessing class
            PP = get_module(specs['modules']['PostProcessing'], 'PostProcessing')
            
            self.SimulatedUniverse = SimUni(**specs)
            self.Observatory = Obs(**specs)
            self.TimeKeeping = TK(**specs)
            self.PostProcessing = PP(**specs)
            
            # bring inherited class objects to top level of Survey Simulation
            self.OpticalSystem = self.SimulatedUniverse.OpticalSystem # optical system class object
            self.PlanetPopulation = self.SimulatedUniverse.PlanetPopulation # planet population class object
            self.ZodiacalLight = self.SimulatedUniverse.ZodiacalLight # zodiacal light class object
            self.Completeness = self.SimulatedUniverse.Completeness # completeness class object
            self.PlanetPhysicalModel = self.SimulatedUniverse.PlanetPhysicalModel # planet physical model class object
            self.TargetList = self.SimulatedUniverse.TargetList # target list class object
        else:
            #these are the modules that must be present if you are passed already instantiated objects
            neededObjMods = ['SimulatedUniverse',
                          'Observatory',
                          'TimeKeeping',
                          'PostProcessing',
                          'OpticalSystem',
                          'PlanetPopulation',
                          'ZodiacalLight',
                          'Completeness',
                          'PlanetPhysicalModel',
                          'TargetList']

            #ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs['modules'].keys():
                    raise ValueError("%s module not provided."%modName)

            for modName in specs['modules'].keys():
                assert (specs['modules'][modName]._modtype == modName), \
                "Provided instance of %s has incorrect modtype."%modName

                setattr(self, modName, specs['modules'][modName])
                
        # list of simulation results, each item is a dictionary
        self.DRM = []        
    
    def __str__(self):
        """String representation of the Survey Simulation object
        
        When the command 'print' is used on the Survey Simulation object, this 
        method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Survey Simulation class object attributes'
        
    def run_sim(self):
        """Performs the survey simulation 
        
        This method has access to the following:
            self.SimulatedUniverse:
                SimulatedUniverse class object
            self.Observatory:
                Observatory class object
            self.TimeKeeping:
                TimeKeeping class object
            self.PostProcessing:
                PostProcessing class object
            self.OpticalSystem:
                OpticalSystem class object
            self.PlanetPopulation:
                PlanetPopulation class object
            self.ZodiacalLight:
                ZodiacalLight class object
            self.Completeness:
                Completeness class object
            self.PlanetPhysicalModel:
                PlanetPhysicalModel class object
            self.TargetList:
                TargetList class object
        
        Returns:
            string (str):
                String 'Simulation results in .DRM'
        
        """
        
        # initialize values updated later
        # number of visits to target star
        visited = np.zeros((len(self.TargetList.dist),), dtype=int)
        # target revisit list
        revisit_list = np.array([])
        # current time (normalized to zero at mission start) of planet positions
        planPosTime = np.array([self.TimeKeeping.currenttimeNorm.to(u.day).value]*self.SimulatedUniverse.nPlans)*u.day
        # number of planet observations
        observed = np.zeros((self.SimulatedUniverse.nPlans,), dtype=int)
        # set occulter separation if haveOcculter
        if self.OpticalSystem.haveOcculter:
            self.Observatory.currentSep = self.Observatory.occulterSep
        
        extended_list = np.array([])
        
        # initialize run options
        # perform characterization if doChar = True
        if self.OpticalSystem.SNchar > 0.:
            doChar = True
        else:
            doChar = False
        
        # keep track of spectral characterizations, 0 is no characterization           
        spectra = np.zeros((self.SimulatedUniverse.nPlans,1), dtype=int)
            
        # get index of first target star
        s_ind = self.initial_target()
        
        # loop until mission is finished
        while self.TimeKeeping.currenttimeNorm < self.TimeKeeping.missionFinishNorm:
            print 'Current mission time'
            print self.TimeKeeping.currenttimeNorm
            obsbegin = copy.copy(self.TimeKeeping.currenttimeNorm)
            # dictionary containing results
            DRM = {}
            # store target star index
            DRM['target_ind'] = s_ind
            # store arrival time
            DRM['arrival_time'] = self.TimeKeeping.currenttimeNorm.to(u.day).value
            
            # store spacecraft mass
            if self.OpticalSystem.haveOcculter:
                DRM['sc_mass'] = self.Observatory.sc_mass.to(u.kg).value
                        
            # get target list star index of detections for extended_list
            if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.missionLife and extended_list.shape[0] == 0:
                for i in xrange(len(self.DRM)):
                    if self.DRM[i]['det'] == 1:
                        extended_list = np.hstack((extended_list, self.DRM[i]['target_ind']))
                        extended_list = np.unique(extended_list)
            
            # find planets belonging to target star
            pInds = np.where(self.SimulatedUniverse.planInds == s_ind)[0]
            
            # update visited list for current star
            visited[s_ind] += 1

            # find out if observations are possible and get relevant data
            observationPossible, t_int, DRM, s, dMag, Ip = self.observation_detection(pInds, s_ind, DRM, planPosTime)        
            t_int += self.Observatory.settling_time
            # store detection integration time
            DRM['det_int_time'] = t_int.to(u.day).value
            
            # if integration time goes beyond observation duration, set quantities
            if self.TimeKeeping.currenttimeNorm + t_int > self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration:
                # integration time is beyond observation duration
                observationPossible = False
                dt = self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration + 1.*u.day - self.TimeKeeping.currenttimeNorm
                self.TimeKeeping.update_times(dt)
            else:
                # integration time is okay
                self.TimeKeeping.update_times(t_int)

            # determine detection, missed detection, false alarm booleans
            FA, DET, MD, NULL = self.PostProcessing.det_occur(observationPossible)
            
            # encode detection status
            s, dMag, Ip, DRM, observed = self.det_data(s, dMag, Ip, DRM, FA, 
                                                       DET, MD, s_ind, pInds, 
                                                       observationPossible, 
                                                       observed)
            
            if doChar:
                DRM, FA, spectra = self.observation_characterization(observationPossible, pInds, s_ind, spectra, s, Ip, DRM, FA, t_int)
                
            # schedule a revisit if there is a planet
            if pInds.shape[0] != 0:
                if DET or FA:
                    # find mu based on planet with minimum separation
                    if np.isscalar(s.value):
                        mu = const.G*(self.SimulatedUniverse.Mp[pInds] + self.TargetList.MsTrue[s_ind]*const.M_sun)
                    else:
                        muind = np.where(np.min(s))[0][0]
                        mu = const.G*(self.SimulatedUniverse.Mp[pInds[muind]] + self.TargetList.MsTrue[s_ind]*const.M_sun)    
                    # calculate the orbital period based on minimum apparent separation
                    T = 2.*np.pi*np.sqrt(np.min(s)**3/mu)
                    t_rev = self.TimeKeeping.currenttimeNorm + T/2.
                else:
                    # find mu based on planet with minimum separation
                    if np.isscalar(s.value):
                        mu = const.G*(self.SimulatedUniverse.Mp[pInds] + self.TargetList.MsTrue[s_ind]*const.M_sun)
                    else:
                        muind = np.where(s == np.min(s))[0][0]
                        mu = const.G*(self.SimulatedUniverse.Mp[pInds[muind]] + self.TargetList.MsTrue[s_ind]*const.M_sun)
                    # calculate orbital period based on average of population semi-major axis
                    T = 2.*np.pi*np.sqrt(self.PlanetPopulation.arange.mean()**3/mu)
                    t_rev = self.TimeKeeping.currenttimeNorm + 0.75*T
            else:
                # revisit based on average of population semi-major axis and mass
                Mp = self.SimulatedUniverse.Mp.mean()
                mu = const.G*(Mp + self.TargetList.MsTrue[s_ind]*const.M_sun)
                T = 2.*np.pi*np.sqrt(self.PlanetPopulation.arange.mean()**3/mu)
                t_rev = self.TimeKeeping.currenttimeNorm + 0.75*T
            
            # populate revisit list (s_ind is converted to float)
            if revisit_list.size == 0:
                revisit_list = np.hstack((revisit_list, np.array([s_ind, t_rev.to(u.day).value])))
            else:
                revisit_list = np.vstack((revisit_list, np.array([s_ind, t_rev.to(u.day).value])))
            
            obsend = copy.copy(self.TimeKeeping.currenttimeNorm)
            # find next available time for planet-finding
            nexttime = self.TimeKeeping.duty_cycle(self.TimeKeeping.currenttimeNorm)
          
            dt = nexttime - self.TimeKeeping.currenttimeNorm

            self.TimeKeeping.update_times(dt)
            
            # update completeness values
            self.TargetList.comp0 = self.Completeness.completeness_update(s_ind, self.TargetList, obsbegin, obsend, nexttime)
            
            # acquire a new target star index
            s_ind, DRM = self.next_target(s_ind, self.TargetList, revisit_list, extended_list, DRM)
            
            # append result values to self.DRM
            self.DRM.append(DRM)
            
            # with occulter if spacecraft fuel is depleted, exit loop
            if self.OpticalSystem.haveOcculter:
                if self.Observatory.sc_mass < self.Observatory.dryMass:
                    print 'Total fuel mass excedeed at %r' % self.TimeKeeping.currenttimeNorm                    
                    break
#            break
#            if visited[s_ind] == 3:
#                self.TimeKeeping.currenttimeNorm = 100*u.year
#            if self.TimeKeeping.currenttimeNorm > 100*u.day:
#                self.TimeKeeping.currenttimeNorm = 100*u.year                       
        
        return 'Simulation results in .DRM'
        
    def initial_target(self):
        """Returns index of initial target star
        
        This method has access to the following:
            self.SimulatedUniverse:
                SimulatedUniverse class object
            self.Observatory:
                Observatoryervatory class object
            self.TimeKeeping:
                TimeKeeping class object
            self.PostProcessing:
                PostProcessing class object
            self.OpticalSystem:
                OpticalSystem class object
            self.PlanetPopulation:
                PlanetPopulation class object
            self.ZodiacalLight:
                ZodiacalLight class object
            self.Completeness:
                Completenessleteness class object
            self.PlanetPhysicalModel:
                PlanetPhysicalModel class object
            self.TargetList:
                TargetList class object
        
        Returns:
            s_ind (int):
                index of initial target star
        
        """
        
        sinds = np.array([])
        dt = 1.*u.day
        while sinds.size == 0:
            a = self.Observatory.keepout(self.TimeKeeping.currenttimeAbs, self.TargetList, self.OpticalSystem.telescopeKeepout)
            # find Observatoryervable targets at current mission time            
            sinds = np.where(self.Observatory.kogood)[0]
            # if no observable targets, update mission time by one day and 
            # check that resulting time is okay with duty cycle
            if sinds.size == 0:
                self.TimeKeeping.update_times(dt)
                if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration:
                    nexttime = self.TimeKeeping.duty_cycle(self.TimeKeeping.currenttimeNorm)
                    dt0 = nexttime - self.TimeKeeping.currenttimeNorm
                    self.TimeKeeping.update_times(dt0)
                
            # if the current mission time is greater than the mission lifetime
            # break this loop
            if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.missionLife:
                break
        
        if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.missionLife:
            print 'No targets available'
            
        s0 = self.TargetList.comp0[sinds].argmax()
        s_ind = sinds[s0]
        
        return s_ind
        
    def observation_detection(self, pInds, s_ind, DRM, planPosTime):
        """Finds if planet observations are possible and relevant information
        
        This method makes use of the following inherited class objects:
        
        Args:
            pInds (ndarray):
                1D numpy ndarray of planet indices
            s_ind (int):
                target star index
            DRM (dict):
                dictionary containing simulation results
            planPosTime (Quantity):
                1D numpy ndarray containing times of planet positions (units of
                time)
        
        Returns:
            observationPossible, t_int, DRM, s, dMag, Ip (ndarray, Quantity, dict, Quantity, ndarray, Quantity):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible, integration time (units of time), 
                dictionary containing survey simulation results, apparent 
                separation (units of distance), 1D numpy ndarray of delta 
                magnitude, difference in magnitude between planet and star,
                irradiance (units of :math:`1/(m^2*nm*s)`)                
        
        """
        
        # dMag and Ip placeholders
        dMag = self.OpticalSystem.dMagLim - np.abs(np.random.randn(1)*0.5)
        Ip = (9.5e7/(u.m**2)/u.nm/u.s)*10.**(-(self.TargetList.Vmag[s_ind]+dMag)/2.5) 
        # apparent separation placeholders
        if pInds.size == 0:
            s = np.array([1.])*u.AU
        else:
            s = np.sqrt(self.SimulatedUniverse.r[pInds][:,0]**2+self.SimulatedUniverse.r[pInds][:,1]**2)
        # determine if there are any planets at the target star and
        # propagate the system if necessary
        if pInds.shape[0] != 0:
            # there are planets so observation is possible
            observationPossible = np.ones((len(pInds),), dtype=bool)
            # if planet position times do not match up with current time
            # propagate the positions to the current time
            if planPosTime[pInds][0] != self.TimeKeeping.currenttimeNorm:
                # propagate planet positions and velocities
                try:
                    self.SimulatedUniverse.r[pInds], self.SimulatedUniverse.v[pInds] = \
                    self.SimulatedUniverse.prop_system(self.SimulatedUniverse.r[pInds], 
                                        self.SimulatedUniverse.v[pInds], 
                                        self.SimulatedUniverse.Mp[pInds], 
                                        self.TargetList.MsTrue[self.SimulatedUniverse.planInds[pInds]], 
                                        self.TimeKeeping.currenttimeNorm - planPosTime[pInds][0])
                    # update planet position times
                    planPosTime[pInds] = self.TimeKeeping.currenttimeNorm
                except ValueError:
                    observationPossible = False
        else:
            observationPossible = False

        # determine if planets are inside IWA or outside OWA
        if np.any(observationPossible):
            observationPossible, s = self.check_IWA_OWA(observationPossible,self.SimulatedUniverse.r[pInds],self.TargetList.dist[s_ind],self.OpticalSystem.IWA,self.OpticalSystem.OWA)

        # determine if planets are bright enough
        if np.any(observationPossible):
            observationPossible, dMag = self.check_brightness(observationPossible, self.SimulatedUniverse.r[pInds], self.SimulatedUniverse.Rp[pInds], self.SimulatedUniverse.p[pInds], self.OpticalSystem.dMagLim)

        # set integration time to max integration time as a default
        t_int = self.TargetList.maxintTime[s_ind]
            
        # determine integration time and update observationPossible
        if np.any(observationPossible):
            # find irradiance
            Ip = (9.5e7/(u.m**2)/u.nm/u.s)*10.**(-(self.TargetList.Vmag[s_ind]+dMag)/2.5)
            # find true integration time
            t_trueint = self.OpticalSystem.calc_intTime(self.TargetList, self.SimulatedUniverse, s_ind, pInds)
            # update observationPossible
            observationPossible = np.logical_and(observationPossible, t_trueint <= self.OpticalSystem.intCutoff) 

        # determine if planets are observable at the end of observation
        # and update integration time
        if np.any(observationPossible):
            try:
                t_int, observationPossible = self.check_visible_end(observationPossible, t_int, t_trueint, s_ind, pInds, False)
            except ValueError:
                observationPossible = False
                
        if self.OpticalSystem.haveOcculter:
            # find disturbance forces on occulter
            dF_lateral, dF_axial = self.Observatory.distForces(self.TimeKeeping, self.TargetList, s_ind)    
            # store these values
            DRM['dF_lateral'] = dF_lateral.to(u.N).value
            DRM['dF_axial'] = dF_axial.to(u.N).value
            # decrement mass for station-keeping
            intMdot, mass_used, deltaV = self.Observatory.mass_dec(dF_lateral, t_int)
            # store these values
            DRM['det_dV'] = deltaV.to(u.m/u.s).value
            DRM['det_mass_used'] = mass_used.to(u.kg).value
            self.Observatory.sc_mass -= mass_used
            
        return observationPossible, t_int, DRM, s, dMag, Ip
        
    def observation_characterization(self, observationPossible, pInds, s_ind, spectra, s, Ip, DRM, FA, t_int):
        """Finds if characterizations are possible and relevant information
        
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            pInds (ndarray):
                1D numpy ndarray of planet indices
            s_ind (int):
                target star index
            spectra (ndarray):
                numpy ndarray of values indicating if planet spectra has been
                captured
            s (Quantity):
                apparent separation (units of distance)
            Ip (Quantity):
                irradiance (units of flux)
            DRM (dict):
                dictionary containing survey simulation results
            FA (bool):
                False Alarm boolean
            t_int (Quantity):
                integration time (units of time)
        
        Returns:
            DRM, FA, spectra (dict, bool, ndarray):
                dictionary containing survey simulation results, False Alarm 
                boolean, numpy ndarray of values indicating if planet spectra 
                has been captured                
                
        """
        
        # check if characterization has been done
        if pInds.shape[0] != 0:
            if np.any(spectra[pInds[observationPossible],0] == 0):
                # perform first characterization
                # find throughput and contrast
                throughput = self.OpticalSystem.throughput(self.OpticalSystem.specLambda, np.arctan(s/(self.TargetList.dist[s_ind]*u.pc)))
                contrast = self.OpticalSystem.contrast(self.OpticalSystem.specLambda, np.arctan(s/(self.TargetList.dist[s_ind]*u.pc)))
                # find characterization time                    
                t_char = self.find_t_char(throughput, contrast, Ip, FA, s_ind, pInds)
                # account for 5 bands and one coronagraph
                t_char = t_char*4
                # determine which planets will be observable at the end
                # of observation
                charPossible = observationPossible                   
                charPossible = np.logical_and(charPossible, t_char <= self.OpticalSystem.intCutoff)
                chargo = False
                        
                try:
                    t_char, charPossible, chargo = self.check_visible_end(charPossible, t_char, t_char, s_ind, pInds, True)
                except ValueError:
                    chargo = False
                #chargo = True            
                if chargo:
                    # encode relevant first characterization data
                    if self.OpticalSystem.haveOcculter:
                        # decrement sc mass
                        # find disturbance forces on occulter
                        dF_lateral, dF_axial = self.Observatory.distForces(self.TimeKeeping, self.TargetList, s_ind)    
                        # decrement mass for station-keeping
                        intMdot, mass_used, deltaV = self.Observatory.mass_dec(dF_lateral, t_int)
                        mass_used_char = t_char*intMdot
                        deltaV_char = dF_lateral/self.Observatory.sc_mass*t_char
                        self.Observatory.sc_mass -= mass_used_char
                        # encode information in DRM
                        DRM['char_1_time'] = t_char.to(u.day).value
                        DRM['char_1_dV'] = deltaV_char.to(u.m/u.s).value
                        DRM['char_1_mass_used'] = mass_used_char.to(u.kg).value
                    else:
                        DRM['char_1_time'] = t_char.to(u.day).value
                
                    # if integration time goes beyond observation duration, set quantities
                    if self.TimeKeeping.currenttimeNorm + t_char.max() > self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration:
                        # integration time is beyond observation duration
                        charPossible = False
                        dt = self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration + 1.*u.day - self.TimeKeeping.currenttimeNorm
                        self.TimeKeeping.update_times(dt)
                    else:        
                        # update mission time
                        self.TimeKeeping.update_times(t_char.max())
                        
                    # if this was a false alarm, it has been noted, update FA
                    if FA:
                        FA = False
                        
                    # if planet is visible at end of characterization,
                    # spectrum is captured
                    if np.any(charPossible):
                        if self.OpticalSystem.haveOcculter:
                            spectra[pInds[charPossible],0] = 1
                            # encode success
                            DRM['char_1_success'] = 1
                        else:
                            nld = self.OpticalSystem.IWA.to(u.rad).value*np.sqrt(
                            self.OpticalSystem.pupilArea/self.OpticalSystem.shapeFac)/self.OpticalSystem.lam
                            lambdaeff = (np.arctan(s/(self.TargetList.dist[s_ind]*u.pc)).to(u.rad).value)
                            lambdaeff = lambdaeff*np.sqrt(self.OpticalSystem.pupilArea/self.OpticalSystem.shapeFac)
                            lambdaeff = lambdaeff/nld
                            charPossible = np.logical_and(charPossible, lambdaeff >= 800.*u.nm)
                            # encode results
                            if np.any(charPossible):
                                spectra[pInds[charPossible],0] = 1
                                DRM['char_1_success'] = 1
                            else:
                                DRM['char_1_success'] = lambdaeff.max().to(u.nm).value
                       
        return DRM, FA, spectra
                
    def check_IWA_OWA(self, observationPossible, r, dist, IWA, OWA):
        """Determines whether planets are inside the IWA or outside the OWA
        
        This method returns a 1D numpy array of booleans where True is outside 
        IWA and inside OWA.
        
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            r (Quantity):
                numpy ndarray of planet position vectors (units of distance)
            dist (float):
                target star distance (in pc)
            IWA (Quantity):
                instrument inner working angle (angle units)
            OWA (Quantity):
                instrument outer working angle (angle units)
        
        Returns:
            observationPossible, s (ndarray, Quantity):
                updated 1D numpy ndarray of booleans indicating if an 
                observation of each planet is possible, apparent separation 
                (units of km)                
        
        """
        
        # find apparent separation
        s = np.sqrt(r[:,0]**2 + r[:,1]**2)
        
        observationPossible = np.logical_and(observationPossible, np.logical_and(
        s >= dist*u.pc*np.tan(IWA), s <= dist*u.pc*np.tan(OWA)))
        
        return observationPossible, s
        
    def check_brightness(self, observationPossible, r, Rp, p, dMagLim):
        """Determines if planets are bright enough for detection 
        
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            r (Quantity):
                numpy ndarray of planet position vectors (units of distance)
            Rp (Quantity):
                1D numpy ndarray of planet radii (units of distance)
            p (ndarray):
                1D numpy ndarray of planet albedos
            dMagLim (float):
                limiting difference in brightness between planet and star
                
        Returns:
            observationPossible, dMag (ndarray, ndarray):
                updated 1D numpy array of booleans indicating if an observation
                of each planet is possible, delta magnitude (difference between 
                star and planet magnitude)                
        
        """
        
        dMag = -2.5*np.log10(p*(Rp**2/(np.sqrt(np.sum(r**2, axis=1)))).decompose().value)
        # planetary phase function calculation, from Sobolev 1975
        beta = np.arccos(r[:,2]/np.sqrt(np.sum(r**2, axis=1)))
        Phi = (np.sin(beta) + (np.pi - beta.to(u.rad).value)*np.cos(beta))/np.pi

        # delta magnitude of each planet
        dMag -= 2.5*np.log10(Phi)
        observationPossible = np.logical_and(observationPossible, dMag <= dMagLim)
        
        return observationPossible, dMag
    
    def check_visible_end(self, observationPossible, t_int, t_trueint, s_ind, pInds, t_char_calc):
        """Determines if planets are visible at the end of the observation time
        
        This method makes use of the following inherited objects:
            self.TargetList:
                TargetList class object
            self.Observatory:
                Observatory class object
            self.TimeKeeping:
                TimeKeeping class object
            self.SimulatedUniverse:
                SimulatedUniverse class object
            self.OpticalSystem:
                OpticalSystem class object
                
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            t_int (Quantity):
                integration time (units of time)
            t_trueint (Quantity):
                1D numpy ndarray of calculated integration times for planets
                (units of time)
            s_ind (int):
                target star index
            pInds (ndarray):
                1D numpy ndarray of planet indices
            t_char_calc (bool):
                boolean where True is for characterization calculations
                
        Returns:
            t_int, observationPossible, chargo (Quantity, ndarray, bool):
                true integration time for planetary system (units of day) 
                (t_char_calc = False) or maximum characterization time for 
                planetary system (t_char_calc = True), updated 1D numpy ndarray 
                of booleans indicating if an observation or characterization of 
                each planet is possible, boolean where True is to encode 
                characterization data (t_char_calc = True only)                
        
        """
        # set chargo to False initially
        chargo = False
        
        for i in xrange(len(observationPossible)):
            if observationPossible[i]:
                r = self.SimulatedUniverse.r[pInds[i]]
                v = self.SimulatedUniverse.v[pInds[i]]
                Mp = self.SimulatedUniverse.Mp[pInds[i]]
                MsTrue = self.TargetList.MsTrue[s_ind]
                Rp = self.SimulatedUniverse.Rp[pInds[i]]
                p = self.SimulatedUniverse.p[pInds[i]]
                if t_char_calc:
                    dt = t_int[i]
                else:
                    dt = t_trueint[i] + self.Observatory.settling_time

                obsRes = self.sys_end_res(dt, s_ind, r, v, Mp, MsTrue, Rp, p)

                if obsRes == 1:
                    # planet visible at the end of observation
                    observationPossible[i] = True
                    if not t_char_calc:
                        # update integration time
                        if t_int == self.TargetList.maxintTime[s_ind]:
                            t_int = t_trueint[i]
                        else:
                            if t_trueint[i] > t_int:
                                t_int = t_trueint[i]
                
                if obsRes != -1:
                    if t_char_calc:
                        chargo = True
                        
        if t_char_calc:
            if np.any(observationPossible):
                t_int = np.max(t_int[observationPossible])
            
            return t_int, observationPossible, chargo
        else:
            return t_int, observationPossible
        
    def sys_end_res(self, dt, s_ind, r, v, Mp, MsTrue, Rp, p):
        """Determines if a planet is visible at the end of the observation time
        
        This method makes use of the following inherited objects:        
            self.Observatory:
                Observatory class object
            self.TimeKeeping:
                TimeKeeping class object
            self.TargetList:
                TargetList class object
            self.OpticalSystem:
                OpticalSystem class object
                
        Args:
            dt (Quantity):
                propagation time (units of time)
            s_ind (int):
                target star index
            r (Quantity):
                planet position vector as 1D numpy ndarray (units of distance)
            v (Quantity):
                planet velocity vector as 1D numpy ndarray (units of velocity)
            Mp (Quantity):
                planetary mass (units of mass)
            MsTrue (float):
                stellar mass (in M_sun)
            Rp (Quantity):
                planet radius (units of distance)
            p (ndarray):
                planet albedo
                
        Returns:
            obsRes (int):
                -1 if star is inside the keepout zone, 0 if planet is not 
                visible, or 1 if planet is visible
        
        """
        
        # update keepout to end of observation time
        a = self.Observatory.keepout(self.TimeKeeping.currenttimeAbs+dt, self.TargetList, self.OpticalSystem.telescopeKeepout)
        # if star is not inside keepout zone at end of observation, see if 
        # planet is still observable
        if self.Observatory.kogood[s_ind]:
            # propagate planet to observational period end and find dMag
            rend, vend = self.SimulatedUniverse.prop_system(r, v, Mp, MsTrue, dt)
            # calculate dMagend
            send = np.sqrt(rend[0].to(u.km)**2 + rend[1].to(u.km)**2)
            rendmag = np.linalg.norm(rend.to(u.km).value)
            beta = np.arccos(rend[2].value/rendmag)
            Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi
            dMagend = -2.5*np.log10(Rp.to(u.km).value**2*p) -2.5*np.log10(Phi) + 5.*np.log10(rendmag)
           
            if dMagend <= self.OpticalSystem.dMagLim and send >= self.TargetList.dist[s_ind]*u.pc*np.tan(self.OpticalSystem.IWA):
                obsRes = 1
            else:
                obsRes = 0
        else:
            obsRes = -1
            
        return obsRes

    def det_data(self, s, dMag, Ip, DRM, FA, DET, MD, s_ind, pInds, observationPossible, observed):
        """Determines detection status
        
        This method updates the arguments s, dMag, Ip, and DRM based on the 
        detection status (FA, DET, MD) and encodes values in the DRM 
        dictionary.

        This method accesses the following inherited class objects:
            self.OpticalSystem:
                OpticalSystem class object
            self.TargetList:
                TargetList class object
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Args:
            s (Quantity):
                apparent separation (units of distance)
            dMag (ndarray):
                delta magnitude
            Ip (Quantity):
                irradiance (units of flux per time)
            DRM (dict):
                dictionary containing simulation results
            FA (bool):
                Boolean signifying False Alarm
            DET (bool):
                Boolean signifying DETection
            MD (bool):
                Boolean signifying Missed Detection
            s_ind (int):
                index of star in target list
            pInds (ndarray):
                idices of planets belonging to target star
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            observed (ndarray):
                1D numpy ndarray indicating number of observations for each
                planet
        
        Returns:
            s, dMag, Ip, DRM, observed (Quantity, ndarray, Quantity, dict, ndarray):
                apparent separation (units of distance), delta magnitude, 
                irradiance (units of flux per time), dictionary containing 
                simulation results, 1D numpy ndarray indicating number of 
                observations for each planet                
        
        """
        # default DRM detection status to null detection
        DRM['det_status'] = 0
        
        if FA:
            # false alarm
            DRM['det_status'] = -2
            # generate random WA and dMag for detection
            s = self.TargetList.dist[s_ind]*u.pc*np.tan(self.OpticalSystem.IWA)
            if self.PlanetPopulation.scaleOrbits:
                s += np.random.rand()*(self.PlanetPopulation.arange.max() - self.PlanetPopulation.arange.min())*np.sqrt(self.TargetList.L[s_ind])
            else:
                s += np.random.rand()*(self.PlanetPopulation.arange.max() - self.PlanetPopulation.arange.min())
            dMag = self.OpticalSystem.dMagLim - np.abs(np.random.randn()*0.5)
            Ip = (9.5e7/(u.m**2)/u.nm/u.s)*10.**(-(self.TargetList.Vmag[s_ind]+dMag)/2.5)
        elif MD:
            # missed detection, set status in DRM
            DRM['det_status'] = -1
        elif DET:
            # encode detection
            observed[pInds[observationPossible]] += 1
            if pInds.shape[0] == 1:
                # set status in DRM
                DRM['det_status'] = 1
                # set WA and Delta Mag of detection in DRM
                DRM['det_WA'] = np.arctan(s/(self.TargetList.dist[s_ind]*u.pc)).to(u.marcsec)[0].value
                DRM['det_dMag'] = dMag[0]
            else:
                DRM['det_status'] = observationPossible.astype(int).tolist()
                DRM['det_WA'] = np.arctan(s/(self.TargetList.dist[s_ind]*u.pc)).to(u.marcsec).min().value
                DRM['det_dMag'] = (dMag[observationPossible]).max()
        
        return s, dMag, Ip, DRM, observed
        
    def find_t_char(self, throughput, contrast, Ip, FA, s_ind, pInds):
        """Finds characterization time
        
        This method uses the following inherited class objects:
            self.OpticalSystem:
                OpticalSystem class object
            self.ZodiacalLight:
                ZodiacalLight class object
            self.TargetList:
                TargetList class object
                
        Args:
            throughput (float):
                scalar value for throughput
            contrast (float):
                scalar value for contrast
            Ip (Quantity):
                irradiance (units of flux per time)
            FA (bool):
                False Alarm boolean
            s_ind (int):
                index of current target star in target list
            pInds (ndarray):
                indices of planets belonging to target star
                
        Returns:
            t_char (Quantity):
                characterization time (units of s)
        
        """
        
        QE = self.OpticalSystem.QE(self.OpticalSystem.specLambda)
        FluxB = 9.5e7/(u.m**2)/u.nm/u.s
        
        # planet count
        cp = QE*self.OpticalSystem.eta2*(self.OpticalSystem.specLambda/self.OpticalSystem.Rspec)*(Ip/(9.5e7/
        (u.m**2)/u.nm/u.s))*FluxB*throughput*self.OpticalSystem.pupilArea
        # star count
        cs = QE*self.OpticalSystem.eta2*(self.OpticalSystem.specLambda/self.OpticalSystem.Rspec)*(
        self.OpticalSystem.pupilArea)*FluxB*10.**(-self.TargetList.Vmag[s_ind]/2.5)*contrast/2
        
        # zodi count
        if FA:
            temp = self.ZodiacalLight.fzodi(s_ind, 0., self.TargetList)
            Z = FluxB*10.**(-23.54/2.5)*(1/u.arcsec**2)*temp*(self.OpticalSystem.specLambda/
            self.OpticalSystem.Rspec)*self.OpticalSystem.pupilArea*(self.OpticalSystem.pixelArea/
            (self.OpticalSystem.focalLength**2))*u.sr
        else:
            Z = FluxB*10.**(-23.54/2.5)*(1/u.arcsec**2)*(
            self.SimulatedUniverse.fzodicurr[pInds])*(self.OpticalSystem.specLambda/self.OpticalSystem.Rspec)*(
            self.OpticalSystem.pupilArea)*(self.OpticalSystem.pixelArea/(self.OpticalSystem.focalLength**2))*u.sr
        
        t_char = (self.OpticalSystem.SNchar/cp*((self.OpticalSystem.Npix*(self.OpticalSystem.sigma_r**2/
        self.OpticalSystem.t_exp + self.OpticalSystem.dr)*(1.+1./self.OpticalSystem.Ndark) + Z) + cp + cs))**2
        t_char = t_char*u.s

        return t_char
        
    def next_target(self, s_ind, targlist, revisit_list, extended_list, DRM):
        """Finds index of next target star
        
        This method chooses the next target star index at random based on which
        stars are available.
        
        This method makes use of the following class objects inherited by the
        SurveySimulation class object:
            self.TimeKeeping:
                TimeKeeping class object
            self.Observatory:
                Observatory class object
        
        Args:
            s_ind (int):
                index of current target star
            targlist (TargetList):
                Target List module
            revisit_list (ndarray):
                numpy ndarray of containing index and time (in days) of targets 
                to revisit
            extended_list (ndarray):
                1D numpy ndarray of star indices for extended mission time
            DRM (dict):
                dictionary of simulation results
                
        Returns:
            new_s_ind, DRM (int, dict):
                index of next target star, dictionary of simulation results                
        
        """
        
        sinds = np.array([])
        dt = 1.*u.day
        while sinds.size == 0:
            a = self.Observatory.keepout(self.TimeKeeping.currenttimeAbs, self.TargetList, self.OpticalSystem.telescopeKeepout)
            # find observable targets at current mission time            
            sinds = np.where(self.Observatory.kogood)[0]
            # if no observable targets, update mission time by one day and 
            # check that resulting time is okay with duty cycle
            if sinds.size == 0:
                self.TimeKeeping.update_times(dt)
                if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration:
                    nexttime = self.TimeKeeping.duty_cycle(self.TimeKeeping.currenttimeNorm)
                    dt0 = nexttime - self.TimeKeeping.currenttimeNorm
                    self.TimeKeeping.update_times(dt0)
                
            # if the current mission time is greater than the mission lifetime
            # break this loop
            if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.missionLife:
                break
        
        if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.missionLife:
            print 'No targets available'
        
        # pick a random star from the stars not in keepout zones
        s0 = np.random.random_integers(0, high=len(sinds)-1)
        
        new_s_ind = sinds[s0]
        
        if self.OpticalSystem.haveOcculter:
            # add transit time and reduce starshade mass
            ao = self.Observatory.thrust/self.Observatory.sc_mass
            targetSep = self.Observatory.occulterSep
            # find position vector of previous target star
            r_old = self.Observatory.starprop(self.TimeKeeping.currenttimeAbs, self.TargetList, s_ind)
            # find position vector of new target star            
            r_new = self.Observatory.starprop(self.TimeKeeping.currenttimeAbs, self.TargetList, new_s_ind)
            # find unit vectors
            u_old = r_old/np.sqrt(np.sum(r_old**2))
            u_new = r_new/np.sqrt(np.sum(r_new**2))
            # find angle between old and new stars
            sd = np.arccos(np.dot(u_old, u_new))
            # find slew distance
            slew_dist = 2.*targetSep*np.sin(sd/2.)
            slew_time = np.sqrt(slew_dist/np.abs(ao)/(self.Observatory.defburnPortion/2. - self.Observatory.defburnPortion**2/4.))
            mass_used = slew_time*self.Observatory.defburnPortion*self.Observatory.flowRate

            # update times
            self.TimeKeeping.update_times(slew_time)
            
            if self.TimeKeeping.currenttimeNorm > self.TimeKeeping.nexttimeAvail + self.TimeKeeping.duration:
                nexttime = self.TimeKeeping.duty_cycle(self.TimeKeeping.currenttimeNorm)
                dt = nexttime - self.TimeKeeping.currenttimeNorm
                self.TimeKeeping.update_times(dt)
            
            DRM['slew_time'] = slew_time.to(u.day).value
            DRM['slew_dV'] = (ao*slew_time*self.Observatory.defburnPortion).to(u.m/u.s).value
            DRM['slew_mass_used'] = mass_used.to(u.kg).value
            DRM['slew_angle'] = sd     
        
        return new_s_ind, DRM
