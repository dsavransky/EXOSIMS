# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
import sys, logging
import numpy as np
import astropy.units as u
import astropy.constants as const
import random as py_random
import time
import json, os.path, copy, re, inspect, subprocess
import hashlib

Logger = logging.getLogger(__name__)

class SurveySimulation(object):
    """Survey Simulation class template
    
    This class contains all variables and methods necessary to perform
    Survey Simulation Module calculations in exoplanet mission simulation.
    
    It inherits the following class objects which are defined in __init__:
    Simulated Universe, Observatory, TimeKeeping, PostProcessing
    
    Args:
        \*\*specs:
            user specified values
        scriptfile (string):
            JSON script file.  If not set, assumes that dictionary has been 
            passed through specs.
            
    Attributes:
        StarCatalog (StarCatalog module):
            StarCatalog class object (only retained if keepStarCatalog is True)
        PlanetPopulation (PlanetPopulation module):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        PostProcessing (PostProcessing module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        TargetList (TargetList module):
            TargetList class object
        SimulatedUniverse (SimulatedUniverse module):
            SimulatedUniverse class object
        Observatory (Observatory module):
            Observatory class object
        TimeKeeping (TimeKeeping module):
            TimeKeeping class object
        fullSpectra (boolean ndarray):
            Indicates if planet spectra have been captured
        partialSpectra (boolean ndarray):
            Indicates if planet partial spectra have been captured
        propagTimes (astropy Quantity array):
            Contains the last time the stellar system was propagated in units of day
        lastObsTimes (astropy Quantity array):
            Contains the last observation start time for future completeness update 
            in units of day
        starVisits (integer ndarray):
            Contains the number of times each target was visited
        starRevisit (float nx2 ndarray):
            Contains indices of targets to revisit and revisit times 
            of these targets in units of day
        starExtended (integer ndarray):
            Contains indices of targets with detected planets, updated throughout 
            the mission
        lastDetected (float nx4 ndarray):
            For each target, contains 4 lists with planets' detected status (boolean),
            exozodi brightness (in units of 1/arcsec2), delta magnitude, 
            and working angles (in units of arcsec)
        DRM (list of dicts):
            Design Reference Mission, contains the results of a survey simulation
        ntFlux (integer):
            Observation time sampling, to determine the integration time interval
        nVisitsMax (integer):
            Maximum number of observations (in detection mode) per star.
        charMargin (float):
            Integration time margin for characterization
        seed (integer):
            Random seed used to make all random number generation reproducible
        WAint (astropy Quantity array):
            Working angle used for integration time calculation in units of arcsec
        dMagint (float ndarray):
            Delta magnitude used for integration time calculation
        scaleWAdMag (bool):
            If True, rescale dMagint and WAint for all stars based on luminosity and 
            to ensure that WA is within the IWA/OWA. Defaults False.
        
    """

    _modtype = 'SurveySimulation'
    _outspec = {}

    def __init__(self, scriptfile=None, ntFlux=1, nVisitsMax=5, charMargin=0.15, 
            WAint=None, dMagint=None, dt_max=1., scaleWAdMag=False, **specs):
        
        # if a script file is provided read it in. If not set, assumes that 
        # dictionary has been passed through specs.
        if scriptfile is not None:
            import json, os.path
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            
            try:
                script = open(scriptfile).read()
                specs.update(json.loads(script))
            except ValueError:
                sys.stderr.write("Script file `%s' is not valid JSON."%scriptfile)
                # must re-raise, or the error will be masked 
                raise
            except:
                sys.stderr.write("Unexpected error while reading specs file: " \
                        + sys.exc_info()[0])
                raise
            
            # modules array must be present
            if 'modules' not in specs.keys():
                raise ValueError("No modules field found in script.")
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # mission simulation logger
        self.logger = specs.get('logger', logging.getLogger(__name__))
       
        # set up numpy random number (generate it if not in specs)
        self.seed = int(specs.get('seed', py_random.randint(1, 1e9)))
        self.vprint('Numpy random seed is: %s'%self.seed)
        np.random.seed(self.seed)

        # if any of the modules is a string, assume that they are all strings 
        # and we need to initalize
        if isinstance(specs['modules'].itervalues().next(), basestring):
            
            # import desired module names (prototype or specific)
            self.SimulatedUniverse = get_module(specs['modules']['SimulatedUniverse'],
                    'SimulatedUniverse')(**specs)
            self.Observatory = get_module(specs['modules']['Observatory'],
                    'Observatory')(**specs)
            self.TimeKeeping = get_module(specs['modules']['TimeKeeping'],
                    'TimeKeeping')(**specs)
            
            # bring inherited class objects to top level of Survey Simulation
            SU = self.SimulatedUniverse
            self.StarCatalog = SU.StarCatalog
            self.PlanetPopulation = SU.PlanetPopulation
            self.PlanetPhysicalModel = SU.PlanetPhysicalModel
            self.OpticalSystem = SU.OpticalSystem
            self.ZodiacalLight = SU.ZodiacalLight
            self.BackgroundSources = SU.BackgroundSources
            self.PostProcessing = SU.PostProcessing
            self.Completeness = SU.Completeness
            self.TargetList = SU.TargetList
        
        else:
            # these are the modules that must be present if passing instantiated objects
            neededObjMods = ['PlanetPopulation',
                          'PlanetPhysicalModel',
                          'OpticalSystem',
                          'ZodiacalLight',
                          'BackgroundSources',
                          'PostProcessing',
                          'Completeness',
                          'TargetList',
                          'SimulatedUniverse',
                          'Observatory',
                          'TimeKeeping']
            
            # ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs['modules'].keys():
                    raise ValueError("%s module is required but was not provided."%modName)
            
            for modName in specs['modules'].keys():
                assert (specs['modules'][modName]._modtype == modName), \
                        "Provided instance of %s has incorrect modtype."%modName
                
                setattr(self, modName, specs['modules'][modName])
        
        # create a dictionary of all modules, except StarCatalog
        self.modules = {}
        self.modules['PlanetPopulation'] = self.PlanetPopulation
        self.modules['PlanetPhysicalModel'] = self.PlanetPhysicalModel
        self.modules['OpticalSystem'] = self.OpticalSystem
        self.modules['ZodiacalLight'] = self.ZodiacalLight
        self.modules['BackgroundSources'] = self.BackgroundSources
        self.modules['PostProcessing'] = self.PostProcessing
        self.modules['Completeness'] = self.Completeness
        self.modules['TargetList'] = self.TargetList
        self.modules['SimulatedUniverse'] = self.SimulatedUniverse
        self.modules['Observatory'] = self.Observatory
        self.modules['TimeKeeping'] = self.TimeKeeping
        
        # observation time sampling
        self.ntFlux = int(ntFlux)
        # maximum number of observations per star
        self.nVisitsMax = int(nVisitsMax)
        # integration time margin for characterization
        self.charMargin = float(charMargin)
        # maximum time for revisit window    
        self.dt_max = dt_max*u.week
        
        # populate outspec with all SurveySimulation scalar attributes
        for att in self.__dict__.keys():
            if att not in ['vprint', 'logger', 'StarCatalog', 'modules'] + self.modules.keys():
                self._outspec[att] = self.__dict__[att]
        
        # load the dMag and WA values for integration:
        # - dMagint defaults to the completeness limiting delta magnitude
        # - WAint defaults to the detection mode IWA-OWA midpoint
        # If inputs are scalars, save scalars to outspec, otherwise save full lists
        Comp = self.Completeness
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        mode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]

        if dMagint is None:
            dMagint = Comp.dMagLim 
        if WAint is None:
            WAint = 2.*mode['IWA'] if np.isinf(mode['OWA']) else (mode['IWA'] + mode['OWA'])/2.
            WAint = WAint.to('arcsec')
        
        self.dMagint = np.array(dMagint,dtype=float,ndmin=1)
        self.WAint = np.array(WAint,dtype=float,ndmin=1)*u.arcsec

        if len(self.dMagint) is 1:
            self._outspec['dMagint'] = self.dMagint[0]
            self.dMagint = np.array([self.dMagint[0]]*TL.nStars)
        else:
            assert (len(self.dMagint) == TL.nStars), \
                    "Input dMagint array doesn't match number of target stars."
            self._outspec['dMagint'] = self.dMagint
        
        if len(self.WAint) is 1:
            self._outspec['WAint'] = self.WAint[0].to('arcsec').value
            self.WAint = np.array([self.WAint[0].value]*TL.nStars)*self.WAint.unit
        else:
            assert (len(self.WAint) == TL.nStars), \
                    "Input WAint array doesn't match number of target stars."
            self._outspec['WAint'] = self.WAint.to('arcsec').value
        
        #if requested, rescale based on luminosities and mode limits
        if scaleWAdMag:
            for i,Lstar in enumerate(TL.L):
                if (Lstar < 1.6) and (Lstar > 0):
                   self.dMagint[i] = Comp.dMagLim - 0.5 + 2.5 * np.log10(Lstar)
                else:
                    self.dMagint[i] = Comp.dMagLim

                EEID = ((np.sqrt(Lstar)*u.AU/TL.dist[i]).decompose()*u.rad).to(u.arcsec)
                if EEID < mode['IWA']:
                    EEID = mode['IWA']
                elif EEID > mode['OWA']:
                    EEID = mode['OWA']

                self.WAint[i] = EEID
        self._outspec['scaleWAdMag'] = scaleWAdMag 

        # initialize arrays updated in run_sim()
        self.DRM = []
        self.fullSpectra = np.zeros(SU.nPlans, dtype=int)
        self.partialSpectra = np.zeros(SU.nPlans, dtype=int)
        self.propagTimes = np.zeros(TL.nStars)*u.d
        self.lastObsTimes = np.zeros(TL.nStars)*u.d
        self.starVisits = np.zeros(TL.nStars, dtype=int)#contains the number of times each star was visited
        self.starRevisit = np.array([])
        self.starExtended = np.array([], dtype=int)
        self.lastDetected = np.empty((TL.nStars, 4), dtype=object)

        #Generate File Hashnames and loction
        self.cachefname = self.generateHashfName(specs)

    def __str__(self):
        """String representation of the Survey Simulation object
        
        When the command 'print' is used on the Survey Simulation object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Survey Simulation class object attributes'

    def run_sim(self):
        """Performs the survey simulation 
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter == True:
            self.currentSep = Obs.occulterSep
        
        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], allModes)
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]
        
        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber + 1)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        cnt = 0
        while not TK.mission_is_over():
            
            # save the start time of this observation (BEFORE any OH/settling/slew time)
            TK.obsStart = TK.currentTimeNorm.to('day')
            
            # acquire the NEXT TARGET star index and create DRM
            DRM, sInd, det_intTime = self.next_target(sInd, det_mode)
            assert det_intTime != 0, "Integration time can't be 0."

            if sInd is not None:
                cnt += 1
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and len(self.starExtended) == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.unique(np.append(self.starExtended,
                                    self.DRM[i]['star_ind']))
                
                # beginning of observation, start to populate DRM
                DRM['star_ind'] = sInd
                DRM['star_name'] = TL.Name[sInd]
                DRM['arrival_time'] = TK.currentTimeNorm.to('day')
                DRM['OB_nb'] = TK.OBnumber + 1
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int)
                log_obs = ('  Observation #%s, star ind %s (of %s) with %s planet(s), ' \
                        + 'mission time: %s')%(cnt, sInd, TL.nStars, len(pInds), 
                        TK.obsStart.round(2))
                self.logger.info(log_obs)
                self.vprint(log_obs)
                
                # PERFORM DETECTION and populate revisit list attribute
                detected, det_fZ, det_systemParams, det_SNR, FA = \
                        self.observation_detection(sInd, det_intTime, det_mode)
                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime, 'det')
                # populate the DRM with detection results
                DRM['det_time'] = det_intTime.to('day')
                DRM['det_status'] = detected
                DRM['det_SNR'] = det_SNR
                DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                DRM['det_params'] = det_systemParams
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                if char_mode['SNR'] not in [0, np.inf]:
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                            self.observation_characterization(sInd, char_mode)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if FA else len(pInds)
                    characterized = np.zeros(lenChar, dtype=float)
                    char_SNR = np.zeros(lenChar, dtype=float)
                    char_fZ = 0./u.arcsec**2
                    char_systemParams = SU.dump_system_params(sInd)
                assert char_intTime != 0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter == True and char_intTime is not None:
                    DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
                # populate the DRM with characterization results
                DRM['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                DRM['char_status'] = characterized[:-1] if FA else characterized
                DRM['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                DRM['char_fZ'] = char_fZ.to('1/arcsec2')
                DRM['char_params'] = char_systemParams
                # populate the DRM with FA results
                DRM['FA_det_status'] = int(FA)
                DRM['FA_char_status'] = characterized[-1] if FA else 0
                DRM['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                DRM['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 \
                        if FA else 0./u.arcsec**2
                DRM['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                DRM['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec \
                        if FA else 0.*u.arcsec
                
                # populate the DRM with observation modes
                DRM['det_mode'] = dict(det_mode)
                del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                DRM['char_mode'] = dict(char_mode)
                del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                
                # append result values to self.DRM
                self.DRM.append(DRM)
                
                # calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.to('day')
                
                # with prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration):
                    obsLength = (TK.obsEnd - TK.obsStart).to('day')
                    TK.next_observing_block(dt=obsLength)
                
                # with occulter, if spacecraft fuel is depleted, exit loop
                if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                    self.vprint('Total fuel mass exceeded at %s'%TK.obsEnd.round(2))
                    break
        
        else:
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            print(log_end)

    def next_target(self, old_sInd, mode):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            mode (dict):
                Selected observing mode for detection
                
        Returns:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            sInd (integer):
                Index of next target star. Defaults to None.
            intTime (astropy Quantity):
                Selected star integration time for detection in units of day. 
                Defaults to None.
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # create DRM
        DRM = {}
        
        # allocate settling time + overhead time
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        
        # now, start to look for available targets
        cnt = 0
        while not TK.mission_is_over():
            # 1. initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            intTimes = np.zeros(TL.nStars)*u.d
            sInds = np.arange(TL.nStars)
            
            # 2. find spacecraft orbital START positions (if occulter, positions 
            # differ for each star) and filter out unavailable targets
            sd = None
            if OS.haveOcculter == True:
                sd,slewTimes = Obs.calculate_slewTimes(TL,old_sInd,sInds,TK.currentTimeAbs)  
                dV = Obs.calculate_dV(Obs.constTOF.value,TL,old_sInd,sInds,TK.currentTimeAbs)
                sInds = sInds[np.where(dV.value < Obs.dVmax.value)]
                
            # start times, including slew times
            startTimes = TK.currentTimeAbs + slewTimes
            startTimesNorm = TK.currentTimeNorm + slewTimes
            # indices of observable stars
            kogoodStart = Obs.keepout(TL, sInds, startTimes, mode)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 3. filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            sInds = self.revisitFilter(sInds,TK.currentTimeNorm)#tmpCurrentTimeNorm)

            # 4. calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            if len(sInds) > 0:  
                intTimes[sInds] = self.calc_targ_intTime(sInds,startTimes[sInds],mode)

                totTimes = intTimes*mode['timeMultiplier']
                # end times
                endTimes = startTimes + totTimes
                endTimesNorm = startTimesNorm + totTimes
                # indices of observable stars
                sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 5. find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(sInds) > 0 and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, sInds, endTimes[sInds], mode)
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 6. choose best target from remaining
            if len(sInds) > 0:
                # choose sInd of next target
                sInd = self.choose_next_target(old_sInd, sInds, slewTimes, intTimes[sInds])
                #Should Choose Next Target decide there are no stars it wishes to observe at this time.
                if sInd == None:
                    TK.allocate_time(TK.waitTime)
                    intTime = None
                    self.vprint('There are no stars Choose Next Target would like to Observe. Waiting 1d')
                    continue
                # store selected star integration time
                intTime = intTimes[sInd]
                break
            
            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.allocate_time(TK.waitTime*TK.waitMultiple**cnt)
                cnt += 1
            
        else:
            return DRM, None, None
        
        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]
        
        # populate DRM with occulter related values
        if OS.haveOcculter == True:
            DRM = Obs.log_occulterResults(DRM,slewTimes[sInd],sInd,sd[sInd],dV[sInd])
            # update current time by adding slew time for the chosen target
            TK.allocate_time(slewTimes[sInd])
            if TK.mission_is_over():
                return DRM, None, None
        
        return DRM, sInd, intTime

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Helper method for next_target to aid in overloading for alternative implementations.

        Given a subset of targets, calculate their integration times given the
        start of observation time.

        Prototype just calculates integration times for fixed contrast depth.  

        Note: next_target filter will discard targets with zero integration times.
        
        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.  
                must be of the same size as sInds 
            mode (dict):
                Selected observing mode for detection

        Returns:
            intTimes (astropy Quantity array):
                Integration times for detection 
                same dimension as sInds
        """
 
        # assumed values for detection
        fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds, startTimes, mode)
        fEZ = self.ZodiacalLight.fEZ0
        dMag = self.dMagint[sInds]
        WA = self.WAint[sInds]
        intTimes = self.OpticalSystem.calc_intTime(self.TargetList, sInds, fZ, fEZ, dMag, WA, mode)
        
        return intTimes

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Helper method for method next_target to simplify alternative implementations.
        
        Given a subset of targets (pre-filtered by method next_target or some 
        other means), select the best next one. The prototype uses completeness 
        as the sole heuristic.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        # choose target with maximum completeness
        sInd = np.random.choice(sInds[comps == max(comps)])
        
        return sInd

    def observation_detection(self, sInd, intTime, mode):
        """Determines SNR and detection status for a given integration time 
        for detetion. Also updates the lastDetected and starRevisit lists.
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            intTime (astropy Quantity):
                Selected star integration time for detection in units of day. 
                Defaults to None.
            mode (dict):
                Selected observing mode for detection
        
        Returns:
            detected (integer ndarray):
                Detection status for each planet orbiting the observed target star:
                1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the 
                duration of the integration
            SNR (float ndarray):
                Detection signal-to-noise ratio of the observable planets
            FA (boolean):
                False alarm (false positive) boolean
        
        """
        
        PPop = self.PlanetPopulation
        Comp = self.Completeness
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPro = self.PostProcessing
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # initialize outputs
        detected = np.array([], dtype=int)
        fZ = 0./u.arcsec**2
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros(len(pInds))
        
        # if any planet, calculate SNR
        if len(pInds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros(self.ntFlux)/u.arcsec**2
            systemParamss = np.empty(self.ntFlux, dtype='object')
            Ss = np.zeros((self.ntFlux, len(pInds)))
            Ns = np.zeros((self.ntFlux, len(pInds)))
            # integrate the signal (planet flux) and noise
            dt = intTime/self.ntFlux
            for i in range(self.ntFlux):
                # allocate first half of dt
                TK.allocate_time(dt/2.)
                # calculate current zodiacal light brightness
                fZs[i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                # propagate the system to match up with current time
                SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                self.propagTimes[sInd] = TK.currentTimeNorm
                # save planet parameters
                systemParamss[i] = SU.dump_system_params(sInd)
                # calculate signal and noise (electron count rates)
                Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, pInds, dt, mode, 
                        fZ=fZs[i])
                # allocate second half of dt
                TK.allocate_time(dt/2.)
            
            # average output parameters
            fZ = np.mean(fZs)
            systemParams = {key: sum([systemParamss[x][key]
                    for x in range(self.ntFlux)])/float(self.ntFlux)
                    for key in sorted(systemParamss[0])}
            # calculate SNR
            S = Ss.sum(0)
            N = Ns.sum(0)
            SNR[N > 0] = S[N > 0]/N[N > 0]
            # allocate extra time for timeMultiplier
            extraTime = intTime*(mode['timeMultiplier'] - 1)
            TK.allocate_time(extraTime)
        
        # if no planet, just save zodiacal brightness in the middle of the integration
        else:
            totTime = intTime*(mode['timeMultiplier'])
            TK.allocate_time(totTime/2.)
            fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
            TK.allocate_time(totTime/2.)
        
        # find out if a false positive (false alarm) or any false negative 
        # (missed detections) have occurred
        FA, MD = PPro.det_occur(SNR, mode, TL, sInd, intTime)
        
        # populate detection status array 
        # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
        if len(pInds) > 0:
            detected = (~MD).astype(int)
            WA = np.array([systemParamss[x]['WA'].to('arcsec').value 
                    for x in range(len(systemParamss))])*u.arcsec
            detected[np.all(WA < mode['IWA'], 0)] = -1
            detected[np.all(WA > mode['OWA'], 0)] = -2
            
        # if planets are detected, calculate the minimum apparent separation
        smin = None
        det = (detected == 1)#If any of the planets around the star have been detected
        if np.any(det):
            smin = np.min(SU.s[pInds[det]])
            log_det = '   - Detected planet inds %s (%s/%s)'%(pInds[det], 
                    len(pInds[det]), len(pInds))
            self.logger.info(log_det)
            self.vprint(log_det)
        
        # populate the lastDetected array by storing det, fEZ, dMag, and WA
        self.lastDetected[sInd,:] = [det, systemParams['fEZ'].to('1/arcsec2').value, 
                    systemParams['dMag'], systemParams['WA'].to('arcsec').value]
        
        # in case of a FA, generate a random delta mag (between PPro.FAdMag0 and
        # Comp.dMagLim) and working angle (between IWA and min(OWA, a_max))
        if FA == True:
            WA = np.random.uniform(mode['IWA'].to('arcsec').value, np.minimum(mode['OWA'],
                    np.arctan(max(PPop.arange)/TL.dist[sInd])).to('arcsec').value)*u.arcsec
            dMag = np.random.uniform(PPro.FAdMag0(WA), Comp.dMagLim)
            self.lastDetected[sInd,0] = np.append(self.lastDetected[sInd,0], True)
            self.lastDetected[sInd,1] = np.append(self.lastDetected[sInd,1], 
                    ZL.fEZ0.to('1/arcsec2').value)
            self.lastDetected[sInd,2] = np.append(self.lastDetected[sInd,2], dMag)
            self.lastDetected[sInd,3] = np.append(self.lastDetected[sInd,3], 
                    WA.to('arcsec').value)
            sminFA = np.tan(WA)*TL.dist[sInd].to('AU')
            smin = np.minimum(smin, sminFA) if smin is not None else sminFA
            log_FA = '   - False Alarm (WA=%s, dMag=%s)'%(np.round(WA, 3), round(dMag, 1))
            self.logger.info(log_FA)
            self.vprint(log_FA)
        
        #Schedule Target Revisit
        self.scheduleRevisit(sInd,smin,det,pInds)

        return detected.astype(int), fZ, systemParams, SNR, FA

    def scheduleRevisit(self,sInd,smin,det,pInds):
        """A Helper Method for scheduling revisits after observation detection
        Args:
            sInd - sInd of the star just detected
            smin - minimum separation of the planet to star of planet just detected
            det - 
            pInds - Indices of planets around target star
        Return:
            updates self.starRevisit attribute
        """
        TK = self.TimeKeeping
        TL = self.TargetList
        SU = self.SimulatedUniverse
        # in both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        Ms = TL.MsTrue[sInd]
        if smin is not None:#smin is None if no planet was detected
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + 0.75*T

        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:#If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])#initialize sterRevisit
        else:
            revInd = np.where(self.starRevisit[:,0] == sInd)[0]#indices of the first column of the starRevisit list containing sInd 
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd,1] = revisit[1]#over

    def observation_characterization(self, sInd, mode):
        """Finds if characterizations are possible and relevant information
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization
        
        Returns:
            characterized (integer list):
                Characterization status for each planet orbiting the observed 
                target star including False Alarm if any, where 1 is full spectrum, 
                -1 partial spectrum, and 0 not characterized
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the 
                duration of the integration
            SNR (float ndarray):
                Characterization signal-to-noise ratio of the observable planets. 
                Defaults to None.
            intTime (astropy Quantity):
                Selected star characterization time in units of day. Defaults to None.
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]
        FA = (len(det) == len(pInds) + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]
        
        # initialize outputs, and check if there's anything (planet or FA) to characterize
        characterized = np.zeros(len(det), dtype=int)
        fZ = 0./u.arcsec**2
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0: # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        if (FA == False): # only true planets, no FA
            tochar = (self.fullSpectra[pIndsDet] == 0)
        else: # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)
        
        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs + mode['syst']['ohTime']
            startTimeNorm = TK.currentTimeNorm + mode['syst']['ohTime']
            # planets to characterize
            tochar[tochar] = Obs.keepout(TL, sInd, startTime, mode)
        
        # 2/ if any planet to characterize, find the characterization times
        # at the detected fEZ, dMag, and WA
        if np.any(tochar):
            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            fEZ = self.lastDetected[sInd,1][det][tochar]/u.arcsec**2
            dMag = self.lastDetected[sInd,2][det][tochar]
            WA = self.lastDetected[sInd,3][det][tochar]*u.arcsec
            intTimes = np.zeros(len(tochar))*u.day
            intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            # add a predetermined margin to the integration times
            intTimes = intTimes*(1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes*(mode['timeMultiplier'])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = ((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                    (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))
        
        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            tochar[tochar] = Obs.keepout(TL, sInd, endTimes[tochar], mode)
        
        # 4/ if yes, allocate the overhead time, and perform the characterization 
        # for the maximum char time
        if np.any(tochar):
            TK.allocate_time(mode['syst']['ohTime'])
            intTime = np.max(intTimes[tochar])
            pIndsChar = pIndsDet[tochar]
            log_char = '   - Charact. planet inds %s (%s/%s detected)'%(pIndsChar, 
                    len(pIndsChar), len(pIndsDet))
            self.logger.info(log_char)
            self.vprint(log_char)
            
            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux)/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime/self.ntFlux
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    TK.allocate_time(dt/2.)
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                            fZ=fZs[i])
                    # allocate second half of dt
                    TK.allocate_time(dt/2.)
                
                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {key: sum([systemParamss[x][key]
                        for x in range(self.ntFlux)])/float(self.ntFlux)
                        for key in sorted(systemParamss[0])}
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0]/N[N > 0]
                # allocate extra time for timeMultiplier
                extraTime = intTime*(mode['timeMultiplier'] - 1)
                TK.allocate_time(extraTime)
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                TK.allocate_time(totTime/2.)
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime/2.)
            
            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][-1]
                WA = self.lastDetected[sInd,3][-1]*u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p*intTime).decompose().value
                N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
                SNRfa = S/N if N > 0 else 0.
            
            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)
            
            # now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            char = (SNR >= mode['SNR'])
            # initialize with full spectra
            characterized = char.astype(int)
            WAchar = self.lastDetected[sInd,3][char]*u.arcsec
            # find the current WAs of characterized planets
            WAs = systemParams['WA']
            if FA:
                WAs = np.append(WAs, self.lastDetected[sInd,3][-1]*u.arcsec)
            # check for partial spectra
            IWA_max = mode['IWA']*(1 + mode['BW']/2.)
            OWA_min = mode['OWA']*(1 - mode['BW']/2.)
            char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1
        
        return characterized.astype(int), fZ, systemParams, SNR, intTime

    def calc_signal_noise(self, sInd, pInds, t_int, mode, fZ=None, fEZ=None, dMag=None, WA=None):
        """Calculates the signal and noise fluxes for a given time interval. Called
        by observation_detection and observation_characterization methods in the 
        SurveySimulation module.
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            t_int (astropy Quantity):
                Integration time interval in units of day
            pInds (integer):
                Integer indices of the planets of interest
            mode (dict):
                Selected observing mode (from OpticalSystem)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (©):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
        
        Returns:
            Signal (float)
                Counts of signal
            Noise (float)
                Counts of background noise variance
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # calculate optional parameters if not provided
        fZ = fZ if fZ else ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)
        fEZ = fEZ if fEZ else SU.fEZ[pInds]
        dMag = dMag if dMag else SU.dMag[pInds]
        WA = WA if WA else SU.WA[pInds]
        
        # initialize Signal and Noise arrays
        Signal = np.zeros(len(pInds))
        Noise = np.zeros(len(pInds))
        
        # find observable planets wrt IWA-OWA range
        obs = (WA > mode['IWA']) & (WA < mode['OWA'])
        
        if np.any(obs):
            # find electron counts
            C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ[obs], dMag[obs], WA[obs], mode)
            # calculate signal and noise levels (based on Nemati14 formula)
            Signal[obs] = (C_p*t_int).decompose().value
            Noise[obs] = np.sqrt((C_b*t_int + (C_sp*t_int)**2).decompose().value)
        
        return Signal, Noise

    def update_occulter_mass(self, DRM, sInd, t_int, skMode):
        """Updates the occulter wet mass in the Observatory module, and stores all 
        the occulter related values in the DRM array.
        
        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            sInd (integer):
                Integer index of the star of interest
            t_int (astropy Quantity):
                Selected star integration time (for detection or characterization)
                in units of day
            skMode (string):
                Station keeping observing mode type ('det' or 'char')
                
        Returns:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
        
        """
        
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        assert skMode in ('det', 'char'), "Observing mode type must be 'det' or 'char'."
        
        #decrement mass for station-keeping
        dF_lateral, dF_axial, intMdot, mass_used, deltaV = Obs.mass_dec_sk(TL, \
                sInd, TK.currentTimeAbs, t_int)
        
        DRM[skMode + '_dV'] = deltaV.to('m/s')
        DRM[skMode + '_mass_used'] = mass_used.to('kg')
        DRM[skMode + '_dF_lateral'] = dF_lateral.to('N')
        DRM[skMode + '_dF_axial'] = dF_axial.to('N')
        # update spacecraft mass
        Obs.scMass = Obs.scMass - mass_used
        DRM['scMass'] = Obs.scMass.to('kg')
        
        return DRM

    def reset_sim(self, genNewPlanets=True, rewindPlanets=True):
        """Performs a full reset of the current simulation by:
        
        1) Re-initializing the TimeKeeping object with its own outspec
        
        2) If genNewPlanets is True (default) then it will also generate all new 
        planets based on the original input specification. If genNewPlanets is False, 
        then the original planets will remain. Setting to True forces rewindPlanets to
        be True as well.

        3) If rewindPlanets is True (default), then the current set of planets will be 
        reset to their original orbital phases. If both genNewPlanets and rewindPlanet
        are False, the original planets will be retained and will not be rewound to their 
        initial starting locations (i.e., all systems will remain at the times they 
        were at the end of the last run, thereby effectively randomizing planet phases.

        4) Re-initializing the SurveySimulation object, including resetting the DRM to [].
        The random seed will be reset as well.
        
        """
        
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
       
        # re-initialize SurveySimulation arrays
        specs = self._outspec
        specs['modules'] = self.modules
        if 'seed' in specs:
            specs.pop('seed')
        self.__init__(**specs)

        # reset mission time and observatory parameters
        TK.__init__(**TK._outspec)
        self.Observatory.__init__(**self.Observatory._outspec)
        
        # generate new planets if requested (default)
        if genNewPlanets:
            SU.gen_physical_properties(**SU._outspec)
            rewindPlanets = True
        # re-initialize systems if requested (default)
        if rewindPlanets:
            SU.init_systems()

        self.vprint("Simulation reset.")

    def genOutSpec(self, tofile=None):
        """Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.
        
        Args:
           tofile (string):
                Name of the file containing all output specifications (outspecs).
                Default to None.
                
        Returns:
            out (dictionary):
                Dictionary containing additional user specification values and 
                desired module names.
        
        """
        
        # start with a copy of MissionSim _outspec
        out = copy.copy(self._outspec)
        
        # add in all modules _outspec's
        for module in self.modules.values():
            out.update(module._outspec)
        
        # add in the specific module names used
        out['modules'] = {}
        for (mod_name, module) in self.modules.items():
            # find the module file 
            mod_name_full = module.__module__
            if mod_name_full.startswith('EXOSIMS'):
                # take just its short name if it is in EXOSIMS
                mod_name_short = mod_name_full.split('.')[-1]
            else:
                # take its full path if it is not in EXOSIMS - changing .pyc -> .py
                mod_name_short = re.sub('\.pyc$', '.py',
                        inspect.getfile(module.__class__))
            out['modules'][mod_name] = mod_name_short
        # add catalog name
        out['modules']['StarCatalog'] = self.StarCatalog
        
        # add in the SVN/Git revision
        path = os.path.split(inspect.getfile(self.__class__))[0]
        path = os.path.split(os.path.split(path)[0])[0]
        #handle case where EXOSIMS was imported from the working directory
        if path is '':
            path = os.getcwd()
        #comm = "git -C " + path + " log -1"
        comm = "git --git-dir=%s --work-tree=%s log -1"%(os.path.join(path,".git"),path)
        rev = subprocess.Popen(comm, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,shell=True)
        (gitRev, err) = rev.communicate()
        if isinstance(gitRev, basestring) & (len(gitRev) > 0):
            tmp = re.compile('\S*(commit [0-9a-fA-F]+)\n[\s\S]*Date: ([\S ]*)\n') \
                    .match(gitRev)
            if tmp:
                out['Revision'] = "Github " + tmp.groups()[0] + " " + tmp.groups()[1]
        else:
            rev = subprocess.Popen("svn info " + path + \
                    "| grep \"Revision\" | awk '{print $2}'", stdout=subprocess.PIPE,
                    shell=True)
            (svnRev, err) = rev.communicate()
            if isinstance(svnRev, basestring) & (len(svnRev) > 0):
                out['Revision'] = "SVN revision is " + svnRev[:-1]
            else: 
                out['Revision'] = "Not a valid Github or SVN revision."
        
        # dump to file
        if tofile is not None:
            with open(tofile, 'w') as outfile:
                json.dump(out, outfile, sort_keys=True, indent=4, ensure_ascii=False,
                        separators=(',', ': '), default=array_encoder)
        
        return out

    def generateHashfName(self, specs):
        """Generate cached file Hashname

        Args:
            specs
                The json script elements of the simulation to be run

        Returns:
            cachefname (string)
                a string containing the file location, hashnumber of the cache name based off
                of the completeness to be computed (completeness specs if available else standard module)
        """
        tmp1 = self.Completeness.PlanetPhysicalModel.__class__.__name__
        tmp2 = self.Completeness.PlanetPopulation.__class__.__name__

        cachefname = ''#declares cachefname
        mods =  ['Completeness','TargetList','OpticalSystem']#modules to look at
        cachefname += str(tmp2)#Planet Pop
        cachefname += str(tmp1)#Planet Physical Model
        for mod in mods: cachefname += self.modules[mod].__module__.split(".")[-1]#add module name to end of cachefname?
        cachefname += hashlib.md5(str(self.TargetList.Name)+str(self.TargetList.tint0.to(u.d).value)).hexdigest()#turn cachefname into hashlib
        fileloc = os.path.split(inspect.getfile(self.__class__))[0]
        cachefname = os.path.join(fileloc,cachefname+os.extsep)#join into filepath and fname
        #Needs file terminator (.starkt0, .t0, etc) appended done by each individual use case.
        ##########################################################
        return cachefname

    def revisitFilter(self,sInds,tmpCurrentTimeNorm):
        """Helper method for Overloading Revisit Filtering

        Args:
            sInds - indices of stars still in observation list
            tmpCurrentTimeNorm (MJD) - the simulation time after overhead was added in MJD form
        Returns:
            sInds - indices of stars still in observation list
        """
        tovisit = np.zeros(self.TargetList.nStars, dtype=bool)#tovisit is a boolean array containing the 
        if len(sInds) > 0:#so long as there is at least 1 star left in sInds
            tovisit[sInds] = ((self.starVisits[sInds] == min(self.starVisits[sInds])) \
                    & (self.starVisits[sInds] < self.nVisitsMax))#Checks that no star has exceeded the number of revisits and the indicies of all considered stars have minimum number of observations
            #The above condition should prevent revisits so long as all stars have not been observed
            if self.starRevisit.size != 0:#There is at least one revisit planned in starRevisit
                dt_rev = np.abs(self.starRevisit[:,1]*u.day - tmpCurrentTimeNorm)#absolute temporal spacing between revisit and now.
                ind_rev = [int(x) for x in self.starRevisit[dt_rev < self.dt_max,0] #return indice of all revisits within a threshold dt_max of revisit day
                        if x in sInds]
                tovisit[ind_rev] = (self.starVisits[ind_rev] < self.nVisitsMax)#IF duplicates exist in ind_rev, the second occurence takes priority
            sInds = np.where(tovisit)[0]
        return sInds

def array_encoder(obj):
    r"""Encodes numpy arrays, astropy Times, and astropy Quantities, into JSON.
    
    Called from json.dump for types that it does not already know how to represent,
    like astropy Quantity's, numpy arrays, etc.  The json.dump() method encodes types
    like integers, strings, and lists itself, so this code does not see these types.
    Likewise, this routine can and does return such objects, which is OK as long as 
    they unpack recursively into types for which encoding is known.
    
    """
    
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    if isinstance(obj, Time):
        # astropy Time -> time string
        return obj.fits # isot also makes sense here
    if isinstance(obj, u.quantity.Quantity):
        # note: it is possible to have a numpy ndarray wrapped in a Quantity.
        # NB: alternatively, can return (obj.value, obj.unit.name)
        return obj.value
    if isinstance(obj, SkyCoord):
        return dict(lon=obj.heliocentrictrueecliptic.lon.value,
                    lat=obj.heliocentrictrueecliptic.lat.value,
                    distance=obj.heliocentrictrueecliptic.distance.value)
    if isinstance(obj, (np.ndarray, np.number)):
        # ndarray -> list of numbers
        return obj.tolist()
    if isinstance(obj, (complex, np.complex)):
        # complex -> (real, imag) pair
        return [obj.real, obj.imag]
    if callable(obj):
        # this case occurs for interpolants like PSF and QE
        # We cannot simply "write" the function to JSON, so we make up a string
        # to keep from throwing an error.
        # The fix is simple: when generating the interpolant, add a _outspec attribute
        # to the function (or the lambda), containing (e.g.) the fits filename, or the
        # explicit number -- whatever string was used.  Then, here, check for that 
        # attribute and write it out instead of this dummy string.  (Attributes can
        # be transparently attached to python functions, even lambda's.)
        return 'interpolant_function'
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode()
    # an EXOSIMS object
    if hasattr(obj, '_modtype'):
        return obj.__dict__
    # an object for which no encoding is defined yet
    #   as noted above, ordinary types (lists, ints, floats) do not take this path
    raise ValueError('Could not JSON-encode an object of type %s' % type(obj))
