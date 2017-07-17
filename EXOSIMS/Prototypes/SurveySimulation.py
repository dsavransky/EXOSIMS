# -*- coding: utf-8 -*-
import sys, logging
import numpy as np
import astropy.units as u
import astropy.constants as const
from EXOSIMS.util.get_module import get_module
import time
import json, os.path, copy, re, inspect, subprocess

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
            
    Attributes:
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
            For each target, contains 4 lists with planets' detected status, exozodi 
            brightness (in units of 1/arcsec2), delta magnitude, and working angles 
            (in units of mas)
        DRM (list of dicts):
            The Design Reference Mission, contains the results of a survey simulation
        WAint (astropy Quantity):
            Working angle used for integration time calculation in units of arcsec
        dMagint (astropy Quantity):
            Delta magnitude used for integration time calculation
        nt_flux (integer):
            Observation time sampling, to determine the integration time interval
        
    """

    _modtype = 'SurveySimulation'
    _outspec = {}

    def __init__(self, scriptfile=None, WAint=None, dMagint=None, nt_flux=1, **specs):
        """Initializes Survey Simulation with default values
        
        Input: 
            scriptfile (string):
                JSON script file. If not set, assumes that dictionary has been 
                passed through specs.
                
        """
        
        # mission simulation logger
        self.logger = specs.get('logger', logging.getLogger(__name__))
        
        # if a script file is provided read it in
        if scriptfile is not None:
            import json, os.path
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            
            try:
                script = open(scriptfile).read()
                specs = json.loads(script)
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
        
        # list of simulation results, each item is a dictionary
        self.DRM = []
        
        # initialize arrays updated in run_sim()
        TL = self.TargetList
        SU = self.SimulatedUniverse
        self.fullSpectra = np.zeros(SU.nPlans, dtype=int)
        self.partialSpectra = np.zeros(SU.nPlans, dtype=int)
        self.propagTimes = np.zeros(TL.nStars)*u.d
        self.lastObsTimes = np.zeros(TL.nStars)*u.d
        self.starVisits = np.zeros(TL.nStars, dtype=int)
        self.starRevisit = np.array([])
        self.starExtended = np.array([], dtype=int)
        self.lastDetected = np.empty((TL.nStars, 4), dtype=object)
       
        # populate outspec
        if WAint: self._outspec['WAint'] = WAint
        if dMagint: self._outspec['dMagint'] = dMagint
        if nt_flux: self._outspec['nt_flux'] = nt_flux

        # load the integration values: working angle (WAint), delta magnitude (dMagint)
        # default to detection mode IWA and dMadLim
        # must be of size equal to TargetList.nStars
        OS = self.OpticalSystem
        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        WAint = float(WAint)*u.arcsec if WAint else detMode['IWA']
        self.WAint = np.array([WAint.value]*TL.nStars)*WAint.unit
        dMagint = float(dMagint) if dMagint else OS.dMagLim
        self.dMagint = np.array([dMagint]*TL.nStars)
        
        # observation time sampling (must be an integer)
        self.nt_flux = int(nt_flux)
        


    def __str__(self):
        """String representation of the Survey Simulation object
        
        When the command 'print' is used on the Survey Simulation object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
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
        detMode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], allModes)
        if np.any(spectroModes):
            charMode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            charMode = allModes[0]
        
        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber + 1)
        self.logger.info(log_begin)
        print log_begin
        t0 = time.time()
        sInd = None
        cnt = 0
        while not TK.mission_is_over():
            
            # save the start time of this observation (BEFORE any OH/settling/slew time)
            TK.obsStart = TK.currentTimeNorm.to('day')
            
            # acquire the NEXT TARGET star index and create DRM
            DRM, sInd, t_det = self.next_target(sInd, detMode)
            assert t_det != 0, "Integration time can't be 0."
            
            if sInd is not None:
                cnt += 1
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and len(self.starExtended) == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.unique(np.append(self.starExtended,
                                    self.DRM[i]['star_ind']))
                
                # beginning of observation, start to populate DRM
                DRM['OB#'] = TK.OBnumber+1
                DRM['Obs#'] = cnt
                DRM['star_ind'] = sInd
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()
                log_obs = ('  Observation #%s, target #%s/%s with %s planet(s), ' \
                        + 'mission time: %s')%(cnt, sInd+1, TL.nStars, len(pInds), 
                        TK.obsStart.round(2))
                self.logger.info(log_obs)
                print log_obs
                
                # PERFORM DETECTION and populate revisit list attribute
                # first store fEZ, dMag, WA
                if np.any(pInds):
                    DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['det_dMag'] = SU.dMag[pInds].tolist()
                    DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                detected, detfZ, detSNR, FA = self.observation_detection(sInd, t_det, detMode)
                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, t_det, 'det')
                # populate the DRM with detection results
                DRM['det_time'] = t_det.to('day').value
                DRM['det_status'] = detected
                DRM['det_fZ'] = detfZ.to('1/arcsec2').value
                DRM['det_SNR'] = detSNR
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                # first store fEZ, dMag, WA, and characterization mode
                if np.any(pInds):
                    DRM['char_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['char_dMag'] = SU.dMag[pInds].tolist()
                    DRM['char_WA'] = SU.WA[pInds].to('mas').value.tolist()
                DRM['char_mode'] = dict(charMode)
                del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                characterized, charSNR, t_char = self.observation_characterization(sInd, 
                        charMode)
                assert t_char !=0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter == True and t_char is not None:
                    DRM = self.update_occulter_mass(DRM, sInd, t_char, 'char')
                # if any false alarm, store its characterization status, fEZ, dMag, and WA
                if FA == True:
                    DRM['FA_status'] = characterized.pop()
                    DRM['FA_SNR'] = charSNR.pop()
                    DRM['FA_fEZ'] = self.lastDetected[sInd,1][-1]
                    DRM['FA_dMag'] = self.lastDetected[sInd,2][-1]
                    DRM['FA_WA'] = self.lastDetected[sInd,3][-1]
                # populate the DRM with characterization results
                DRM['char_time'] = t_char.to('day').value if t_char else 0.
                DRM['char_status'] = characterized
                DRM['char_SNR'] = charSNR
                
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
                    print 'Total fuel mass exceeded at %s'%TK.obsEnd.round(2)
                    break
        
        else:
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            print log_end

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
            DRM (array of dicts):
                Contains the results of survey simulation
            sInd (integer):
                Index of next target star. Defaults to None.
            t_det (astropy Quantity):
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
        
        # in case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        if OS.haveOcculter == True:
            ao = Obs.thrust/Obs.scMass
            slewTime_fac = (2.*Obs.occulterSep/np.abs(ao)/(Obs.defburnPortion/2. - 
                    Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        # now, start to look for available targets
        while not TK.mission_is_over():
            # 1/ initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            t_dets = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)
            
            # 2/ find spacecraft orbital START positions (if occulter, positions 
            # differ for each star) and filter out unavailable targets
            sd = None
            if OS.haveOcculter == True:
                # find angle between old and new stars, default to pi/2 for first target
                if old_sInd is None:
                    sd = np.array([np.radians(90)]*TL.nStars)*u.rad
                else:
                    # position vector of previous target star
                    r_old = TL.starprop(old_sInd, TK.currentTimeAbs)[0]
                    u_old = r_old.value/np.linalg.norm(r_old)
                    # position vector of new target stars
                    r_new = TL.starprop(sInds, TK.currentTimeAbs)
                    u_new = (r_new.value.T/np.linalg.norm(r_new, axis=1)).T
                    # angle between old and new stars
                    sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1))*u.rad
                # calculate slew time
                slewTimes = np.sqrt(slewTime_fac*np.sin(sd/2.))
            # save observation start times including, slew times
            startTimes = TK.currentTimeAbs + slewTimes
            # get target indices where keepout angle is good at observation start
            kogoodStart = Obs.keepout(TL, sInds, startTimes, mode)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 3/ calculate integration times for ALL preselected targets, 
            # and filter out t_tots > integration cutoff
            if np.any(sInds):
                # assumed values for detection
                fZ = ZL.fZ(Obs, TL, sInds, startTimes[sInds], mode)
                fEZ = ZL.fEZ0
                dMag = self.dMagint[sInds]
                WA = self.WAint[sInds]
                t_dets[sInds] = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
                t_tots = t_dets*mode['timeMultiplier']
                endTimes = startTimes + t_tots
                endTimesNorm = (endTimes - TK.missionStart).jd*u.day
                sInds = np.where((t_tots > 0) & (t_tots <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 4/ find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if np.any(sInds) and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, sInds, endTimes[sInds], mode)
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 5/ filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if np.any(sInds):
                tovisit[sInds] = (self.starVisits[sInds] == min(self.starVisits[sInds]))
                if self.starRevisit.size != 0:
                    dt_max = 1.*u.week
                    startTimesNorm = (startTimes - TK.missionStart).jd*u.day
                    dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] 
                            if x in sInds]
                    tovisit[ind_rev] = True
                sInds = np.where(tovisit)[0]
            
            # 6/ choose best target from remaining
            if np.any(sInds):
                # choose sInd of next target
                sInd = self.choose_next_target(old_sInd, sInds, slewTimes, t_dets[sInds])
                # store selected star integration time
                t_det = t_dets[sInd]
                break
            
            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.wait()
            
        else:
            return DRM, None, None
        
        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = (startTimes[sInd] - TK.missionStart).jd*u.day
        
        # populate DRM with occulter related values
        if OS.haveOcculter == True:
            # find values related to slew time
            DRM['slew_time'] = slewTimes[sInd].to('day').value
            DRM['slew_angle'] = sd[sInd].to('deg').value
            slew_mass_used = slewTimes[sInd]*Obs.defburnPortion*Obs.flowRate
            DRM['slew_dV'] = (slewTimes[sInd]*ao*Obs.defburnPortion).to('m/s').value
            DRM['slew_mass_used'] = slew_mass_used.to('kg').value
            Obs.scMass = Obs.scMass - slew_mass_used
            DRM['scMass'] = Obs.scMass.to('kg').value
            # update current time by adding slew time for the chosen target
            TK.allocate_time(slewTimes[sInd])
            if TK.mission_is_over():
                return DRM, None, None
        
        return DRM, sInd, t_det

    def choose_next_target(self, old_sInd, sInds, slewTimes, t_dets):
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
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
        # reshape sInds
        sInds = np.array(sInds, ndmin=1)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        # choose target with maximum completeness
        sInd = np.random.choice(sInds[comps == max(comps)])
        
        return sInd

    def observation_detection(self, sInd, t_det, mode):
        """Determines SNR and detection status for a given integration time 
        for detetion. Also updates the lastDetected and starRevisit lists.
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            t_det (astropy Quantity):
                Selected star integration time for detection in units of day. 
                Defaults to None.
            mode (dict):
                Selected observing mode for detection
        
        Returns:
            detected (integer list):
                Detection status for each planet orbiting the observed target star:
                1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
            fZ (astropy Quantity):
                Zodiacal brightness at detection
            SNR (float list):
                Detection signal-to-noise ratio of the observable planets
            FA (boolean):
                False alarm (false positive) boolean
        
        """
        
        PPop = self.PlanetPopulation
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPro = self.PostProcessing
        TL = self.TargetList
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        # find cases with working angles (WA) out of IWA-OWA range
        observable = np.ones(len(pInds), dtype=int)
        if np.any(observable):
            WA = SU.WA[pInds]
            observable[WA < mode['IWA']] = -1
            observable[WA > mode['OWA']] = -2
        
        # now, calculate SNR for any observable planet (within IWA-OWA range)
        obs = (observable == 1)
        if np.any(obs):
            # initialize Signal and Noise arrays
            Signal = np.zeros((self.nt_flux, len(pInds[obs])))
            Noise = np.zeros((self.nt_flux, len(pInds[obs])))
            # integrate the signal (planet flux) and noise
            dt = t_det/self.nt_flux
            for i in range(self.nt_flux):
                s, n, fZ = self.calc_signal_noise(sInd, pInds[obs], dt, mode)
                Signal[i,:] = s
                Noise[i,:] = n
            # calculate SNRobs
            with np.errstate(divide='ignore', invalid='ignore'):
                SNRobs = Signal.sum(0) / Noise.sum(0)
            SNRobs[np.isnan(SNRobs)] = 0.
            # allocate extra time for timeMultiplier
            t_extra = t_det*(mode['timeMultiplier'] - 1)
            TK.allocate_time(t_extra)
        # if no planet, just observe for t_tot (including time multiplier)
        else:
            fZ = 0/u.arcsec**2
            SNRobs = np.array([])
            t_tot = t_det*(mode['timeMultiplier'])
            TK.allocate_time(t_tot)
        
        # find out if a false positive (false alarm) or any false negative 
        # (missed detections) have occurred, and populate detection status array
        FA, MD = PPro.det_occur(SNRobs, mode['SNR'])
        detected = observable
        SNR = np.zeros(len(pInds))
        if np.any(obs):
            detected[obs] = (~MD).astype(int)
            SNR[obs] = SNRobs
        
        # if planets are detected, calculate the minimum apparent separation
        smin = None
        det = (detected == 1)
        if np.any(det):
            smin = np.min(SU.s[pInds[det]])
            log_det = '   - Detected planet(s) %s (%s/%s)'%(pInds[det], 
                    len(pInds[det]), len(pInds))
            self.logger.info(log_det)
            print log_det
        
        # populate the lastDetected array by storing det, fEZ, dMag, and WA
        self.lastDetected[sInd,:] = [det, SU.fEZ[pInds].to('1/arcsec2').value, 
                    SU.dMag[pInds], SU.WA[pInds].to('mas').value]
        
        # in case of a FA, generate a random delta mag (between maxFAfluxratio and
        # dMagLim) and working angle (between IWA and min(OWA, a_max))
        if FA == True:
            WA = np.random.uniform(mode['IWA'].to('mas').value, np.minimum(mode['OWA'],
                    np.arctan(max(PPop.arange)/TL.dist[sInd])).to('mas').value)
            dMag = np.random.uniform(-2.5*np.log10(PPro.maxFAfluxratio(WA*u.mas)), 
                    OS.dMagLim)
            fEZ = ZL.fEZ0.to('1/arcsec2').value
            self.lastDetected[sInd,0] = np.append(self.lastDetected[sInd,0], True)
            self.lastDetected[sInd,1] = np.append(self.lastDetected[sInd,1], fEZ)
            self.lastDetected[sInd,2] = np.append(self.lastDetected[sInd,2], dMag)
            self.lastDetected[sInd,3] = np.append(self.lastDetected[sInd,3], WA)
            sminFA = np.tan(WA*u.mas)*TL.dist[sInd].to('AU')
            smin = np.minimum(smin,sminFA) if smin is not None else sminFA
            log_FA = '   - False Alarm (WA=%s, dMag=%s)'%(int(WA)*u.mas, round(dMag, 1))
            self.logger.info(log_FA)
            print log_FA
        
        # in both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        Ms = TL.MsTrue[sInd]
        if smin is not None:
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
        if self.starRevisit.size == 0:
            self.starRevisit = np.array([revisit])
        else:
            self.starRevisit = np.vstack((self.starRevisit, revisit))
        
        return detected.tolist(), fZ, SNR.tolist(), FA

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
            SNR (float list):
                Characterization signal-to-noise ratio of the observable planets. 
                Defaults to None.
            t_char (astropy Quantity):
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
        # get the last detected planets, and check if there was a FA
        det = self.lastDetected[sInd,0]
        FA = (det.size == pInds.size + 1)
        if FA == True:
            pInds = np.append(pInds, -1)
        
        # initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
        SNR = np.zeros(det.size)
        t_char = None
        if not np.any(pInds):
            return characterized.tolist(), SNR.tolist(), t_char
        
        # look for last detected planets that have not been fully characterized
        tochar = np.zeros(len(det), dtype=bool)
        if (FA == False):
            tochar[det] = (self.fullSpectra[pInds[det]] != 1)
        elif pInds[det].size > 1:
            tochar[det] = np.append((self.fullSpectra[pInds[det][:-1]] != 1), True)
        else:
            tochar[det] = np.array([True])
        
        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            startTimes = TK.currentTimeAbs
            tochar[tochar] = Obs.keepout(TL, sInd, startTimes, mode)
        
        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
            self.propagTimes[sInd] = TK.currentTimeNorm
            # calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(Obs, TL, sInd, startTimes, mode)
            fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            dMag = self.lastDetected[sInd,2][tochar]
            WA = self.lastDetected[sInd,3][tochar]*u.mas
            t_chars = np.zeros(len(pInds))*u.d
            t_chars[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            t_tots = t_chars*(mode['timeMultiplier'])
            # total time must be positive, shorter than integration cut-off,
            # and it must not exceed the Observing Block end time
            startTimesNorm = (startTimes - TK.missionStart).jd*u.day
            tochar = ((t_tots > 0) & (t_tots <= OS.intCutoff) & 
                    (startTimesNorm + t_tots <= TK.OBendTimes[TK.OBnumber]))
        
        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            endTime = startTimes + t_tots[tochar]
            tochar[tochar] = Obs.keepout(TL, sInd, endTime, mode)
        
        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            t_char = np.max(t_chars[tochar])
            pIndsChar = pInds[tochar]
            log_char = '   - Charact. planet(s) %s (%s/%s)'%(pIndsChar, 
                    len(pIndsChar), len(pInds))
            self.logger.info(log_char)
            print log_char
            
            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            if np.any(planinds):
                Signal = np.zeros((self.nt_flux, len(planinds)))
                Noise = np.zeros((self.nt_flux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = t_char/self.nt_flux
                for i in range(self.nt_flux):
                    s,n,_, = self.calc_signal_noise(sInd, planinds, dt, mode)
                    Signal[i,:] = s
                    Noise[i,:] = n
                # calculate SNRobs
                with np.errstate(divide='ignore', invalid='ignore'):
                    SNRobs = Signal.sum(0) / Noise.sum(0)
                # allocate extra time for timeMultiplier
                t_extra = t_char*(mode['timeMultiplier'] - 1)
                TK.allocate_time(t_extra)
            # if only a false alarm, just observe for t_tot including time multiplier
            else:
                SNRobs = np.array([])
                t_tot = t_char*(mode['timeMultiplier'])
                TK.allocate_time(t_tot)
            
            # append the false alarm SNR (if any)
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][-1]
                WA = self.lastDetected[sInd,3][-1]*u.mas
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                SNRfa = (C_p*t_char).decompose().value
                SNRfa[SNRfa > 0] /= np.sqrt(C_b*t_char + (C_sp*t_char)**2) \
                        .decompose().value[SNRfa > 0]
                SNRobs = np.append(SNRobs, SNRfa)
            
            # now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            if np.any(SNRobs):
                SNR[tochar] = SNRobs
                char = (SNR >= mode['SNR'])
                if np.any(SNR):
                    # initialize with partial spectra
                    characterized[char] = -1
                    # check for full spectra
                    WA = self.lastDetected[sInd,3]*u.mas
                    IWA_max = mode['IWA']*(1 + mode['BW']/2.)
                    OWA_min = mode['OWA']*(1 - mode['BW']/2.)
                    char[char] = (WA[char] > IWA_max) & (WA[char] < OWA_min)
                    characterized[char] = 1
                    # encode results in spectra lists
                    partial = pInds[characterized == -1]
                    if np.any(partial != -1):
                        partial = partial[:-1] if partial[-1] == -1 else partial
                        self.partialSpectra[partial] += 1
                    full = pInds[np.where(characterized == 1)[0]]
                    if np.any(full != -1):
                        full = full[:-1] if full[-1] == -1 else full
                        self.fullSpectra[full] += 1
        
        return characterized.tolist(), SNR.tolist(), t_char

    def calc_signal_noise(self, sInd, pInds, t_int, mode):
        """Calculates the signal and noise fluxes for a given time interval. Called
        by observation_detection and observation_characterization methods in the 
        SurveySimulation module.
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            pInds (integer):
                Integer indices of the planets of interest
            t_int (astropy Quantity):
                Integration time interval in units of day
            mode (dict):
                Selected observing mode (from OpticalSystem)
        
        Returns:
            Signal (float)
                Counts of signal
            Noise (float)
                Counts of background noise variance
            fZ (astropy Quantity):
                Zodiacal brightness at detection
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # allocate first half of t_int
        TK.allocate_time(t_int/2.)
        # propagate the system to match up with current time
        SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
        self.propagTimes[sInd] = TK.currentTimeNorm
        # find spacecraft position and ZodiacalLight
        fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)
        # find electron counts for planet, background, and speckle residual 
        C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, SU.fEZ[pInds], SU.dMag[pInds], 
                SU.WA[pInds], mode)
        # calculate signal and noise levels (based on Nemati14 formula)
        Signal = (C_p*t_int).decompose().value
        Noise = np.sqrt((C_b*t_int + (C_sp*t_int)**2).decompose().value)
        # allocate second half of t_int
        TK.allocate_time(t_int/2.)
        
        return Signal, Noise, fZ[0]

    def update_occulter_mass(self, DRM, sInd, t_int, skMode):
        """Updates the occulter wet mass in the Observatory module, and stores all 
        the occulter related values in the DRM array.
        
        Args:
            DRM (dicts):
                Contains the results of survey simulation
            sInd (integer):
                Integer index of the star of interest
            t_int (astropy Quantity):
                Selected star integration time (for detection or characterization)
                in units of day
            skMode (string):
                Station keeping observing mode type
                
        Returns:
            DRM (dicts):
                Contains the results of survey simulation
        
        """
        
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # find disturbance forces on occulter
        dF_lateral, dF_axial = Obs.distForces(TL, sInd, TK.currentTimeAbs)
        # decrement mass for station-keeping
        intMdot, mass_used, deltaV = Obs.mass_dec(dF_lateral, t_int)
        DRM[skMode+'_dV'] = deltaV.to('m/s').value
        DRM[skMode+'_mass_used'] = mass_used.to('kg').value
        DRM[skMode+'_dF_lateral'] = dF_lateral.to('N').value
        DRM[skMode+'_dF_axial'] = dF_axial.to('N').value
        # update spacecraft mass
        Obs.scMass = Obs.scMass - mass_used
        DRM['scMass'] = Obs.scMass.to('kg').value
        
        return DRM

    def reset_sim(self, genNewPlanets=True, rewindPlanets=True):
        """Performs a full reset of the current simulation by:
        
        1) Re-initializing the TimeKeeping object with its own outspec
        
        2) If genNewPlanets is True (default) then it will also generate all new 
        planets based on the original input specification. If genNewPlanets is False, 
        then the original planets will remain, but they will not be rewound to their 
        initial starting locations (i.e., all systems will remain at the times they 
        were at the end of the last run, thereby effectively randomizing planet phases.
        
        3) If rewindPlanets is True (default), then the current set of planets will be 
        reset to their original orbital phases. This has no effect if genNewPlanets is 
        True, but if genNewPlanets is False, will have the effect of resetting the full 
        simulation to its exact original state.
        
        4) Re-initializing the SurveySimulation object, including resetting the DRM to []
        
        """
        
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        
        # reset mission time parameters
        TK.__init__(**TK._outspec)
        # generate new planets if requested (default)
        if genNewPlanets:
            SU.gen_physical_properties(**SU._outspec)
            rewindPlanets = True
        # re-initialize systems if requested (default)
        if rewindPlanets:
            SU.init_systems()
        # re-initialize SurveySimulation arrays
        specs = self._outspec
        specs['modules'] = self.modules
        self.__init__(**specs)
        
        print "Simulation reset."

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
        rev = subprocess.Popen("git -C " + path + " log -1", stdout=subprocess.PIPE,
                shell=True)
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

def array_encoder(obj):
    r"""Encodes numpy arrays, astropy Times, and astropy Quantities, into JSON.
    
    Called from json.dump for types that it does not already know how to represent,
    like astropy Quantity's, numpy arrays, etc.  The json.dump() method encodes types
    like integers, strings, and lists itself, so this code does not see these types.
    Likewise, this routine can and does return such objects, which is OK as long as 
    they unpack recursively into types for which encoding is known.
    
    """
    
    from astropy.time import Time
    if isinstance(obj, Time):
        # astropy Time -> time string
        return obj.fits # isot also makes sense here
    if isinstance(obj, u.quantity.Quantity):
        # note: it is possible to have a numpy ndarray wrapped in a Quantity.
        # NB: alternatively, can return (obj.value, obj.unit.name)
        return obj.value
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
    # nothing worked, bail out
    
    return json.JSONEncoder.default(obj)
