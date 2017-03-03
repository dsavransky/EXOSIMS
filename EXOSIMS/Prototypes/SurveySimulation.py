# -*- coding: utf-8 -*-
import numpy as np
import sys, logging
import astropy.units as u
import astropy.constants as const
from EXOSIMS.util.get_module import get_module

# the EXOSIMS logger
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
        nt_flux (integer):
            Observation time sampling, to determine the integration time interval
        fullSpectra (boolean ndarray):
            Indicates if planet spectra have been captured
        partialSpectra (boolean ndarray):
            Indicates if planet partial spectra have been captured
        starVisits (integer ndarray):
            Contains the number of times each target was visited
        starTimes (astropy Quantity array):
            Contains the last time the star was observed in units of day
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
            Contains the results of survey simulation
        
    """

    _modtype = 'SurveySimulation'
    _outspec = {}

    def __init__(self, nt_flux=1, logLevel='ERROR', scriptfile=None, **specs):
        """Initializes Survey Simulation with default values
        
        Input: 
            nt_flux (integer):
                Observation time sampling, to determine the integration time interval
            logLevel (string):
                Defines a logging level for the logger handler. Valid levels are: INFO, 
                CRITICAL, ERROR, WARNING, DEBUG (case is ignored). Defaults to 'INFO'.
            scriptfile (string):
                JSON script file.  If not set, assumes that 
                dictionary has been passed through specs 
                
        """
        
        # toggle the logging level: INFO, DEBUG, WARNING, ERROR, CRITICAL
        if logLevel.upper() == 'INFO':
            logging.basicConfig(level=logging.INFO)
        elif logLevel.upper() == 'DEBUG':
            logging.basicConfig(level=logging.DEBUG)
        elif logLevel.upper() == 'WARNING':
            logging.basicConfig(level=logging.WARNING)
        elif logLevel.upper() == 'ERROR':
            logging.basicConfig(level=logging.ERROR)
        elif logLevel.upper() == 'CRITICAL':
            logging.basicConfig(level=logging.CRITICAL)
        
        # if a script file is provided read it in
        if scriptfile is not None:
            import json
            import os.path
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            
            try:
                script = open(scriptfile).read()
                specs = json.loads(script)
            except ValueError:
                sys.stderr.write("Error.  Script file `%s' is not valid JSON." % scriptfile)
                # must re-raise, or the error will be masked 
                raise
            except:
                sys.stderr.write("Unexpected error while reading specs file: " + sys.exc_info()[0])
                raise
            
            # modules array must be present
            if 'modules' not in specs.keys():
                raise ValueError("No modules field found in script.")
        
        #if any of the modules is a string, assume that they are all strings and we need to initalize
        if isinstance(specs['modules'].itervalues().next(),basestring):
            
            # import desired module names (prototype or specific)
            self.SimulatedUniverse = get_module(specs['modules'] \
                    ['SimulatedUniverse'],'SimulatedUniverse')(**specs)
            self.Observatory = get_module(specs['modules'] \
                    ['Observatory'],'Observatory')(**specs)
            self.TimeKeeping = get_module(specs['modules'] \
                    ['TimeKeeping'],'TimeKeeping')(**specs)
            
            # bring inherited class objects to top level of Survey Simulation
            SU = self.SimulatedUniverse
            self.PlanetPopulation = SU.PlanetPopulation
            self.PlanetPhysicalModel = SU.PlanetPhysicalModel
            self.OpticalSystem = SU.OpticalSystem
            self.ZodiacalLight = SU.ZodiacalLight
            self.BackgroundSources = SU.BackgroundSources
            self.PostProcessing = SU.PostProcessing
            self.Completeness = SU.Completeness
            self.TargetList = SU.TargetList
        else:
            #these are the modules that must be present if passing instantiated objects
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
            
            #ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs['modules'].keys():
                    raise ValueError("%s module is required but was not provided." % modName)
            
            for modName in specs['modules'].keys():
                assert (specs['modules'][modName]._modtype == modName), \
                "Provided instance of %s has incorrect modtype."%modName
                
                setattr(self, modName, specs['modules'][modName])
        
        # observation time sampling (must be an integer)
        self.nt_flux = int(nt_flux)
        # list of simulation results, each item is a dictionary
        self.DRM = []

    def __str__(self):
        """String representation of the Survey Simulation object
        
        When the command 'print' is used on the Survey Simulation object, this 
        method will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Survey Simulation class object attributes'

    def run_sim(self):
        """Performs the survey simulation 
        
        Returns:
            mission_end (string):
                Messaged printed at the end of a survey simulation.
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        Logger.info('run_sim beginning')
        
        # initialize lists updated later
        self.fullSpectra = np.zeros(SU.nPlans, dtype=int)
        self.partialSpectra = np.zeros(SU.nPlans, dtype=int)
        self.starVisits = np.zeros(TL.nStars,dtype=int)
        self.starTimes = np.zeros(TL.nStars)*u.d
        self.starRevisit = np.array([])
        self.starExtended = np.array([])
        self.lastDetected = np.empty((TL.nStars, 4), dtype=object)
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter == True:
            self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        if np.any(spectroModes):
            charMode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            charMode = OS.observingModes[0]
        
        # loop until mission is finished
        sInd = None
        while not TK.mission_is_over():
            
            # Acquire the NEXT TARGET star index and create DRM
            DRM, sInd, t_det = self.next_target(sInd, detMode)
            
            if sInd is not None:
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)
                
                # Beginning of observation, create DRM and start to populate it
                obsBegin = TK.currentTimeNorm.to('day')
                Logger.info('current time is %r' % obsBegin)
                print 'Current mission time: ', obsBegin
                DRM['star_ind'] = sInd
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()
                
                # PERFORM DETECTION and populate revisit list attribute.
                # First store fEZ, dMag, WA
                if np.any(pInds):
                    DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['det_dMag'] = SU.dMag[pInds].tolist()
                    DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                detected, detSNR, FA = self.observation_detection(sInd, t_det, detMode)
                # Update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, t_det, 'det')
                # Populate the DRM with detection results
                DRM['det_time'] = t_det.to('day').value
                DRM['det_status'] = detected
                DRM['det_SNR'] = detSNR
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute.
                # First store fEZ, dMag, WA, and characterization mode
                if np.any(pInds):
                    DRM['char_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['char_dMag'] = SU.dMag[pInds].tolist()
                    DRM['char_WA'] = SU.WA[pInds].to('mas').value.tolist()
                DRM['char_mode'] = dict(charMode)
                del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                characterized, charSNR, t_char = self.observation_characterization(sInd, charMode)
                # Update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, t_char, 'char')
                # if any false alarm, store its characterization status, fEZ, dMag, and WA
                if FA == True:
                    DRM['FA_status'] = characterized.pop()
                    DRM['FA_SNR'] = charSNR.pop()
                    DRM['FA_fEZ'] = self.lastDetected[sInd,1][-1]
                    DRM['FA_dMag'] = self.lastDetected[sInd,2][-1]
                    DRM['FA_WA'] = self.lastDetected[sInd,3][-1]
                # Populate the DRM with characterization results
                DRM['char_time'] = t_char.to('day').value
                DRM['char_status'] = characterized
                DRM['char_SNR'] = charSNR
                
                # update target time
                self.starTimes[sInd] = TK.currentTimeNorm
                
                # append result values to self.DRM
                self.DRM.append(DRM)
                
                # with occulter, if spacecraft fuel is depleted, exit loop
                if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %r' % TK.currentTimeNorm
                    break
        
        mission_end = "Simulation finishing OK. Results stored in SurveySimulation.DRM"
        Logger.info(mission_end)
        print mission_end
        
        return mission_end

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
            DRM (dicts):
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
        
        # Create DRM
        DRM = {}
        
        # Allocate settling time + overhead time
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        if OS.haveOcculter == True:
            ao = Obs.thrust/Obs.scMass
            slewTime_fac = (2.*Obs.occulterSep/np.abs(ao)/(Obs.defburnPortion/2. \
                    - Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        # Now, start to look for available targets
        while not TK.mission_is_over():
            # 0/ initialize arrays
            slewTime = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            t_dets = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)
            
            # 1/ find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            if OS.haveOcculter == True:
                # find angle between old and new stars, default to pi/2 for first target
                if old_sInd is None:
                    sd = np.zeros(TL.nStars)*u.rad
                else:
                    # position vector of previous target star
                    r_old = Obs.starprop(TL, old_sInd, TK.currentTimeAbs)
                    u_old = r_old/np.sqrt(np.sum(r_old**2))
                    # position vector of new target stars
                    r_new = Obs.starprop(TL, sInds, TK.currentTimeAbs)
                    u_new = r_new/np.sqrt(np.sum(r_new**2))
                    # angle between old and new stars
                    sd = np.arccos(np.dot(u_old, u_new.T))[0]
                    sd[np.where(np.isnan(sd))] = 0.
                # calculate slew time
                slewTime = np.sqrt(slewTime_fac*np.sin(sd/2.))
            
            startTime = TK.currentTimeAbs + slewTime
            kogoodStart = Obs.keepout(TL, sInds, startTime, OS.telescopeKeepout)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 2/ calculate integration times for the preselected targets, 
            # and filter out t_tot > integration cutoff
            if np.any(sInds):
                fZ = ZL.fZ(TL, sInds, mode['lam'], Obs.orbit(startTime[sInds]))
                fEZ = ZL.fEZ0
                t_dets[sInds] = OS.calc_maxintTime(TL, sInds, fZ, fEZ, mode)
                # include integration time multiplier
                t_tot = t_dets*mode['timeMultiplier']
                # total time must be positive and shorter than treshold
                sInds = np.where((0 < t_tot) & (t_tot < OS.intCutoff))[0]
            
            # 3/ find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if np.any(sInds):
                endTime = startTime[sInds] + t_dets[sInds]
                kogoodEnd = Obs.keepout(TL, sInds, endTime, OS.telescopeKeepout)
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 4/ filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if np.any(sInds):
                tovisit[sInds] = (self.starVisits[sInds] == self.starVisits[sInds].min())
                if self.starRevisit.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]
                    tovisit[ind_rev] = True
                sInds = np.where(tovisit)[0]
            
            # 5/ choose best target from remaining
            if np.any(sInds):
                # prototype version choose sInd among targets with highest completeness
                comps = TL.comp0[sInds]
                updated = (self.starVisits[sInds] > 0)
                comps[updated] =  self.Completeness.completeness_update(TL, \
                        sInds[updated], TK.currentTimeNorm)
                sInd = np.random.choice(sInds[comps == max(comps)])
                # update visited list for current star
                self.starVisits[sInd] += 1
                # update visited list for Completeness for current star
                Comp.visits[sInd] += 1
                # store relevant values
                t_det = t_dets[sInd]
                break
            
            # if no observable target, allocate time and try again
            # TODO: improve how to deal with no available target
            else:
                TK.allocate_time(TK.dtAlloc)
            
        else:
            Logger.info('Mission complete: no more time available')
            return DRM, None, None
        
        if OS.haveOcculter == True:
            # find values related to slew time
            DRM['slew_time'] = slewTime[sInd].to('day').value
            DRM['slew_angle'] = sd[sInd].to('deg').value
            slew_mass_used = slewTime[sInd]*Obs.defburnPortion*Obs.flowRate
            DRM['slew_dV'] = (slewTime[sInd]*ao*Obs.defburnPortion).to('m/s').value
            DRM['slew_mass_used'] = slew_mass_used.to('kg').value
            Obs.scMass = Obs.scMass - slew_mass_used
            DRM['scMass'] = Obs.scMass.to('kg').value
            # update current time by adding slew time for the chosen target
            TK.allocate_time(slewTime[sInd])
            if TK.mission_is_over():
                Logger.info('Mission complete: no more time available')
                return DRM, None, None
        
        return DRM, sInd, t_det

    def observation_detection(self, sInd, t_det, mode):
        """Determines the detection status, and updates the last detected list 
        and the revisit list. 
        
        This method encodes detection status values in the DRM 
        dictionary.
        
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
                Detection status for each planet orbiting the observed target star,
                where 1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
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
        
        # Find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        # Find cases with working angles (WA) out of IWA-OWA range
        observable = np.ones(len(pInds), dtype=int)
        if np.any(observable):
            WA = SU.WA[pInds]
            observable[WA < mode['IWA']] = -1
            observable[WA > mode['OWA']] = -2
        
        # Now, calculate SNR for any observable planet (within IWA-OWA range)
        obs = (observable == 1)
        if np.any(obs):
            # initialize Signal and Noise arrays
            Signal = np.zeros((self.nt_flux, len(pInds[obs])))
            Noise = np.zeros((self.nt_flux, len(pInds[obs])))
            # integrate the signal (planet flux) and noise
            dt = t_det/self.nt_flux
            for i in range(self.nt_flux):
                s,n = self.calc_signal_noise(sInd, pInds[obs], dt, mode)
                Signal[i,:] = s
                Noise[i,:] = n
            # calculate SNRobs
            with np.errstate(divide='ignore',invalid='ignore'):
                SNRobs = Signal.sum(0) / Noise.sum(0)
            SNRobs[np.isnan(SNRobs)] = 0.
            # allocate extra time for timeMultiplier
            t_extra = t_det*(mode['timeMultiplier'] - 1)
            TK.allocate_time(t_extra)
        # if no planet, just observe for t_tot (including time multiplier)
        else:
            SNRobs = np.array([])
            t_tot = t_det*(mode['timeMultiplier'])
            TK.allocate_time(t_tot)
        
        # Find out if a false positive (false alarm) or any false negative 
        # (missed detections) have occurred, and populate detection status array
        FA, MD = PPro.det_occur(SNRobs, mode['SNR'])
        detected = observable
        SNR = np.zeros(len(pInds))
        if np.any(obs):
            detected[obs] = (~MD).astype(int)
            SNR[obs] = SNRobs
        
        # If planets are detected, calculate the minimum apparent separation
        smin = None
        det = (detected == 1)
        if np.any(det):
            smin = np.min(SU.s[pInds[det]])
            Logger.info('Detected planet(s) %r of target %r' % (pInds[det], sInd))
            print 'Detected planet(s)', pInds[det], 'of target', sInd
        
        # Populate the lastDetected array by storing det, fEZ, dMag, and WA
        self.lastDetected[sInd,:] = det, SU.fEZ[pInds].to('1/arcsec2').value, \
                    SU.dMag[pInds], SU.WA[pInds].to('mas').value
        
        # In case of a FA, generate a random delta mag (between maxFAfluxratio and
        # dMagLim) and working angle (between IWA and min(OWA, a_max))
        if FA == True:
            WA = np.random.uniform(mode['IWA'].to('mas'), np.minimum(mode['OWA'], \
                    np.arctan(max(PPop.arange)/TL.dist[sInd])).to('mas'))
            dMag = np.random.uniform(-2.5*np.log10(PPro.maxFAfluxratio(WA*u.mas)), OS.dMagLim)
            fEZ = ZL.fEZ0.to('1/arcsec2').value
            self.lastDetected[sInd,0] = np.append(self.lastDetected[sInd,0], True)
            self.lastDetected[sInd,1] = np.append(self.lastDetected[sInd,1], fEZ)
            self.lastDetected[sInd,2] = np.append(self.lastDetected[sInd,2], dMag)
            self.lastDetected[sInd,3] = np.append(self.lastDetected[sInd,3], WA)
            sminFA = np.tan(WA*u.mas)*TL.dist[sInd].to('AU')
            smin = np.minimum(smin,sminFA) if smin is not None else sminFA
            Logger.info('False Alarm at target %r with WA %r and dMag %r' % (sInd, WA, dMag))
            print 'False Alarm at target', sInd, 'with WA', WA, 'and dMag', dMag
        
        # In both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        Ms = TL.MsTrue[sInd]*const.M_sun
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
        # Otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + 0.75*T
        
        # Finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:
            self.starRevisit = np.array([revisit])
        else:
            self.starRevisit = np.vstack((self.starRevisit, revisit))
        
        return detected.tolist(), SNR.tolist(), FA

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
        
        # Find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        # Get the last detected planets, and check if there was a FA
        det = self.lastDetected[sInd,0]
        FA = (det.size == pInds.size+1)
        if FA == True:
            pInds = np.append(pInds,-1)
        
        # Initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
        SNR = np.zeros(det.size)
        t_char = 0.0*u.d
        if not np.any(pInds):
            return characterized.tolist(), SNR.tolist(), t_char
        
        # Look for last detected planets that have not been fully characterized
        tochar = np.zeros(len(det), dtype=bool)
        if (FA == False):
            tochar[det] = (self.fullSpectra[pInds[det]] != 1)
        elif pInds[det].size > 1:
            tochar[det] = np.append((self.fullSpectra[pInds[det][:-1]] != 1), True)
        else:
            tochar[det] = np.array([True])
        
        # Find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            startTime = TK.currentTimeAbs
            tochar[tochar] = Obs.keepout(TL, sInd, startTime, OS.telescopeKeepout)
        # If any planet to characterize, find the characterization times
        if np.any(tochar):
            # Propagate the whole system to match up with current time
            SU.propag_system(sInd, TK.currentTimeNorm)
            # Calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(TL, sInd, mode['lam'], Obs.orbit(startTime))
            fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            dMag = self.lastDetected[sInd,2][tochar]
            WA = self.lastDetected[sInd,3][tochar]*u.mas
            t_chars = np.zeros(len(pInds))*u.d
            t_chars[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            t_tots = t_chars*(mode['timeMultiplier'])
            # Filter out planets with t_tots > integration cutoff
            tochar = (t_tots > 0) & (t_tots < OS.intCutoff)
        # Is target still observable at the end of any char time?
        if np.any(tochar):
            endTime = TK.currentTimeAbs + t_tots[tochar]
            tochar[tochar] = Obs.keepout(TL, sInd, endTime, OS.telescopeKeepout)
        # If yes, perform the characterization for the maximum char time
        if np.any(tochar):
            t_char = np.max(t_chars[tochar])
            pIndsChar = pInds[tochar]
            Logger.info('Characterized planet(s) %r of target %r' % (pIndsChar, sInd))
            print 'Characterized planet(s)', pIndsChar, 'of target', sInd
            
            # SNR CALCULATION:
            # First, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            if np.any(planinds):
                Signal = np.zeros((self.nt_flux, len(planinds)))
                Noise = np.zeros((self.nt_flux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = t_char/self.nt_flux
                for i in range(self.nt_flux):
                    s,n = self.calc_signal_noise(sInd, planinds, dt, mode)
                    Signal[i,:] = s
                    Noise[i,:] = n
                # calculate SNRobs
                with np.errstate(divide='ignore',invalid='ignore'):
                    SNRobs = Signal.sum(0) / Noise.sum(0)
                # allocate extra time for timeMultiplier
                t_extra = t_char*(mode['timeMultiplier'] - 1)
                TK.allocate_time(t_extra)
            # if no planet (only false alarm), just observe for t_tot (including time multiplier)
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
                SNRfa[SNRfa > 0] /= np.sqrt(C_b*t_char + (C_sp*t_char)**2)\
                        .decompose().value[SNRfa > 0]
                SNRobs = np.append(SNRobs, SNRfa)
            
            # Now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            if np.any(SNRobs):
                SNR[tochar] = SNRobs
                char = (SNR >= mode['SNR'])
                if np.any(SNR):
                    # initialize with partial spectra
                    characterized[char] = -1
                    # check for full spectra
                    WA = self.lastDetected[sInd,3]*u.mas
                    IWA_max = mode['IWA']*(1+mode['BW']/2.)
                    OWA_min = mode['OWA']*(1-mode['BW']/2.)
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
        SU.propag_system(sInd, TK.currentTimeNorm)
        # find spacecraft position and ZodiacalLight
        fZ = ZL.fZ(TL, sInd, mode['lam'], Obs.orbit(TK.currentTimeAbs))
        # find electron counts for planet, background, and speckle residual 
        C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, \
                SU.fEZ[pInds], SU.dMag[pInds], SU.WA[pInds], mode)
        # calculate signal and noise levels (based on Nemati14 formula)
        Signal = (C_p*t_int).decompose().value
        Noise = np.sqrt((C_b*t_int + (C_sp*t_int)**2).decompose().value)
        # allocate second half of t_int
        TK.allocate_time(t_int/2.)
        
        return Signal, Noise

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
