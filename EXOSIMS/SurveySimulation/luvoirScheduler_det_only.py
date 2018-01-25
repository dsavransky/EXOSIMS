# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
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

class luvoirScheduler_det_only(SurveySimulation):

    def __init__(self, **specs):
        
        SurveySimulation.__init__(self, **specs)

        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TL = self.TargetList

        allModes = OS.observingModes
        num_char_modes = len(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
        num_det_modes = len(filter(lambda mode: 'imag' in mode['inst']['name'], allModes))
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.lastDetected = np.empty((TL.nStars, 4, num_det_modes), dtype=object)

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
        det_modes = filter(lambda mode: 'imag' in mode['inst']['name'], allModes)
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], allModes)
        if np.any(spectroModes):
            char_modes = spectroModes
        # if no spectro mode, default char mode is first observing mode
        else:
            char_modes = [allModes[0]]
        
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
            DRM, sInd, det_intTime = self.next_target(sInd, det_modes)
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
                DRM['det_info'] = []
                detected, det_fZ, det_systemParams, det_SNR, FA = \
                        self.observation_detection(sInd, det_intTime, det_modes)
                for mode_index, det_mode in enumerate(det_modes):
                    det_data = {}
                    det_data['det_status'] = detected[mode_index]
                    det_data['det_SNR'] = det_SNR[:,mode_index]
                    det_data['det_fZ'] = det_fZ[mode_index].to('1/arcsec2')
                    det_data['det_params'] = det_systemParams
                    det_data['det_mode'] = dict(det_mode)
                    det_data['det_time'] = det_intTime.to('day')
                    del det_data['det_mode']['inst'], det_data['det_mode']['syst']
                    DRM['det_info'].append(det_data)

                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime, 'det')
                # populate the DRM with detection results
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                DRM['char_info'] = []
                # for mode_index, char_mode in enumerate(char_modes):
                #     char_data = {}
                #     if char_mode['SNR'] not in [0, np.inf]:
                #         characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                #                 self.observation_characterization(sInd, char_mode, mode_index)
                #     else:
                #         char_intTime = None
                #         lenChar = len(pInds) + 1 if FA[mode_index] else len(pInds)
                #         characterized = np.zeros(lenChar, dtype=float)
                #         char_SNR = np.zeros(lenChar, dtype=float)
                #         char_fZ = 0./u.arcsec**2
                #         char_systemParams = SU.dump_system_params(sInd)
                #     assert char_intTime != 0, "Integration time can't be 0."
                #     # update the occulter wet mass
                #     if OS.haveOcculter == True and char_intTime is not None:
                #         char_data = self.update_occulter_mass(char_data, sInd, char_intTime, 'char')
                #     # populate the DRM with characterization results
                #     char_data['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                #     char_data['char_status'] = characterized[:-1] if FA[mode_index] else characterized
                #     char_data['char_SNR'] = char_SNR[:-1] if FA[mode_index] else char_SNR
                #     char_data['char_fZ'] = char_fZ.to('1/arcsec2')
                #     char_data['char_params'] = char_systemParams
                #     # populate the DRM with FA results
                #     char_data['FA_det_status'] = int(FA[mode_index])
                #     char_data['FA_char_status'] = characterized[-1] if FA[mode_index] else 0
                #     char_data['FA_char_SNR'] = char_SNR[-1] if FA[mode_index] else 0.
                #     char_data['FA_char_fEZ'] = self.lastDetected[sInd,1,mode_index][-1]/u.arcsec**2 \
                #             if FA[mode_index] else 0./u.arcsec**2
                #     char_data['FA_char_dMag'] = self.lastDetected[sInd,2,mode_index][-1] if FA[mode_index] else 0.
                #     char_data['FA_char_WA'] = self.lastDetected[sInd,3,mode_index][-1]*u.arcsec \
                #             if FA[mode_index] else 0.*u.arcsec
                    
                #     # populate the DRM with observation modes
                #     char_data['char_mode'] = dict(char_mode)
                #     del char_data['char_mode']['inst'], char_data['char_mode']['syst']
                #     DRM['char_info'].append(char_data)
                
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

    def next_target(self, old_sInd, modes):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            mode (dict):
                Selected observing modes for detection
                
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
        TK.allocate_time(Obs.settlingTime + modes[0]['syst']['ohTime'])
        
        # now, start to look for available targets
        cnt = 0
        while not TK.mission_is_over():
            # 1. initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            intTimes = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
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
            kogoodStart = Obs.keepout(TL, sInds, startTimes, modes[0])
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 3. filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if len(sInds) > 0:
                tovisit[sInds] = ((self.starVisits[sInds] == min(self.starVisits[sInds])) \
                        & (self.starVisits[sInds] < self.nVisitsMax))
                if self.starRevisit.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] 
                            if x in sInds]
                    tovisit[ind_rev] = (self.starVisits[ind_rev] < self.nVisitsMax)
                sInds = np.where(tovisit)[0]

            # 4. calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            if len(sInds) > 0:
                intTimes[sInds] = self.calc_targ_intTime(sInds,startTimes[sInds], modes[0])

                totTimes = intTimes*modes[0]['timeMultiplier']
                # end times
                endTimes = startTimes + totTimes
                endTimesNorm = startTimesNorm + totTimes
                # indices of observable stars
                sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 5. find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(sInds) > 0 and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, sInds, endTimes[sInds], modes[0])
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 6. choose best target from remaining
            if len(sInds) > 0:
                # choose sInd of next target
                sInd = self.choose_next_target(old_sInd, sInds, slewTimes, intTimes[sInds])
                #Should Choose Next Target decide there are no stars it wishes to observe at this time.
                if sInd is None:
                    TK.allocate_time(TK.waitTime)
                    intTime = None
                    self.vprint('There are no stars Choose Next Target would like to Observe. Waiting 1d')
                    continue
                # store selected star integration time
                # intTime_by_mode = np.zeros(len(modes))*u.d
                # for m_i, mode in enumerate(modes):
                #     intTime_by_mode[m_i] = self.calc_targ_intTime(sInd, startTimes[sInd], mode)
                # intTime = max(intTime_by_mode)
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

    def observation_detection(self, sInd, intTime, modes):
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
        nmodes = len(modes)
        
        # initialize outputs
        detecteds = []
        fZ = 0./u.arcsec**2 * np.ones(nmodes)
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros((len(pInds), nmodes))
        FAs = []
        smins = []
        
        # if any planet, calculate SNR
        if len(pInds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros((self.ntFlux, nmodes))/u.arcsec**2
            systemParamss = np.empty(self.ntFlux, dtype='object')
            Ss = np.zeros((self.ntFlux, len(pInds), nmodes))
            Ns = np.zeros((self.ntFlux, len(pInds), nmodes))
            # integrate the signal (planet flux) and noise
            dt = intTime/self.ntFlux
            for i in range(self.ntFlux):
                # allocate first half of dt
                TK.allocate_time(dt/2.)
                # calculate current zodiacal light brightness
                for m_i, mode in enumerate(modes):
                    fZs[i, m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                # propagate the system to match up with current time
                SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                self.propagTimes[sInd] = TK.currentTimeNorm
                # save planet parameters
                systemParamss[i] = SU.dump_system_params(sInd)
                # calculate signal and noise (electron count rates)
                for m_i, mode in enumerate(modes):
                    Ss[i,:,m_i], Ns[i,:,m_i] = self.calc_signal_noise(sInd, pInds, dt, mode, 
                                                                      fZ=fZs[i,m_i])
                # allocate second half of dt
                TK.allocate_time(dt/2.)
            
            # average output parameters
            for m_i, mode in enumerate(modes):
                fZ[m_i] = np.mean(fZs[:,m_i])
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
            totTime = intTime*(modes[0]['timeMultiplier'])
            TK.allocate_time(totTime/2.)
            for m_i, mode in enumerate(modes):
                fZ[m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
            TK.allocate_time(totTime/2.)
        
        # find out if a false positive (false alarm) or any false negative 
        # (missed detections) have occurred
        for m_i, mode in enumerate(modes):
            FA, MD = PPro.det_occur(SNR[:,m_i], mode, TL, sInd, intTime)
            FAs.append(FA)
            detected = np.array([], dtype=int)
        
            # populate detection status array 
            # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
            if len(pInds) > 0:
                detected = (~MD).astype(int)
                WA = np.array([systemParamss[x]['WA'].to('arcsec').value 
                        for x in range(len(systemParamss))])*u.arcsec
                detected[np.all(WA < mode['IWA'], 0)] = -1
                detected[np.all(WA > mode['OWA'], 0)] = -2

            detecteds.append(detected)
                
            # if planets are detected, calculate the minimum apparent separation
            smin = np.nan
            det = (detected == 1)
            if np.any(det):
                smin = np.min(SU.s[pInds[det]])
                log_det = '   - Detected planet inds %s (%s/%s)'%(pInds[det], 
                        len(pInds[det]), len(pInds))
                self.logger.info(log_det)
                self.vprint(log_det)
            
            # populate the lastDetected array by storing det, fEZ, dMag, and WA
            self.lastDetected[sInd,:,m_i] = [det, systemParams['fEZ'].to('1/arcsec2').value, 
                        systemParams['dMag'], systemParams['WA'].to('arcsec').value]
            
            # in case of a FA, generate a random delta mag (between PPro.FAdMag0 and
            # Comp.dMagLim) and working angle (between IWA and min(OWA, a_max))
            if FA == True:
                WA = np.random.uniform(mode['IWA'].to('arcsec').value, np.minimum(mode['OWA'],
                        np.arctan(max(PPop.arange)/TL.dist[sInd])).to('arcsec').value)*u.arcsec
                dMag = np.random.uniform(PPro.FAdMag0(WA), Comp.dMagLim)
                self.lastDetected[sInd,0,m_i] = np.append(self.lastDetected[sInd,0,m_i], True)
                self.lastDetected[sInd,1,m_i] = np.append(self.lastDetected[sInd,1,m_i], 
                        ZL.fEZ0.to('1/arcsec2').value)
                self.lastDetected[sInd,2,m_i] = np.append(self.lastDetected[sInd,2,m_i], dMag)
                self.lastDetected[sInd,3,m_i] = np.append(self.lastDetected[sInd,3,m_i], 
                        WA.to('arcsec').value)
                sminFA = np.tan(WA)*TL.dist[sInd].to('AU')
                smin = np.minimum(smin, sminFA) if smin is not np.nan else sminFA
                log_FA = '   - False Alarm (WA=%s, dMag=%s)'%(np.round(WA, 3), round(dMag, 1))
                self.logger.info(log_FA)
                self.vprint(log_FA)

            smins.append(smin)
            
            # in both cases (detection or false alarm), schedule a revisit 
            # based on minimum separation
            Ms = TL.MsTrue[sInd]
            if m_i == len(modes) - 1:
                if np.nan not in smins:
                    sp = smins[0]
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
                    revInd = np.where(self.starRevisit[:,0] == sInd)[0]
                    if revInd.size == 0:
                        self.starRevisit = np.vstack((self.starRevisit, revisit))
                    else:
                        self.starRevisit[revInd,1] = revisit[1]
        
        return np.array(detecteds).astype(int), fZ, systemParams, SNR, np.array(FAs)

