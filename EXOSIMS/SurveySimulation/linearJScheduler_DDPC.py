# -*- coding: utf-8 -*-
from EXOSIMS.SurveySimulation.linearJScheduler import linearJScheduler
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

class linearJScheduler_DDPC(linearJScheduler):
    """linearJScheduler_DDPC - linearJScheduler Dual Detection Parallel Charachterization

    This scheduler inherits from the LJS, but is capable of taking in two detection
    modes and two chracterization modes. Detections can then be performed using a dual-band
    mode, while characterizations are performed in parallel.
    """

    def __init__(self, **specs):
        
        linearJScheduler.__init__(self, **specs)

        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TL = self.TargetList

        allModes = OS.observingModes
        num_char_modes = len(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)

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
            DRM, sInd, det_intTime, dmode = self.next_target(sInd, det_modes)
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
                        self.observation_detection(sInd, det_intTime, dmode)
                det_data = {}
                det_data['det_status'] = detected
                det_data['det_SNR'] = det_SNR
                det_data['det_fZ'] = det_fZ.to('1/arcsec2')
                det_data['det_params'] = det_systemParams
                det_data['det_mode'] = dict(dmode)
                det_data['det_time'] = det_intTime.to('day')
                del det_data['det_mode']['inst'], det_data['det_mode']['syst']
                DRM['det_info'].append(det_data)

                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime, 'det')
                # populate the DRM with detection results

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                DRM['char_info'] = []
                if char_modes[0]['SNR'] not in [0, np.inf]:
                        characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                                self.observation_characterization(sInd, char_modes)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if True in FA else len(pInds)
                    characterized = np.zeros((lenChar,len(char_modes)), dtype=float)
                    char_SNR = np.zeros((lenChar,len(char_modes)), dtype=float)
                    char_fZ = np.array([0./u.arcsec**2, 0./u.arcsec**2])
                    char_systemParams = SU.dump_system_params(sInd)

                for mode_index, char_mode in enumerate(char_modes):
                    char_data = {}
                    assert char_intTime != 0, "Integration time can't be 0."
                    # update the occulter wet mass
                    if OS.haveOcculter == True and char_intTime is not None:
                        char_data = self.update_occulter_mass(char_data, sInd, char_intTime, 'char')
                    # populate the DRM with characterization results
                    char_data['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                    char_data['char_status'] = characterized[:-1, mode_index] if FA else characterized[:,mode_index]
                    char_data['char_SNR'] = char_SNR[:-1, mode_index] if FA else char_SNR[:, mode_index]
                    char_data['char_fZ'] = char_fZ[mode_index].to('1/arcsec2')
                    char_data['char_params'] = char_systemParams
                    # populate the DRM with FA results
                    char_data['FA_det_status'] = int(FA)
                    char_data['FA_char_status'] = characterized[-1, mode_index] if FA else 0
                    char_data['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                    char_data['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 \
                            if FA else 0./u.arcsec**2
                    char_data['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                    char_data['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec \
                            if FA else 0.*u.arcsec
                    
                    # populate the DRM with observation modes
                    char_data['char_mode'] = dict(char_mode)
                    del char_data['char_mode']['inst'], char_data['char_mode']['syst']
                    DRM['char_info'].append(char_data)
                
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

                dmode = copy.deepcopy(modes[0])
                if self.WAint[sInd] > modes[1]['IWA'] and self.WAint[sInd] < modes[1]['OWA']:
                    dmode['BW'] = dmode['BW'] + modes[1]['BW']
                    dmode['inst']['sread'] = dmode['inst']['sread'] + modes[1]['inst']['sread']
                    dmode['inst']['idark'] = dmode['inst']['idark'] + modes[1]['inst']['idark']
                    dmode['inst']['CIC'] = dmode['inst']['CIC'] + modes[1]['inst']['CIC']
                    dmode['syst']['optics'] = np.mean((dmode['syst']['optics'], modes[1]['syst']['optics']))
                    dmode['instName'] = 'combined'
                    intTime = self.calc_targ_intTime(sInd, startTimes[sInd], dmode)[0]
                else:
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
        
        return DRM, sInd, intTime, dmode


    def observation_characterization(self, sInd, modes):
        """Finds if characterizations are possible and relevant information
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            modes (dict):
                Selected observing modes for characterization
        
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

        nmodes = len(modes)
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]

        pIndsDet = []
        tochars = []
        intTimes_all = []
        FA = (len(det) == len(pInds) + 1)

        # initialize outputs, and check if there's anything (planet or FA) to characterize
        characterizeds = np.zeros((det.size, len(modes)), dtype=int)
        fZ = 0./u.arcsec**2 * np.ones(nmodes)
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros((len(det),len(modes)))
        intTime = None
        if det.size == 0: # nothing to characterize
            return characterizeds, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        for m_i, mode in enumerate(modes):

            if FA is True:
                pIndsDet.append(np.append(pInds, -1)[det])
            else:
                pIndsDet.append(pInds[det])

            if (FA == False): # only true planets, no FA
                tochar = (self.fullSpectra[m_i][pIndsDet[m_i]] == 0)
            else: # mix of planets and a FA
                truePlans = pIndsDet[m_i][:-1]
                tochar = np.append((self.fullSpectra[m_i][truePlans] == 0), True)
        
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
                fZ[m_i] = ZL.fZ(Obs, TL, sInd, startTime, mode)
                fEZ = self.lastDetected[sInd,1][det][tochar]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][det][tochar]
                WA = self.lastDetected[sInd,3][det][tochar]*u.arcsec
                intTimes = np.zeros(len(tochar))*u.day
                intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ[m_i], fEZ, dMag, WA, mode)
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

                tochars.append(tochar)
                intTimes_all.append(intTimes)
            else:
                tochar[tochar] = False
                tochars.append(tochar)

        # 4/ if yes, allocate the overhead time, and perform the characterization 
        # for the maximum char time
        if np.any(tochars):
            pIndsChar = []
            TK.allocate_time(modes[0]['syst']['ohTime'])
            for m_i, mode in enumerate(modes):
                if len(pIndsDet[m_i]) > 0:
                    if intTime is None or np.max(intTimes_all[0][tochars[0]]) > intTime:
                        intTime = np.max(intTimes_all[0][tochars[0]])
                    pIndsChar.append(pIndsDet[m_i][tochars[m_i]])
                    log_char = '   - Charact. planet inds %s (%s/%s detected)'%(pIndsChar[m_i], 
                            len(pIndsChar[m_i]), len(pIndsDet[m_i]))
                    self.logger.info(log_char)
                    self.vprint(log_char)
                else:
                    pIndsChar.append([])
            
            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            if len(pIndsChar[0]) > 0:
                planinds = pIndsChar[0][:-1] if pIndsChar[0][-1] == -1 else pIndsChar[0]
            else: 
                planinds = []
            if len(pIndsChar[1]) > 0:
                planinds2 = pIndsChar[1][:-1] if pIndsChar[1][-1] == -1 else pIndsChar[1]
            else:
                planinds2 = []
            SNRplans = np.zeros((len(planinds)))
            SNRplans2 = np.zeros((len(planinds2)))
            if len(planinds) > 0 and len(planinds2) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros((self.ntFlux, nmodes))/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                Ss2 = np.zeros((self.ntFlux, len(planinds2)))
                Ns2 = np.zeros((self.ntFlux, len(planinds2)))
                # integrate the signal (planet flux) and noise
                dt = intTime/self.ntFlux
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    TK.allocate_time(dt/2.)
                    fZs[i,0] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, modes[0])[0]
                    fZs[i,1] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, modes[1])[0]
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm
                    systemParamss[i] = SU.dump_system_params(sInd)
                    Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, modes[0], fZ=fZs[i,0])
                    Ss2[i,:], Ns2[i,:] = self.calc_signal_noise(sInd, planinds2, dt, modes[1], fZ=fZs[i,1])

                    # for m_i, mode in enumerate(modes):
                    #     # calculate current zodiacal light brightness
                    #     fZs[i,m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                    #     # propagate the system to match up with current time
                    #     SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    #     self.propagTimes[sInd] = TK.currentTimeNorm
                    #     # save planet parameters
                    #     systemParamss[i] = SU.dump_system_params(sInd)
                    #     # calculate signal and noise (electron count rates)
                    #     Ss[i,:,m_i], Ns[i,:,m_i] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                    #                                                       fZ=fZs[i,m_i])
                    # allocate second half of dt
                    TK.allocate_time(dt/2.)
                
                # average output parameters
                systemParams = {key: sum([systemParamss[x][key]
                            for x in range(self.ntFlux)])/float(self.ntFlux)
                            for key in sorted(systemParamss[0])}
                for m_i, mode in enumerate(modes):
                    fZ[m_i] = np.mean(fZs[:,m_i])
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                S2 = Ss2.sum(0)
                N2 = Ns2.sum(0)
                SNRplans[N > 0] = S[N > 0]/N[N > 0]
                SNRplans2[N2 > 0] = S2[N2 > 0]/N2[N2 > 0]
                # allocate extra time for timeMultiplier
                extraTime = intTime*(mode['timeMultiplier'] - 1)
                TK.allocate_time(extraTime)
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                TK.allocate_time(totTime/2.)
                for m_i, mode in enumerate(modes):
                    fZ[m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime/2.)
            
            # calculate the false alarm SNR (if any)
            for m_i, mode in enumerate(modes):
                if len(pIndsChar[m_i]) > 0:
                    SNRfa = []
                    if pIndsChar[m_i][-1] == -1:
                        fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2
                        dMag = self.lastDetected[sInd,2][-1]
                        WA = self.lastDetected[sInd,3][-1]*u.arcsec
                        C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ[m_i], fEZ, dMag, WA, mode)
                        S = (C_p*intTime).decompose().value
                        N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
                        SNRfa.append([S/N if N > 0 else 0.])
                
                    # save all SNRs (planets and FA) to one array
                    SNRinds = np.where(det)[0][tochars[m_i]]
                    if m_i == 0:
                        SNR[SNRinds, 0] = np.append(SNRplans[:], SNRfa)
                    else:
                        SNR[SNRinds, 1] = np.append(SNRplans2[:], SNRfa)

                    # if np.append(SNRplans[:, m_i], SNRfa).shape != SNR[SNRinds, m_i].shape:
                    #     SNR[SNRinds, m_i] = np.append(SNRplans[:, m_i], SNRfa)[SNRinds]
                    # else:
                    #     SNR[SNRinds, m_i] = np.append(SNRplans[:, m_i], SNRfa)
                
                    # now, store characterization status: 1 for full spectrum, 
                    # -1 for partial spectrum, 0 for not characterized
                    char = (SNR[:,m_i] >= mode['SNR'])
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
                    self.fullSpectra[m_i][pInds[charplans == 1]] += 1
                    self.partialSpectra[m_i][pInds[charplans == -1]] += 1
                    characterizeds[:,m_i] = characterized.astype(int)
        
        return characterizeds, fZ, systemParams, SNR, intTime


