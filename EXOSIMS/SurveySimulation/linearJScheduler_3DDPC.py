# -*- coding: utf-8 -*-
from EXOSIMS.SurveySimulation.linearJScheduler_DDPC import linearJScheduler_DDPC
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

import pdb

Logger = logging.getLogger(__name__)

class linearJScheduler_3DDPC(linearJScheduler_DDPC):
    """linearJScheduler_3DDPC - linearJScheduler 3 Dual Detection Parallel Charachterization

    This scheduler inherits from the LJS_DDPC, but is capable of taking in six detection
    modes and six  characterization modes. Detections can then be performed using a dual-band
    mode that is selected from the best available mode-pair, while characterizations 
    are performed in parallel.
    """

    def __init__(self, **specs):
        
        linearJScheduler_DDPC.__init__(self, **specs)

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
        det_modes = filter(lambda mode: 'imag' in mode['inst']['name'], allModes)[1:]
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
                cmodes = [cm for cm in char_modes if cm['systName'][-2] == dmode['systName'][-2]]

                if char_modes[0]['SNR'] not in [0, np.inf]:
                        characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                                self.observation_characterization(sInd, cmodes)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if True in FA else len(pInds)
                    characterized = np.zeros((lenChar,len(cmodes)), dtype=float)
                    char_SNR = np.zeros((lenChar,len(cmodes)), dtype=float)
                    char_fZ = np.array([0./u.arcsec**2, 0./u.arcsec**2])
                    char_systemParams = SU.dump_system_params(sInd)

                for mode_index, char_mode in enumerate(cmodes):
                    char_data = {}
                    assert char_intTime != 0, "Integration time can't be 0."
                    # update the occulter wet mass
                    if OS.haveOcculter == True and char_intTime is not None:
                        char_data = self.update_occulter_mass(char_data, sInd, char_intTime, 'char')
                    if np.any(characterized):
                        print '  Char. results are: %s'%(characterized[:-1, mode_index])
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
        PP = self.PlanetPopulation
        
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
            all_intTimes = np.zeros(TL.nStars)*u.d
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
            all_sInds = np.array([])
            # indices of observable stars
            # Iterate over all modes and compile a list of available targets after each filter. 
            # Combining them into a whole 
            for mode in modes:
                kogoodStart = Obs.keepout(TL, sInds, startTimes, mode)
                mode_sInds = sInds[np.where(kogoodStart)[0]]

                # 3. filter out all previously (more-)visited targets, unless in 
                # revisit list, with time within some dt of start (+- 1 week)
                mode_sInds = self.revisitFilter(mode_sInds, TK.currentTimeNorm)

                # 4. calculate integration times for ALL preselected targets, 
                # and filter out totTimes > integration cutoff
                if len(mode_sInds) > 0:
                    intTimes[mode_sInds] = self.calc_targ_intTime(mode_sInds, startTimes[mode_sInds], mode)

                    totTimes = intTimes*mode['timeMultiplier']
                    # end times
                    endTimes = startTimes + totTimes
                    endTimesNorm = startTimesNorm + totTimes
                    # indices of observable stars
                    mode_sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                            (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]

                for t in mode_sInds:
                    if intTimes[t] < all_intTimes[t]:
                        all_intTimes[t] = intTimes[t]
                # all_intTimes[mode_sInds] = intTimes[mode_sInds]
                
                # 5. find spacecraft orbital END positions (for each candidate target), 
                # and filter out unavailable targets
                if len(mode_sInds) > 0 and Obs.checkKeepoutEnd:
                    kogoodEnd = Obs.keepout(TL, mode_sInds, endTimes[mode_sInds], mode)
                    mode_sInds = mode_sInds[np.where(kogoodEnd)[0]]

                all_sInds = np.concatenate([all_sInds, mode_sInds]).astype(int)

            blue_modes = [mode for mode in modes if mode['systName'][-1] == 'b']
            sInds = np.unique(all_sInds)
            dmode = copy.deepcopy(blue_modes[0])

            # 6. choose best target from remaining
            if len(sInds) > 0:
                # choose sInd of next target
                sInd = self.choose_next_target(old_sInd, sInds, slewTimes, all_intTimes[sInds])
                #Should Choose Next Target decide there are no stars it wishes to observe at this time.
                if sInd is None:
                    TK.allocate_time(TK.waitTime)
                    intTime = None
                    self.vprint('There are no stars Choose Next Target would like to Observe. Waiting 1d')
                    continue

                s_IWA_OWA = (PP.arange * np.sqrt(TL.L[sInd])/TL.dist[sInd]).value*u.arcsec
                for bmode in blue_modes:
                    intTime = self.calc_targ_intTime(sInd, startTimes[sInd], bmode)[0]
                    if intTime != 0.0*u.d:
                        if s_IWA_OWA[0] < bmode['IWA'] < s_IWA_OWA[1] or s_IWA_OWA[0] < bmode['OWA'] < s_IWA_OWA[1]:
                            b_overlap = max(0, min(s_IWA_OWA[1], bmode['OWA']) - max(s_IWA_OWA[0], bmode['IWA']))
                            d_overlap = max(0, min(s_IWA_OWA[1], dmode['OWA']) - max(s_IWA_OWA[0], dmode['IWA']))
                            if b_overlap > d_overlap:
                                dmode = copy.deepcopy(bmode)
                            elif b_overlap == d_overlap:
                                if (bmode['OWA'] - bmode['IWA']) > (dmode['OWA'] - dmode['IWA']):
                                    dmode = copy.deepcopy(bmode)

                print(dmode['instName'], dmode['IWA'], dmode['OWA'])
                r_mode = [mode for mode in modes if mode['systName'][-1] == 'r' and mode['systName'][-2] == dmode['systName'][-2]][0]
                print(self.WAint[sInd])
                if self.WAint[sInd] > r_mode['IWA'] and self.WAint[sInd] < r_mode['OWA']:
                    dmode['BW'] = dmode['BW'] + r_mode['BW']
                    dmode['OWA'] = r_mode['OWA']
                    dmode['inst']['sread'] = dmode['inst']['sread'] + r_mode['inst']['sread']
                    dmode['inst']['idark'] = dmode['inst']['idark'] + r_mode['inst']['idark']
                    dmode['inst']['CIC'] = dmode['inst']['CIC'] + r_mode['inst']['CIC']
                    dmode['syst']['optics'] = np.mean((dmode['syst']['optics'], r_mode['syst']['optics']))
                    dmode['instName'] = dmode['instName'] + '_combined'
                print(dmode['instName'], dmode['IWA'], dmode['OWA'])
                intTime = self.calc_targ_intTime(sInd, startTimes[sInd], dmode)[0]

                break
            
            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.allocate_time(TK.waitTime*TK.waitMultiple**cnt)
                cnt += 1
            
        else:
            return DRM, None, None, None
        
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
                return DRM, None, None, None
        
        return DRM, sInd, intTime, dmode

