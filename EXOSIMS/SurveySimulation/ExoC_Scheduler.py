# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
import sys, logging
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
import random as py_random
import time
import json, os.path, copy, re, inspect, subprocess
import hashlib


class ExoC_Scheduler(SurveySimulation):
    """Survey Simulation class template
    
    This class contains all variables and methods necessary to perform
    Survey Simulation Module calculations in exoplanet mission simulation.
    
    It inherits from the prototype SurveySimulation module.

    The ExoC_Scheduler is designed to perform detections as normal
    and characterizations in series.

    Args:
        \*\*specs:
            user specified values
        
    """
    
    def __init__(self, **specs):

        SurveySimulation.__init__(self, **specs)

        OS = self.OpticalSystem
        SU = self.SimulatedUniverse

        allModes = OS.observingModes
        num_char_modes = len(list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes)))
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
        det_mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
        if np.any(spectroModes):
            char_modes = spectroModes
        # if no spectro mode, default char mode is first observing mode
        else:
            char_modes = [allModes[0]]
        
        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        while not TK.mission_is_over(OS, Obs, det_mode):
            
            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd #used to save sInd if returned sInd is None
            DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)

            if sInd is not None:
                ObsNum += 1 #we're making an observation so increment observation number
                
                if OS.haveOcculter == True:
                    # advance to start of observation (add slew time for selected target)
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                    
                # beginning of observation, start to populate DRM
                DRM['star_ind'] = sInd
                DRM['star_name'] = TL.Name[sInd]
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').copy()
                DRM['OB_nb'] = TK.OBnumber
                DRM['ObsNum'] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int)
                log_obs = ('  Observation #%s, star ind %s (of %s) with %s planet(s), ' \
                        + 'mission time at Obs start: %s, exoplanetObsTime: %s')%(ObsNum, sInd, TL.nStars, len(pInds), 
                        TK.currentTimeNorm.to('day').copy().round(2), TK.exoplanetObsTime.to('day').copy().round(2))
                self.logger.info(log_obs)
                self.vprint(log_obs)
                
                # PERFORM DETECTION and populate revisit list attribute
                detected, det_fZ, det_systemParams, det_SNR, FA = \
                        self.observation_detection(sInd, det_intTime.copy(), det_mode)
                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime.copy(), 'det')
                # populate the DRM with detection results
                DRM['det_time'] = det_intTime.to('day')
                DRM['det_status'] = detected
                DRM['det_SNR'] = det_SNR
                DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                DRM['det_params'] = det_systemParams
                DRM['det_mode'] = dict(det_mode)
                del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                DRM['char_info'] = []
                for mode_index, char_mode in enumerate(char_modes):
                    char_data = {}
                    if char_mode['SNR'] not in [0, np.inf]:
                        characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                                self.observation_characterization(sInd, char_mode, mode_index)
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
                    char_data['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                    char_data['char_status'] = characterized[:-1] if FA else characterized
                    char_data['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                    char_data['char_fZ'] = char_fZ.to('1/arcsec2')
                    char_data['char_params'] = char_systemParams
                    # populate the DRM with FA results
                    char_data['FA_det_status'] = int(FA)
                    char_data['FA_char_status'] = characterized[-1] if FA else 0
                    char_data['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                    char_data['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 \
                            if FA else 0./u.arcsec**2
                    char_data['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                    char_data['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec \
                            if FA else 0.*u.arcsec
                    
                    # populate the DRM with observation modes
                    char_data['char_mode'] = dict(char_mode)
                    del char_data['char_mode']['inst'], char_data['char_mode']['syst']

                    char_data['exoplanetObsTime'] = TK.exoplanetObsTime.copy()
                    DRM['char_info'].append(char_data)
                
                # append result values to self.DRM
                self.DRM.append(DRM)

                # handle case of inf OBs and missionPortion < 1
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1.):
                    self.arbitrary_time_advancement(TK.currentTimeNorm.to('day').copy() - DRM['arrival_time'])
                
            else:#sInd == None
                sInd = old_sInd#Retain the last observed star
                if(TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]): # currentTime is at end of OB
                    #Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(OS, Obs,det_mode):#as long as the mission is not over
                        TK.advancetToStartOfNextOB()#Advance To Start of Next OB
                elif(waitTime is not None):
                    #CASE 1: Advance specific wait time
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                    self.vprint('waitTime is not None')
                else:
                    startTimes = TK.currentTimeAbs.copy() + np.zeros(TL.nStars)*u.d # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(TL,np.arange(TL.nStars),startTimes,self.koMaps,self.koTimes,self.mode)[0]
                    #CASE 2 If There are no observable targets for the rest of the mission
                    if((observableTimes[(TK.missionFinishAbs.copy().value*u.d > observableTimes.value*u.d)*(observableTimes.value*u.d >= TK.currentTimeAbs.copy().value*u.d)].shape[0]) == 0):#Are there any stars coming out of keepout before end of mission
                        self.vprint('No Observable Targets for Remainder of mission at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
                        #Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife
                        TK.currentTimeAbs = TK.missionFinishAbs
                    else:#CASE 3    nominal wait time if at least 1 target is still in list and observable
                        #TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[observableTimes.value*u.d > TK.currentTimeAbs.copy().value*u.d]
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1) #apply intTime filter
                        inds3 = self.revisitFilter(inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)) #apply revisit Filter #NOTE this means stars you added to the revisit list 
                        self.vprint("Filtering %d stars from advanceToAbsTime"%(TL.nStars - len(inds3)))
                        oTnowToEnd = observableTimes[inds3]
                        if not oTnowToEnd.value.shape[0] == 0: #there is at least one observableTime between now and the end of the mission
                            tAbs = np.min(oTnowToEnd)#advance to that observable time
                        else:
                            tAbs = TK.missionStart + TK.missionLife#advance to end of mission
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        success = TK.advanceToAbsTime(tAbs)#Advance Time to this time OR start of next OB following this time
                        self.vprint('No Observable Targets a currentTimeNorm= %.2f Advanced To currentTimeNorm= %.2f'%(tmpcurrentTimeNorm.to('day').value, TK.currentTimeNorm.to('day').value))
        else:#TK.mission_is_over()
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            self.vprint(log_end)


    def observation_characterization(self, sInd, mode, mode_index):
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
        fZ = 0./u.arcsec**2.
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0: # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        if (FA == False): # only true planets, no FA
            tochar = (self.fullSpectra[mode_index][pIndsDet] == 0)
        else: # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[mode_index][truePlans] == 0), True)
        
        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs.copy() + mode['syst']['ohTime'] + Obs.settlingTime
            startTimeNorm = TK.currentTimeNorm.copy() + mode['syst']['ohTime'] + Obs.settlingTime
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value)-self.koTimes.value==0)[0][0]  # find indice where koTime is startTime[0]
            #wherever koMap is 1, the target is observable
            koMap = self.koMaps[mode['syst']['name']]
            tochar[tochar] = koMap[sInd][koTimeInd]
        
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
            intTimes = intTimes*(1. + self.charMargin)
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
            koTimeInds = np.zeros(len(endTimes.value[tochar]),dtype=int)
            # find index in koMap where each endTime is closest to koTimes
            for t,endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(np.floor(endTime)-self.koTimes.value==0)[0]
                    koTimeInds[t] = endTimeInBounds[0] if endTimeInBounds.size is not 0 else -1
                else:
                    koTimeInds[t] = np.where(np.round(endTime)-self.koTimes.value==0)[0][0]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]
        
        # 4/ if yes, allocate the overhead time, and perform the characterization 
        # for the maximum char time
        if np.any(tochar):
            #Save Current Time before attempting time allocation
            currentTimeNorm = TK.currentTimeNorm.copy()
            currentTimeAbs = TK.currentTimeAbs.copy()

            #Allocate Time
            intTime = np.max(intTimes[tochar])
            extraTime = intTime*(mode['timeMultiplier'] - 1.)#calculates extraTime
            success = TK.allocate_time(intTime + extraTime + mode['syst']['ohTime'] + Obs.settlingTime, True)#allocates time
            if success == False: #Time was not successfully allocated
                #Identical to when "if char_mode['SNR'] not in [0, np.inf]:" in run_sim()
                char_intTime = None
                lenChar = len(pInds) + 1 if FA else len(pInds)
                characterized = np.zeros(lenChar, dtype=float)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0./u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)
                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

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
                fZs = np.zeros(self.ntFlux)/u.arcsec**2.
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime/float(self.ntFlux)
                timePlus = Obs.settlingTime.copy() + mode['syst']['ohTime'].copy()#accounts for the time since the current time
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    timePlus += dt/2.
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, currentTimeNorm + timePlus - self.propagTimes[sInd])
                    self.propagTimes[sInd] = currentTimeNorm + timePlus
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                            fZ=fZs[i])
                    # allocate second half of dt
                    timePlus += dt/2.
                
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
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs + totTime/2., mode)[0]
            
            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2.
                dMag = self.lastDetected[sInd,2][-1]
                WA = self.lastDetected[sInd,3][-1]*u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p*intTime).decompose().value
                N = np.sqrt((C_b*intTime + (C_sp*intTime)**2.).decompose().value)
                SNRfa = S/N if N > 0. else 0.
            
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
            IWA_max = mode['IWA']*(1. + mode['BW']/2.)
            OWA_min = mode['OWA']*(1. - mode['BW']/2.)
            char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[mode_index][pInds[charplans == 1]] += 1
            self.partialSpectra[mode_index][pInds[charplans == -1]] += 1
        
        return characterized.astype(int), fZ, systemParams, SNR, intTime

