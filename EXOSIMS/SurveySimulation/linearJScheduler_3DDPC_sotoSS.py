# -*- coding: utf-8 -*-
from EXOSIMS.SurveySimulation.linearJScheduler_DDPC_sotoSS import linearJScheduler_DDPC_sotoSS
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

class linearJScheduler_3DDPC_sotoSS(linearJScheduler_DDPC_sotoSS):
    """linearJScheduler_3DDPC_sotoSS - linearJScheduler 3 Dual Detection Parallel Characterization SotoStarshade

    This scheduler inherits from the LJS_DDPC, but is capable of taking in six detection
    modes and six  characterization modes. Detections can then be performed using a dual-band
    mode that is selected from the best available mode-pair, while characterizations 
    are performed in parallel.

    Args:
        \*\*specs:
            user specified values
    """

    def __init__(self, **specs):
        
        linearJScheduler_DDPC_sotoSS.__init__(self, **specs)

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
        det_modes = list(filter(lambda mode: 'imag' in mode['inst']['name'], allModes))[1:]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
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
        ObsNum = 0
        while not TK.mission_is_over(OS, Obs, det_modes[0]):
            
            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd #used to save sInd if returned sInd is None
            DRM, sInd, det_intTime, waitTime, det_mode = self.next_target(sInd, det_modes)

            if sInd is not None:
                ObsNum += 1

                if OS.haveOcculter == True:
                    # advance to start of observation (add slew time for selected target)
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                
                # beginning of observation, start to populate DRM
                DRM['star_ind'] = sInd
                DRM['star_name'] = TL.Name[sInd]
                DRM['arrival_time'] = TK.currentTimeNorm.copy().to('day')
                DRM['OB_nb'] = TK.OBnumber
                DRM['ObsNum'] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int)
                log_obs = ('  Observation #%s, star ind %s (of %s) with %s planet(s), ' \
                        + 'mission time at Obs start: %s')%(ObsNum, sInd, TL.nStars, len(pInds), 
                        TK.currentTimeNorm.to('day').copy().round(2))
                self.logger.info(log_obs)
                self.vprint(log_obs)

                # PERFORM DETECTION and populate revisit list attribute
                DRM['det_info'] = []
                detected, det_fZ, det_systemParams, det_SNR, FA = \
                        self.observation_detection(sInd, det_intTime, det_mode)
                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime, 'det')
                det_data = {}
                det_data['det_status'] = detected
                det_data['det_SNR'] = det_SNR
                det_data['det_fZ'] = det_fZ.to('1/arcsec2')
                det_data['det_params'] = det_systemParams
                det_data['det_mode'] = dict(det_mode)
                det_data['det_time'] = det_intTime.to('day')
                del det_data['det_mode']['inst'], det_data['det_mode']['syst']
                DRM['det_info'].append(det_data)

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                DRM['char_info'] = []
                cmodes = [cm for cm in char_modes if cm['systName'][-2] == det_mode['systName'][-2]]

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
                        vprint('  Char. results are: {}'.format(characterized[:-1, mode_index]))
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
                
                DRM['exoplanetObsTime'] = TK.exoplanetObsTime.copy()

                # append result values to self.DRM
                self.DRM.append(DRM)
                
            else:#sInd == None
                sInd = old_sInd#Retain the last observed star
                if(TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]): # currentTime is at end of OB
                    #Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(OS, Obs, det_mode):#as long as the mission is not over
                        TK.advancetToStartOfNextOB()#Advance To Start of Next OB
                elif(waitTime is not None):
                    #CASE 1: Advance specific wait time
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                    self.vprint('waitTime is not None')
                else:
                    startTimes = TK.currentTimeAbs.copy() + np.zeros(TL.nStars)*u.d # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(TL, np.arange(TL.nStars), startTimes, self.koMaps, self.koTimes, self.mode)[0]
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
            waitTime (astropy Quantity):
                a strategically advantageous amount of time to wait in the case of an occulter for slew times
            det_mode (dict):
                Selected detection mode
        
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
        
        # selecting appropriate koMap
        koMap = self.koMaps[char_mode['syst']['name']]
        
        # allocate settling time + overhead time
        tmpCurrentTimeAbs = TK.currentTimeAbs.copy() + Obs.settlingTime + modes[0]['syst']['ohTime']
        tmpCurrentTimeNorm = TK.currentTimeNorm.copy() + Obs.settlingTime + modes[0]['syst']['ohTime']

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars)*u.d
        fZs = np.zeros(TL.nStars)/u.arcsec**2
        dV  = np.zeros(TL.nStars)*u.m/u.s
        intTimes = np.zeros(TL.nStars)*u.d
        all_intTimes = np.zeros(TL.nStars)*u.d
        tovisit = np.zeros(TL.nStars, dtype=bool)
        obsTimes = np.zeros([2,TL.nStars])*u.d
        sInds = np.arange(TL.nStars)
        all_sInds = np.array([])

        for mode in modes:
            # 2. find spacecraft orbital START positions (if occulter, positions 
            # differ for each star) and filter out unavailable targets 
            sd = None
            if OS.haveOcculter == True:
                sd        = Obs.star_angularSep(TL, old_sInd, sInds, tmpCurrentTimeAbs)
                obsTimes  = Obs.calculate_observableTimes(TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode)
                slewTimes = Obs.calculate_slewTimes(TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs)  
     
            # 2.1 filter out totTimes > integration cutoff
            if len(sInds.tolist()) > 0:
                mode_sInds = np.intersect1d(self.intTimeFilterInds, sInds)
                
            # start times, including slew times
            startTimes = tmpCurrentTimeAbs.copy() + slewTimes
            startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

            # 2.5 Filter stars not observable at startTimes
            try:
                koTimeInd = np.where(np.round(startTimes[0].value)-self.koTimes.value==0)[0][0]  # find indice where koTime is startTime[0]
                mode_sInds = mode_sInds[np.where(np.transpose(koMap)[koTimeInd].astype(bool)[mode_sInds])[0]]# filters inds by koMap #verified against v1.35
            except:#If there are no target stars to observe 
                mode_sInds = np.asarray([],dtype=int)
            
            # 3. filter out all previously (more-)visited targets, unless in 
            if len(mode_sInds.tolist()) > 0:
                mode_sInds = self.revisitFilter(mode_sInds, tmpCurrentTimeNorm)

            # 4.1 calculate integration times for ALL preselected targets
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
            maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed

            if len(mode_sInds.tolist()) > 0:
                if OS.haveOcculter == True and old_sInd is not None:
                    mode_sInds,slewTimes[mode_sInds],intTimes[mode_sInds],dV[mode_sInds] = self.refineOcculterSlews(old_sInd, mode_sInds, slewTimes, obsTimes, sd, mode)  
                    endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
                else:                
                    intTimes[mode_sInds] = self.calc_targ_intTime(mode_sInds, startTimes[mode_sInds], mode)
                    mode_sInds = mode_sInds[np.where(intTimes[mode_sInds] <= maxIntTime)]  # Filters targets exceeding end of OB
                    endTimes = startTimes + intTimes
                    
                    if maxIntTime.value <= 0:
                        mode_sInds = np.asarray([],dtype=int)

            for t in mode_sInds:
                if (intTimes[t] < all_intTimes[t] and intTimes[t] > 0) or all_intTimes[t] == 0:
                    all_intTimes[t] = intTimes[t]

            # 5.1 TODO Add filter to filter out stars entering and exiting keepout between startTimes and endTimes
            
            # 5.2 find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(mode_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                try: # endTimes may exist past koTimes so we have an exception to hand this case
                    koTimeInd = np.where(np.round(endTimes[0].value)-self.koTimes.value==0)[0][0]#koTimeInd[0][0]  # find indice where koTime is endTime[0]
                    mode_sInds = mode_sInds[np.where(np.transpose(koMap)[koTimeInd].astype(bool)[mode_sInds])[0]]# filters inds by koMap #verified against v1.35
                except:
                    mode_sInds = np.asarray([],dtype=int)

            all_sInds = np.concatenate([all_sInds, mode_sInds]).astype(int)

        blue_modes = [mode for mode in modes if mode['systName'][-1] == 'b']
        sInds = np.unique(all_sInds)
        det_mode = copy.deepcopy(blue_modes[0])
        
        # 6. choose best target from remaining
        if len(sInds) > 0:
            # choose sInd of next target
            sInd, waitTime = self.choose_next_target(old_sInd, sInds, slewTimes, all_intTimes[sInds])

            if sInd == None and waitTime is not None:#Should Choose Next Target decide there are no stars it wishes to observe at this time.
                self.vprint('There are no stars Choose Next Target would like to Observe. Waiting %dd'%waitTime.value)
                return DRM, None, None, waitTime, None
            elif sInd == None and waitTime == None:
                self.vprint('There are no stars Choose Next Target would like to Observe and waitTime is None')
                return DRM, None, None, waitTime, None

            s_IWA_OWA = (PP.arange * np.sqrt(TL.L[sInd])/TL.dist[sInd]).value*u.arcsec
            for bmode in blue_modes:
                intTime = self.calc_targ_intTime(sInd, startTimes[sInd], bmode)[0]
                if intTime != 0.0*u.d:
                    if s_IWA_OWA[0] < bmode['IWA'] < s_IWA_OWA[1] or s_IWA_OWA[0] < bmode['OWA'] < s_IWA_OWA[1]:
                        b_overlap = max(0, min(s_IWA_OWA[1], bmode['OWA']) - max(s_IWA_OWA[0], bmode['IWA']))
                        d_overlap = max(0, min(s_IWA_OWA[1], det_mode['OWA']) - max(s_IWA_OWA[0], det_mode['IWA']))
                        if b_overlap > d_overlap:
                            det_mode = copy.deepcopy(bmode)
                        elif b_overlap == d_overlap:
                            if (bmode['OWA'] - bmode['IWA']) > (det_mode['OWA'] - det_mode['IWA']):
                                det_mode = copy.deepcopy(bmode)
            r_mode = [mode for mode in modes if mode['systName'][-1] == 'r' and mode['systName'][-2] == det_mode['systName'][-2]][0]
            if self.WAint[sInd] > r_mode['IWA'] and self.WAint[sInd] < r_mode['OWA']:
                det_mode['BW'] = det_mode['BW'] + r_mode['BW']
                det_mode['OWA'] = r_mode['OWA']
                det_mode['inst']['sread'] = det_mode['inst']['sread'] + r_mode['inst']['sread']
                det_mode['inst']['idark'] = det_mode['inst']['idark'] + r_mode['inst']['idark']
                det_mode['inst']['CIC'] = det_mode['inst']['CIC'] + r_mode['inst']['CIC']
                det_mode['syst']['optics'] = np.mean((det_mode['syst']['optics'], r_mode['syst']['optics']))
                det_mode['instName'] = det_mode['instName'] + '_combined'
            intTime = self.calc_targ_intTime(sInd, startTimes[sInd], det_mode)[0]

        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint('No Observable Targets at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
            return DRM, None, None, None, None
    
        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]
        
        # populate DRM with occulter related values
        if OS.haveOcculter:
            DRM = Obs.log_occulterResults(DRM,slewTimes[sInd],sInd,sd[sInd],dV[sInd])
            return DRM, sInd, intTime, waitTime, det_mode

        return DRM, sInd, intTime, waitTime, det_mode

