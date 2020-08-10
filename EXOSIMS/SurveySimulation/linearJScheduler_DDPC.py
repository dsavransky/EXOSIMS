# -*- coding: utf-8 -*-
from EXOSIMS.SurveySimulation.linearJScheduler import linearJScheduler
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

    def __init__(self, revisit_weight=1.0, **specs):
        
        linearJScheduler.__init__(self, **specs)

        self._outspec['revisit_weight'] = revisit_weight

        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TL = self.TargetList

        allModes = OS.observingModes
        num_char_modes = len(list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes)))
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)

        self.revisit_weight = revisit_weight


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
        det_modes = list(filter(lambda mode: 'imag' in mode['inst']['name'], allModes))
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
                    if np.any(characterized):
                        self.vprint('  Char. results are: {}'.format(characterized[:-1, mode_index]))
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


    def next_target(self, old_sInd, modes):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            modes (dict):
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
        
        # create DRM
        DRM = {}
        
        # selecting appropriate koMap
        koMap = self.koMaps[modes[0]['syst']['name']]
        
        # allocate settling time + overhead time
        tmpCurrentTimeAbs = TK.currentTimeAbs.copy() + Obs.settlingTime + modes[0]['syst']['ohTime']
        tmpCurrentTimeNorm = TK.currentTimeNorm.copy() + Obs.settlingTime + modes[0]['syst']['ohTime']

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars)*u.d
        fZs = np.zeros(TL.nStars)/u.arcsec**2
        dV  = np.zeros(TL.nStars)*u.m/u.s
        intTimes = np.zeros(TL.nStars)*u.d
        obsTimes = np.zeros([2,TL.nStars])*u.d
        sInds = np.arange(TL.nStars)
        
        # 2. find spacecraft orbital START positions (if occulter, positions 
        # differ for each star) and filter out unavailable targets 
        sd = None
        if OS.haveOcculter == True:
            sd        = Obs.star_angularSep(TL, old_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes  = Obs.calculate_observableTimes(TL,sInds,tmpCurrentTimeAbs,self.koMaps,self.koTimes,modes[0])
            slewTimes = Obs.calculate_slewTimes(TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs)  
 
        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)
            
        # start times, including slew times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        # 2.5 Filter stars not observable at startTimes
        try:
            koTimeInd = np.where(np.round(startTimes[0].value)-self.koTimes.value==0)[0][0]  # find indice where koTime is startTime[0]
            sInds = sInds[np.where(np.transpose(koMap)[koTimeInd].astype(bool)[sInds])[0]]# filters inds by koMap #verified against v1.35
        except:#If there are no target stars to observe 
            sInds = np.asarray([],dtype=int)
        
        # 3. filter out all previously (more-)visited targets, unless in 
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # 4.1 calculate integration times for ALL preselected targets
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, modes[0])
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            # if OS.haveOcculter == True and old_sInd is not None:
            #     sInds,slewTimes[sInds],intTimes[sInds],dV[sInds] = self.refineOcculterSlews(old_sInd, sInds, slewTimes, obsTimes, sd, mode)  
            #     endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
            # else:                
            intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], modes[0])
            sInds = sInds[np.where(intTimes[sInds] <= maxIntTime)]  # Filters targets exceeding end of OB
            endTimes = startTimes + intTimes
            
            if maxIntTime.value <= 0:
                sInds = np.asarray([],dtype=int)

        # 5.1 TODO Add filter to filter out stars entering and exiting keepout between startTimes and endTimes
        
        # 5.2 find spacecraft orbital END positions (for each candidate target), 
        # and filter out unavailable targets
        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            try: # endTimes may exist past koTimes so we have an exception to hand this case
                koTimeInd = np.where(np.round(endTimes[0].value)-self.koTimes.value==0)[0][0]#koTimeInd[0][0]  # find indice where koTime is endTime[0]
                sInds = sInds[np.where(np.transpose(koMap)[koTimeInd].astype(bool)[sInds])[0]]# filters inds by koMap #verified against v1.35
            except:
                sInds = np.asarray([],dtype=int)

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            sInd, waitTime = self.choose_next_target(old_sInd, sInds, slewTimes, intTimes[sInds])
            
            if sInd == None and waitTime is not None:#Should Choose Next Target decide there are no stars it wishes to observe at this time.
                self.vprint('There are no stars Choose Next Target would like to Observe. Waiting %dd'%waitTime.value)
                return DRM, None, None, waitTime, None
            elif sInd == None and waitTime == None:
                self.vprint('There are no stars Choose Next Target would like to Observe and waitTime is None')
                return DRM, None, None, waitTime, None
            # store selected star integration time
            det_mode = copy.deepcopy(modes[0])
            if self.WAint[sInd] > modes[1]['IWA'] and self.WAint[sInd] < modes[1]['OWA']:
                det_mode['BW'] = det_mode['BW'] + modes[1]['BW']
                det_mode['OWA'] = modes[1]['OWA']
                det_mode['inst']['sread'] = det_mode['inst']['sread'] + modes[1]['inst']['sread']
                det_mode['inst']['idark'] = det_mode['inst']['idark'] + modes[1]['inst']['idark']
                det_mode['inst']['CIC'] = det_mode['inst']['CIC'] + modes[1]['inst']['CIC']
                det_mode['syst']['optics'] = np.mean((det_mode['syst']['optics'], modes[1]['syst']['optics']))
                det_mode['instName'] = 'combined'
                intTime = self.calc_targ_intTime(np.array([sInd]), startTimes[sInd], det_mode)[0]
            else:
                intTime = intTimes[sInd]
        
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


    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
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
        OS = self.OpticalSystem
        Obs = self.Observatory
        allModes = OS.observingModes

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        known_sInds = np.intersect1d(sInds, self.known_rocky)

        if OS.haveOcculter:
            # current star has to be in the adjmat
            if (old_sInd is not None) and (old_sInd not in sInds):
                sInds = np.append(sInds, old_sInd)
            
            # calculate dt since previous observation
            dt = TK.currentTimeNorm.copy() + slewTimes[sInds] - self.lastObsTimes[sInds]
            # get dynamic completeness values
            comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
            for idx, sInd in enumerate(sInds):
                if sInd in known_sInds:
                    comps[idx] = 1.0

            # if first target, or if only 1 available target, 
            # choose highest available completeness
            nStars = len(sInds)
            if (old_sInd is None) or (nStars == 1):
                sInd = np.random.choice(sInds[comps == max(comps)])
                return sInd, slewTimes[sInd]
            
            # define adjacency matrix
            A = np.zeros((nStars,nStars))
            
            # only consider slew distance when there's an occulter
            if OS.haveOcculter:
                r_ts = TL.starprop(sInds, TK.currentTimeAbs)
                u_ts = (r_ts.to('AU').value.T/np.linalg.norm(r_ts.to('AU').value, axis=1)).T
                angdists = np.arccos(np.clip(np.dot(u_ts, u_ts.T), -1, 1))
                A[np.ones((nStars), dtype=bool)] = angdists
                A = self.coeffs[0]*(A)/np.pi
            
            # add factor due to completeness
            A = A + self.coeffs[1]*(1 - comps)

            # add factor for unvisited ramp for known stars
            if np.any(known_sInds):
                 # add factor for least visited known stars
                f_uv = np.zeros(nStars)
                u1 = np.in1d(sInds, known_sInds)
                u2 = self.starVisits[sInds]==min(self.starVisits[known_sInds])
                unvisited = np.logical_and(u1, u2)
                f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife.copy())**2
                A = A - self.coeffs[2]*f_uv

                # add factor for unvisited known stars
                no_visits = np.zeros(nStars)
                u2 = self.starVisits[sInds]==0
                unvisited = np.logical_and(u1, u2)
                no_visits[unvisited] = 1.
                A = A - self.coeffs[3]*no_visits
            
            # add factor due to unvisited ramp
            f_uv = np.zeros(nStars)
            unvisited = self.starVisits[sInds]==0
            f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife.copy())**2
            A = A - self.coeffs[4]*f_uv

            # add factor due to revisited ramp
            # f2_uv = np.where(self.starVisits[sInds] > 0, 1, 0) *\
            #         (1 - (np.in1d(sInds, self.starRevisit[:,0],invert=True)))
            if self.starRevisit.size != 0:
                f2_uv = 1 - (np.in1d(sInds, self.starRevisit[:,0]))
                A = A + self.coeffs[5]*f2_uv
            
            # kill diagonal
            A = A + np.diag(np.ones(nStars)*np.Inf)
            
            # take two traversal steps
            step1 = np.tile(A[sInds==old_sInd,:], (nStars, 1)).flatten('F')
            step2 = A[np.array(np.ones((nStars, nStars)), dtype=bool)]
            tmp = np.nanargmin(step1 + step2)
            sInd = sInds[int(np.floor(tmp/float(nStars)))]

        else:
            nStars = len(sInds)

            # 1/ Choose next telescope target
            comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], TK.currentTimeNorm.copy())

            # add weight for star revisits
            ind_rev = []
            if self.starRevisit.size != 0:
                dt_rev = self.starRevisit[:,1]*u.day - TK.currentTimeNorm.copy()
                ind_rev = [int(x) for x in self.starRevisit[dt_rev < 0 , 0] if x in sInds]

            f2_uv = np.where((self.starVisits[sInds] > 0) & (self.starVisits[sInds] < self.nVisitsMax), 
                              self.starVisits[sInds], 0) * (1 - (np.in1d(sInds, ind_rev, invert=True)))

            weights = (comps + self.revisit_weight*f2_uv/float(self.nVisitsMax))/intTimes

            sInd = np.random.choice(sInds[weights == max(weights)])

        waitTime = slewTimes[sInd]
        #Check if exoplanetObsTime would be exceeded
        mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed
        intTimes2 = self.calc_targ_intTime(np.array([sInd]), TK.currentTimeAbs.copy(), mode)
        if intTimes2 > maxIntTime: # check if max allowed integration time would be exceeded
            self.vprint('max allowed integration time would be exceeded')
            sInd = None
            waitTime = 1.*u.d
        
        return sInd, waitTime


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
        
        # selecting appropriate koMap
        koMap = self.koMaps[modes[0]['syst']['name']]
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]

        pIndsDet = []
        tochars = []
        intTimes_all = []
        FA = (len(det) == len(pInds) + 1)
        is_earthlike = []

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

            # look for last detected planets that have not been fully characterized
            if (FA == False): # only true planets, no FA
                tochar = (self.fullSpectra[m_i][pIndsDet[m_i]] == 0)
            else: # mix of planets and a FA
                truePlans = pIndsDet[m_i][:-1]
                tochar = np.append((self.fullSpectra[m_i][truePlans] == 0), True)
        
            # 1/ find spacecraft orbital START position including overhead time,
            # and check keepout angle
            if np.any(tochar):
                # start times
                startTime = TK.currentTimeAbs.copy() + mode['syst']['ohTime'] + Obs.settlingTime
                startTimeNorm = TK.currentTimeNorm.copy() + mode['syst']['ohTime'] + Obs.settlingTime
                # planets to characterize
                koTimeInd = np.where(np.round(startTime.value)-self.koTimes.value==0)[0][0]  # find indice where koTime is startTime[0]
                #wherever koMap is 1, the target is observable
                tochar[tochar] = koMap[sInd][koTimeInd]

            # 2/ if any planet to characterize, find the characterization times
            # at the detected fEZ, dMag, and WA
            is_earthlike.append(np.logical_and(np.array([(p in self.earth_candidates) for p in pIndsDet[m_i]]), tochar))
            if np.any(tochar):
                fZ[m_i] = ZL.fZ(Obs, TL, sInd, startTime, mode)
                fEZ = self.lastDetected[sInd,1][det][tochar]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][det][tochar]
                WA = self.lastDetected[sInd,3][det][tochar]*u.arcsec
                WA[is_earthlike[m_i][tochar]] = SU.WA[pIndsDet[m_i][is_earthlike[m_i]]]
                dMag[is_earthlike[m_i][tochar]] = SU.dMag[pIndsDet[m_i][is_earthlike[m_i]]]

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

                tochars.append(tochar)
                intTimes_all.append(intTimes)
            else:
                tochar[tochar] = False
                tochars.append(tochar)
                intTimes_all.append(np.zeros(len(tochar))*u.day)

        # 4/ if yes, allocate the overhead time, and perform the characterization 
        # for the maximum char time
        if np.any(tochars):
            pIndsChar = []
            for m_i, mode in enumerate(modes):
                if len(pIndsDet[m_i]) > 0 and np.any(tochars[m_i]):
                    if intTime is None or np.max(intTimes_all[m_i][tochars[m_i]]) > intTime:
                        #Allocate Time
                        if np.any(np.logical_and(is_earthlike[m_i], tochars[m_i])):
                            intTime = np.max(intTimes_all[m_i][np.logical_and(is_earthlike[m_i], tochars[m_i])])
                        else:
                            intTime = np.max(intTimes_all[m_i][tochars[m_i]])
                    pIndsChar.append(pIndsDet[m_i][tochars[m_i]])
                    log_char = '   - Charact. planet inds %s (%s/%s detected)'%(pIndsChar[m_i], 
                            len(pIndsChar[m_i]), len(pIndsDet[m_i]))
                    self.logger.info(log_char)
                    self.vprint(log_char)
                else:
                    pIndsChar.append([])

            if intTime is not None:
                extraTime = intTime*(modes[0]['timeMultiplier'] - 1.)#calculates extraTime
                success = TK.allocate_time(intTime + extraTime + modes[0]['syst']['ohTime'] + Obs.settlingTime, True)#allocates time
                if success == False: #Time was not successfully allocated
                    #Identical to when "if char_mode['SNR'] not in [0, np.inf]:" in run_sim()
                    return(characterizeds, fZ, systemParams, SNR, None)
            
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
                timePlus = Obs.settlingTime.copy() + modes[0]['syst']['ohTime'].copy()#accounts for the time since the current time
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    timePlus += dt/2.
                    fZs[i,0] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy() + timePlus, modes[0])[0]
                    fZs[i,1] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy() + timePlus, modes[1])[0]
                    SU.propag_system(sInd, TK.currentTimeNorm.copy() + timePlus - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm.copy() + timePlus
                    systemParamss[i] = SU.dump_system_params(sInd)
                    Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, modes[0], fZ=fZs[i,0])
                    Ss2[i,:], Ns2[i,:] = self.calc_signal_noise(sInd, planinds2, dt, modes[1], fZ=fZs[i,1])

                    # allocate second half of dt
                    timePlus += dt/2.
                
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
                    fZ[m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy(), mode)[0]
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


