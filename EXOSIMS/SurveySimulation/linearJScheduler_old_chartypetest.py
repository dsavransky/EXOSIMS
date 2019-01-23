from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import time
import astropy.constants as const

class linearJScheduler_old_chartypetest(SurveySimulation):
    """linearJScheduler 
    
    This class implements the linear cost function scheduler described
    in Savransky et al. (2010).
    
        Args:
        coeffs (iterable 3x1):
            Cost function coefficients: slew distance, completeness, target list coverage
        
        \*\*specs:
            user specified values
    
    """

    def __init__(self, coeffs=[1,1,2,1], revisit_wait=91.25, **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 4x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 4):
            raise TypeError("coeffs must be a 4 element iterable")

        #Add to outspec
        self._outspec['coeffs'] = coeffs
        self._outspec['revisit_wait'] = revisit_wait
        
        # normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs, ord=1)
        
        self.coeffs = coeffs

        self.revisit_wait = revisit_wait*u.d
        self.no_dets = np.ones(self.TargetList.nStars, dtype=bool)
        self.det_only_flag = {}


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
                        + 'mission time at Obs start: %s')%(ObsNum, sInd, TL.nStars, len(pInds), 
                        TK.currentTimeNorm.to('day').copy().round(2))
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

                for pInd in pInds[(detected == 1)]:
                    if self.det_only_flag.get(pInd) is not None:
                        self.det_only_flag[pInd] = self.det_only_flag.get(pInd) + [TK.currentTimeNorm.copy()]

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                do_char = True
                sp = SU.s[pInds]
                Ms = TL.MsTrue[sInd]
                Mp = SU.Mp[pInds]
                mu = const.G*(Mp + Ms)
                T = (2.*np.pi*np.sqrt(sp**3/mu)).to('d')
                for indx, pInd in enumerate(pInds):
                    if self.det_only_flag.get(pInd) is not None and len(self.det_only_flag.get(pInd)) < 3:
                        do_char = False
                    elif (self.det_only_flag.get(pInd) is not None 
                          and len(self.det_only_flag.get(pInd)) >= 3 
                          and (self.det_only_flag.get(pInd)[-1] - self.det_only_flag.get(pInd)[0]) < T[indx]/2):
                        do_char = False

                if char_mode['SNR'] not in [0, np.inf] and do_char:
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime, char_mode = \
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

                DRM['exoplanetObsTime'] = TK.exoplanetObsTime.copy()
                
                # append result values to self.DRM
                self.DRM.append(DRM)

                # handle case of inf OBs and missionPortion < 1
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1):
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
                    observableTimes = Obs.calculate_observableTimes(TL,np.arange(TL.nStars),startTimes,self.koMap,self.koTimes,self.mode)[0]
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
            waitTime (astropy Quantity):
                a strategically advantageous amount of time to wait in the case of an occulter for slew times
        
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
        tmpCurrentTimeAbs = TK.currentTimeAbs.copy() + Obs.settlingTime + mode['syst']['ohTime']
        tmpCurrentTimeNorm = TK.currentTimeNorm.copy() + Obs.settlingTime + mode['syst']['ohTime']

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
            obsTimes  = Obs.calculate_observableTimes(TL,sInds,tmpCurrentTimeAbs,self.koMap,self.koTimes,mode)
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
            sInds = sInds[np.where(np.transpose(self.koMap)[koTimeInd].astype(bool)[sInds])[0]]# filters inds by koMap #verified against v1.35
        except:#If there are no target stars to observe 
            sInds = np.asarray([],dtype=int)
        
        # 3. filter out all previously (more-)visited targets, unless in 
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # 4.1 calculate integration times for ALL preselected targets
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            # if OS.haveOcculter == True and old_sInd is not None:
            #     sInds,slewTimes[sInds],intTimes[sInds],dV[sInds] = self.refineOcculterSlews( old_sInd, sInds, slewTimes, obsTimes, sd, mode)  
            #     endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
            # else:                
            intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)
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
                sInds = sInds[np.where(np.transpose(self.koMap)[koTimeInd].astype(bool)[sInds])[0]]# filters inds by koMap #verified against v1.35
            except:
                sInds = np.asarray([],dtype=int)
        
        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            sInd, waitTime = self.choose_next_target(old_sInd, sInds, slewTimes, intTimes[sInds])
            
            if sInd == None and waitTime is not None:#Should Choose Next Target decide there are no stars it wishes to observe at this time.
                self.vprint('There are no stars Choose Next Target would like to Observe. Waiting %dd'%waitTime.value)
                return DRM, None, None, waitTime
            elif sInd == None and waitTime == None:
                self.vprint('There are no stars Choose Next Target would like to Observe and waitTime is None')
                return DRM, None, None, waitTime
            # store selected star integration time
            intTime = intTimes[sInd]
        
        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint('No Observable Targets at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
            return DRM, None, None, None
    
        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]
        
        # populate DRM with occulter related values
        if OS.haveOcculter == True:
            DRM = Obs.log_occulterResults(DRM,slewTimes[sInd],sInd,sd[sInd],dV[sInd])
            return DRM, sInd, intTime, slewTimes[sInd]
        
        return DRM, sInd, intTime, waitTime


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
            waitTime (astropy Quantity):
                the amount of time to wait (this method returns None)
        
        """

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        Obs = self.Observatory
        allModes = OS.observingModes

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)

        # current star has to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds, old_sInd)

        # calculate dt since previous observation
        dt = TK.currentTimeNorm.copy() + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)

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
            r_ts = TL.starprop(sInds, TK.currentTimeAbs.copy())
            u_ts = (r_ts.value.T/np.linalg.norm(r_ts, axis=1)).T
            angdists = np.arccos(np.clip(np.dot(u_ts, u_ts.T), -1, 1))
            A[np.ones((nStars), dtype=bool)] = angdists
            A = self.coeffs[0]*(A)/np.pi

        # add factor due to completeness
        A = A + self.coeffs[1]*(1 - comps)

        # add factor due to unvisited ramp
        f_uv = np.zeros(nStars)
        unvisited = self.starVisits[sInds]==0
        f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife.copy())**2
        A = A - self.coeffs[2]*f_uv

        # add factor due to revisited ramp
        f2_uv = 1 - (np.in1d(sInds, self.starRevisit[:,0]))
        A = A + self.coeffs[3]*f2_uv

        # kill diagonal
        A = A + np.diag(np.ones(nStars)*np.Inf)

        # take two traversal steps
        step1 = np.tile(A[sInds==old_sInd,:], (nStars, 1)).flatten('F')
        step2 = A[np.array(np.ones((nStars, nStars)), dtype=bool)]
        tmp = np.nanargmin(step1 + step2)
        sInd = sInds[int(np.floor(tmp/float(nStars)))]

        waitTime = slewTimes[sInd]
        #Check if exoplanetObsTime would be exceeded
        mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed
        intTimes2 = self.calc_targ_intTime(sInd, TK.currentTimeAbs.copy(), mode)
        if intTimes2 > maxIntTime: # check if max allowed integration time would be exceeded
            self.vprint('max allowed integration time would be exceeded')
            sInd = None
            waitTime = 1.*u.d
        
        return sInd, waitTime

    def revisitFilter(self, sInds, tmpCurrentTimeNorm):
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
                    & (self.starVisits[sInds] < self.nVisitsMax))# Checks that no star has exceeded the number of revisits
            if self.starRevisit.size != 0:#There is at least one revisit planned in starRevisit
                dt_rev = self.starRevisit[:,1]*u.day - tmpCurrentTimeNorm#absolute temporal spacing between revisit and now.

                #return indices of all revisits within a threshold dt_max of revisit day and indices of all revisits with no detections past the revisit time
                # ind_rev = [int(x) for x in self.starRevisit[np.abs(dt_rev) < self.dt_max, 0] if (x in sInds and self.no_dets[int(x)] == False)]
                # ind_rev2 = [int(x) for x in self.starRevisit[dt_rev < 0*u.d, 0] if (x in sInds and self.no_dets[int(x)] == True)]
                # tovisit[ind_rev] = (self.starVisits[ind_rev] < self.nVisitsMax)#IF duplicates exist in ind_rev, the second occurence takes priority
                ind_rev2 = [int(x) for x in self.starRevisit[dt_rev < 0*u.d, 0] if (x in sInds)]
                tovisit[ind_rev2] = (self.starVisits[ind_rev2] < self.nVisitsMax)
            sInds = np.where(tovisit)[0]

        return sInds

    def scheduleRevisit(self, sInd, smin, det, pInds):
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
        if smin is not None and smin is not np.nan: #smin is None if no planet was detected
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75*T

        # if no detections then schedule revisit based off of revisit_weight
        if not np.any(det):
            t_rev = TK.currentTimeNorm.copy() + self.revisit_wait
            self.no_dets[sInd] = True
        else:
            self.no_dets[sInd] = False

        t_rev = TK.currentTimeNorm.copy() + self.revisit_wait
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

        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]
        FA = (len(det) == len(pInds) + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # check if any of the detected planets were earthlike
        pIndsDet_earthlike = pIndsDet[np.where((SU.Rp[pInds[det]].value > (.8*SU.a[pInds[det]]**-.5).value)
                                                & (SU.Rp[pInds[det]] < 1.4*u.earthRad) 
                                                & (SU.a[pInds[det]] > .96*u.AU) 
                                                & (SU.a[pInds[det]] < 1.67*u.AU))[0]]
        if np.any(pIndsDet_earthlike):
            # if there is an earthlike planet, then switch to second charmode
            mode = spectroModes[1]
        else:
            mode = spectroModes[0]
        
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
            startTime = TK.currentTimeAbs.copy() + mode['syst']['ohTime'] + Obs.settlingTime
            startTimeNorm = TK.currentTimeNorm.copy() + mode['syst']['ohTime'] + Obs.settlingTime
            # planets to characterize
            tochar[tochar] = Obs.keepout(TL, sInd, startTime)
        
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
            tochar[tochar] = Obs.keepout(TL, sInd, endTimes[tochar])
        
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
                fZs = np.zeros(self.ntFlux)/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime/float(self.ntFlux)
                timePlus = Obs.settlingTime.copy() + mode['syst']['ohTime'].copy()#accounts for the time since the current time
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    timePlus += dt
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
                    timePlus += dt

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

            # if an earthlike planet is characterized, flag it
            char_earthlike = np.intersect1d(pInds[charplans == 1], pIndsDet_earthlike)
            if np.any(char_earthlike):
                for pind in char_earthlike:
                    self.det_only_flag[pind] = [currentTimeNorm]

        return characterized.astype(int), fZ, systemParams, SNR, intTime, mode

