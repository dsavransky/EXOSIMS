from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import astropy.constants as const

class linearJScheduler_old(SurveySimulation):
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
        coeffs = coeffs/np.linalg.norm(coeffs)
        
        self.coeffs = coeffs

        self.revisit_wait = revisit_wait*u.d
        self.no_dets = np.ones(self.TargetList.nStars, dtype=bool)


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

        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
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
            return sInd, None
        
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
        tmp = np.argmin(step1 + step2)
        sInd = sInds[int(np.floor(tmp/float(nStars)))]
        
        return sInd, None

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

