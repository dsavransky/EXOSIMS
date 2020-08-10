from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools
import astropy.constants as const
import time
from EXOSIMS.util.get_module import get_module
import sys, logging
import numpy as np
import astropy.units as u
import astropy.constants as const
import random as py_random
import time
import json, os.path, copy, re, inspect, subprocess
import hashlib

class linearJScheduler_det_only_sotoSS(SurveySimulation):
    """linearJScheduler_det_only_sotoSS
    
    This class implements the linear cost function scheduler described
    in Savransky et al. (2010).
    
    Args:
        coeffs (iterable 4x1):
            Cost function coefficients: slew distance, completeness, target list coverage, revisit weight
        
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

                # populate the DRM with observation modes
                DRM['det_mode'] = dict(det_mode)
                del DRM['det_mode']['inst'], DRM['det_mode']['syst']

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
            print(log_end)


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
        
        # current star has to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds, old_sInd)
        
        # calculate dt since previous observation
        dt = TK.currentTimeNorm + slewTimes[sInds] - self.lastObsTimes[sInds]
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
            r_ts = TL.starprop(sInds, TK.currentTimeAbs)
            u_ts = (r_ts.to("AU").value.T/np.linalg.norm(r_ts.to("AU").value, axis=1)).T
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
        
        waitTime = slewTimes[sInd]
        #Check if exoplanetObsTime would be exceeded
        mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
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
            t_rev = TK.currentTimeNorm + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + 0.75*T

        t_rev = TK.currentTimeNorm.copy() + self.revisit_wait
        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:#If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])#initialize starRevisit
        else:
            revInd = np.where(self.starRevisit[:,0] == sInd)[0]#indices of the first column of the starRevisit list containing sInd
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd,1] = revisit[1]#over


