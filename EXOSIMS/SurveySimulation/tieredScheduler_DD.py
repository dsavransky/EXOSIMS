from EXOSIMS.SurveySimulation.tieredScheduler import tieredScheduler
import EXOSIMS, os
import astropy.units as u
import astropy.constants as const
import numpy as np
import itertools
from scipy import interpolate
try:
    import cPickle as pickle
except:
    import pickle
import time
import copy
from EXOSIMS.util.deltaMag import deltaMag

class tieredScheduler_DD(tieredScheduler):
    """tieredScheduler_DD - tieredScheduler Dual Detection
    
    This class implements a version of the tieredScheduler that performs dual-band
    detections
    """

    def __init__(self, **specs):
        
        tieredScheduler.__init__(self, **specs)
        self.inttime_predict = 0*u.d
        

    def run_sim(self):
        """Performs the survey simulation 
        
        Returns:
            mission_end (string):
                Message printed at the end of a survey simulation.
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        Comp = self.Completeness
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        det_modes = list(filter(lambda mode: 'imag' in mode['inst']['name'], OS.observingModes))
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = OS.observingModes[0]
        
        # Begin Survey, and loop until mission is finished
        self.logger.info('OB{}: survey beginning.'.format(TK.OBnumber+1))
        self.vprint('OB{}: survey beginning.'.format(TK.OBnumber+1))
        t0 = time.time()
        sInd = None
        occ_sInd = None
        cnt = 0

        while not TK.mission_is_over(OS, Obs, det_modes[0]):
             
            # Acquire the NEXT TARGET star index and create DRM
            prev_occ_sInd = occ_sInd
            old_sInd = sInd #used to save sInd if returned sInd is None
            waitTime = None
            DRM, sInd, occ_sInd, t_det, sd, occ_sInds, det_mode = self.next_target(sInd, occ_sInd, det_modes, char_mode)
            
            if det_mode is not None:
                true_t_det = t_det*det_mode['timeMultiplier'] + Obs.settlingTime + det_mode['syst']['ohTime']
            else:
                true_t_det = t_det

            if sInd != occ_sInd and sInd is not None:
                assert t_det != 0, "Integration time can't be 0."

            if sInd is not None and (TK.currentTimeAbs.copy() + true_t_det) >= self.occ_arrives and occ_sInd != self.last_chard:
                sInd = occ_sInd
            if sInd == occ_sInd:
                self.ready_to_update = True

            time2arrive = self.occ_arrives - TK.currentTimeAbs.copy()

            if sInd is not None:
                cnt += 1

                # clean up revisit list when one occurs to prevent repeats
                if np.any(self.starRevisit) and np.any(np.where(self.starRevisit[:,0] == float(sInd))):
                    s_revs = np.where(self.starRevisit[:,0] == float(sInd))[0]
                    t_revs = np.where(self.starRevisit[:,1]*u.day - TK.currentTimeNorm.copy() < 0*u.d)[0]
                    self.starRevisit = np.delete(self.starRevisit, np.intersect1d(s_revs,t_revs),0)

                # get the index of the selected target for the extended list
                if TK.currentTimeNorm.copy() > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)
                
                # Beginning of observation, start to populate DRM
                DRM['OB_nb'] = TK.OBnumber+1
                DRM['ObsNum'] = cnt
                DRM['star_ind'] = sInd
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()

                if sInd == occ_sInd:
                    # wait until expected arrival time is observed
                    if time2arrive > 0*u.d:
                        TK.advanceToAbsTime(self.occ_arrives)
                        if time2arrive > 1*u.d:
                            self.GAtime = self.GAtime + time2arrive.to('day')

                TK.obsStart = TK.currentTimeNorm.copy().to('day')

                self.logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))
                self.vprint('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))

                DRM['arrival_time'] = TK.currentTimeNorm.copy().to('day')

                if sInd != occ_sInd:
                    self.starVisits[sInd] += 1
                    # PERFORM DETECTION and populate revisit list attribute.
                    # First store fEZ, dMag, WA
                    if np.any(pInds):
                        DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                        DRM['det_dMag'] = SU.dMag[pInds].tolist()
                        DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, det_mode)

                    if np.any(detected):
                        self.sInd_detcounts[sInd] += 1
                        self.sInd_dettimes[sInd] = (self.sInd_dettimes.get(sInd) or []) + [TK.currentTimeNorm.copy().to('day')]
                        self.vprint('  Det. results are: %s'%(detected))

                    # update GAtime
                    self.GAtime = self.GAtime + t_det.to('day')*self.GA_simult_det_fraction
                    self.tot_dettime += t_det.to('day')

                    # populate the DRM with detection results
                    DRM['det_time'] = t_det.to('day')
                    DRM['det_status'] = detected
                    DRM['det_SNR'] = det_SNR
                    DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                    DRM['det_params'] = det_systemParams
                    DRM['FA_det_status'] = int(FA)

                    det_comp = Comp.comp_per_intTime(t_det, TL, sInd, det_fZ, self.ZodiacalLight.fEZ0, self.WAint[sInd], det_mode)[0]
                    DRM['det_comp'] = det_comp
                    DRM['det_mode'] = dict(det_mode)
                    del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                
                elif sInd == occ_sInd:
                    self.occ_starVisits[occ_sInd] += 1
                    self.last_chard = occ_sInd
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    occ_pInds = np.where(SU.plan2star == occ_sInd)[0]

                    DRM['slew_time'] = self.occ_slewTime.to('day').value
                    DRM['slew_angle'] = self.occ_sd.to('deg').value
                    slew_mass_used = self.occ_slewTime*Obs.defburnPortion*Obs.flowRate
                    DRM['slew_dV'] = (self.occ_slewTime*self.ao*Obs.defburnPortion).to('m/s').value
                    DRM['slew_mass_used'] = slew_mass_used.to('kg')
                    Obs.scMass = Obs.scMass - slew_mass_used
                    DRM['scMass'] = Obs.scMass.to('kg')

                    self.logger.info('  Starshade and telescope aligned at target star')
                    self.vprint('  Starshade and telescope aligned at target star')

                    # PERFORM CHARACTERIZATION and populate spectra list attribute
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                            self.observation_characterization(sInd, char_mode)
                    if np.any(characterized):
                        self.vprint('  Char. results are: %s'%(characterized))
                    else:
                        # make sure we don't accidentally double characterize
                        TK.advanceToAbsTime(TK.currentTimeAbs.copy() + .01*u.d)
                    assert char_intTime != 0.0*u.d, "Integration time can't be 0."
                    if np.any(occ_pInds):
                        DRM['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                        DRM['char_dMag'] = SU.dMag[occ_pInds].tolist()
                        DRM['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                    DRM['char_mode'] = dict(char_mode)
                    del DRM['char_mode']['inst'], DRM['char_mode']['syst']

                    # update the occulter wet mass
                    if OS.haveOcculter and char_intTime is not None:
                        DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
                        char_comp = Comp.comp_per_intTime(char_intTime, TL, occ_sInd, char_fZ, self.ZodiacalLight.fEZ0, self.WAint[occ_sInd], char_mode)[0]
                        DRM['char_comp'] = char_comp
                    FA = False
                    # populate the DRM with characterization results
                    DRM['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                    #DRM['char_counts'] = self.sInd_charcounts[sInd]
                    DRM['char_status'] = characterized[:-1] if FA else characterized
                    DRM['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                    DRM['char_fZ'] = char_fZ.to('1/arcsec2')
                    DRM['char_params'] = char_systemParams
                    # populate the DRM with FA results
                    DRM['FA_det_status'] = int(FA)
                    DRM['FA_char_status'] = characterized[-1] if FA else 0
                    DRM['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                    DRM['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 if FA else 0./u.arcsec**2
                    DRM['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                    DRM['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec if FA else 0.*u.arcsec

                    # add star back into the revisit list
                    if np.any(characterized):
                        char = np.where(characterized)[0]
                        pInds = np.where(SU.plan2star == sInd)[0]
                        smin = np.min(SU.s[pInds[char]])
                        pInd_smin = pInds[np.argmin(SU.s[pInds[char]])]

                        Ms = TL.MsTrue[sInd]
                        sp = smin
                        Mp = SU.Mp[pInd_smin]
                        mu = const.G*(Mp + Ms)
                        T = 2.*np.pi*np.sqrt(sp**3/mu)
                        t_rev = TK.currentTimeNorm.copy() + T/2.

                self.goal_GAtime = self.GA_percentage * TK.currentTimeNorm.copy().to('day')
                goal_GAdiff = self.goal_GAtime - self.GAtime

                # allocate extra time to GA if we are falling behind
                if goal_GAdiff > 1*u.d and TK.currentTimeAbs.copy() < self.occ_arrives:
                    GA_diff = min(self.occ_arrives - TK.currentTimeAbs.copy(), goal_GAdiff)
                    self.vprint('Allocating time %s to general astrophysics'%(GA_diff))
                    self.GAtime = self.GAtime + GA_diff
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + GA_diff)
                # allocate time if there is no target for the starshade
                elif goal_GAdiff > 1*u.d and (self.occ_arrives - TK.currentTimeAbs.copy()) < -5*u.d and not np.any(occ_sInds):
                    self.vprint('No Available Occulter Targets: Allocating time %s to general astrophysics'%(goal_GAdiff))
                    self.GAtime = self.GAtime + goal_GAdiff
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + goal_GAdiff)

                DRM['exoplanetObsTime'] = TK.exoplanetObsTime.copy()

                # Append result values to self.DRM
                self.DRM.append(DRM)

                # Calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.copy().to('day')

                # With prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1):
                    self.arbitrary_time_advancement(TK.currentTimeNorm.to('day').copy() - DRM['arrival_time'])

                # With occulter, if spacecraft fuel is depleted, exit loop
                if Obs.scMass < Obs.dryMass:
                    self.vprint('Total fuel mass exceeded at %s' %TK.obsEnd.round(2))
                    break

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

        else:
            dtsim = (time.time()-t0)*u.s
            mission_end = "Mission complete: no more time available.\n"\
                    + "Simulation duration: %s.\n" %dtsim.astype('int')\
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."

            self.logger.info(mission_end)
            self.vprint(mission_end)

            return mission_end

    def next_target(self, old_sInd, old_occ_sInd, det_modes, char_mode):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star for the telescope
            old_occ_sInd (integer):
                Index of the previous target star for the occulter
            det_modes (dict array):
                Selected observing mode for detection
            char_mode (dict):
                Selected observing mode for characterization
                
        Returns:
            DRM (dicts):
                Contains the results of survey simulation
            sInd (integer):
                Index of next target star. Defaults to None.
            occ_sInd (integer):
                Index of next occulter target star. Defaults to None.
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
        SU = self.SimulatedUniverse
        
        # Create DRM
        DRM = {}
        
        # selecting appropriate koMap
        occ_koMap = self.koMaps[char_mode['syst']['name']]
        koMap = self.koMaps[det_modes[0]['syst']['name']]

        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        assert OS.haveOcculter == True
        self.ao = Obs.thrust/Obs.scMass

        # Star indices that correspond with the given HIPs numbers for the occulter
        # XXX ToDo: print out HIPs that don't show up in TL
        HIP_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
        if TL.earths_only:
            HIP_sInds = np.union1d(HIP_sInds, self.promoted_stars).astype(int)
        sInd = None

        # Now, start to look for available targets
        while not TK.mission_is_over(OS, Obs, det_modes[0]):
            # allocate settling time + overhead time
            tmpCurrentTimeAbs = TK.currentTimeAbs.copy()
            tmpCurrentTimeNorm = TK.currentTimeNorm.copy()
            occ_tmpCurrentTimeAbs = TK.currentTimeAbs.copy()
            occ_tmpCurrentTimeNorm = TK.currentTimeNorm.copy()

            # 0 initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            dV = np.zeros(TL.nStars)*u.m/u.s
            intTimes = np.zeros(TL.nStars)*u.d
            occ_intTimes = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            occ_tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)

            # 1 Find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            sd = Obs.star_angularSep(TL, old_occ_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, char_mode)
            slewTimes = Obs.calculate_slewTimes(TL, old_occ_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs)

            # 2.1 filter out totTimes > integration cutoff
            if len(sInds) > 0:
                occ_sInds = np.intersect1d(self.occ_intTimeFilterInds, sInds)
            if len(sInds) > 0:
                sInds = np.intersect1d(self.intTimeFilterInds, sInds)

            # Starttimes based off of slewtime
            occ_startTimes = occ_tmpCurrentTimeAbs.copy() + slewTimes
            occ_startTimesNorm = occ_tmpCurrentTimeNorm.copy() + slewTimes

            startTimes = tmpCurrentTimeAbs.copy() + np.zeros(TL.nStars)*u.d
            startTimesNorm = tmpCurrentTimeNorm.copy()

            # 2.5 Filter stars not observable at startTimes
            try:
                tmpIndsbool = list()
                for i in np.arange(len(occ_sInds)):
                    koTimeInd = np.where(np.round(occ_startTimes[occ_sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                    tmpIndsbool.append(occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                sInds_occ_ko = occ_sInds[tmpIndsbool]
                occ_sInds = sInds_occ_ko[np.where(np.in1d(sInds_occ_ko, HIP_sInds))[0]]
                del tmpIndsbool
            except:#If there are no target stars to observe 
                sInds_occ_ko = np.asarray([],dtype=int)
                occ_sInds = np.asarray([],dtype=int)

            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(np.round(startTimes[sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                    tmpIndsbool.append(koMap[sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:#If there are no target stars to observe 
                sInds = np.asarray([],dtype=int)

            # 2.9 Occulter target promotion step
            occ_sInds = self.promote_coro_targets(occ_sInds, sInds_occ_ko)

            # 3 Filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if len(sInds.tolist()) > 0:
                sInds = self.revisitFilter(sInds, TK.currentTimeNorm.copy())

            # revisit list, with time after start
            if np.any(occ_sInds):
                occ_tovisit[occ_sInds] = (self.occ_starVisits[occ_sInds] == self.occ_starVisits[occ_sInds].min())
                if self.occ_starRevisit.size != 0:
                    dt_rev = TK.currentTimeNorm.copy() - self.occ_starRevisit[:,1]*u.day
                    ind_rev = [int(x) for x in self.occ_starRevisit[dt_rev > 0, 0] if x in occ_sInds]
                    occ_tovisit[ind_rev] = True
                occ_sInds = np.where(occ_tovisit)[0]

            # 4 calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, det_modes[0])
            maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife, OS.intCutoff)#Maximum intTime allowed

            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
            occ_maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife, OS.intCutoff)#Maximum intTime allowed
            if len(occ_sInds) > 0:
                if self.int_inflection:
                    fEZ = ZL.fEZ0
                    WA = self.WAint
                    occ_intTimes[occ_sInds] = self.calc_int_inflection(occ_sInds, fEZ, occ_startTimes, WA[occ_sInds], char_mode, ischar=True)
                    totTimes = occ_intTimes*char_mode['timeMultiplier']
                    occ_endTimes = occ_startTimes + totTimes
                else:
                    # characterization_start = occ_startTimes
                    occ_intTimes[occ_sInds] = self.calc_targ_intTime(occ_sInds, occ_startTimes[occ_sInds], char_mode) * (1 + self.charMargin)

                    # Adjust integration time for stars with known earths around them
                    for occ_star in occ_sInds:
                        if occ_star in self.promoted_stars:
                            occ_earths = np.intersect1d(np.where(SU.plan2star == occ_star)[0], self.known_earths).astype(int)
                            if np.any(occ_earths):
                                fZ = ZL.fZ(Obs, TL, occ_star, occ_startTimes[occ_star], char_mode)
                                fEZ = SU.fEZ[occ_earths].to('1/arcsec2').value/u.arcsec**2
                                if SU.lucky_planets:
                                    phi = (1/np.pi)*np.ones(len(SU.d))
                                    dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[occ_earths]                   # delta magnitude
                                    WA = np.arctan(SU.a/TL.dist[SU.plan2star]).to('arcsec')[occ_earths]   # working angle
                                else:
                                    dMag = SU.dMag[occ_earths]
                                    WA = SU.WA[occ_earths]

                                if np.all((WA < char_mode['IWA']) | (WA > char_mode['OWA'])):
                                    occ_intTimes[occ_star] = 0.*u.d
                                else:
                                    earthlike_inttimes = OS.calc_intTime(TL, occ_star, fZ, fEZ, dMag, WA, char_mode) * (1 + self.charMargin)
                                    earthlike_inttime = earthlike_inttimes[(earthlike_inttimes < occ_maxIntTime)]
                                    if len(earthlike_inttime) > 0:
                                        occ_intTimes[occ_star] = np.max(earthlike_inttime)
                                    else:
                                        occ_intTimes[occ_star] = np.max(earthlike_inttimes)
                    occ_endTimes = occ_startTimes + (occ_intTimes * char_mode['timeMultiplier']) + Obs.settlingTime + char_mode['syst']['ohTime']

                    occ_sInds = occ_sInds[(occ_intTimes[occ_sInds] <= occ_maxIntTime)]  # Filters targets exceeding maximum intTime
                    occ_sInds = occ_sInds[(occ_intTimes[occ_sInds] > 0.0*u.d)]  # Filters with an inttime of 0

                if occ_maxIntTime.value <= 0:
                    occ_sInds = np.asarray([],dtype=int)

            if len(sInds.tolist()) > 0:
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], det_modes[0])
                sInds = sInds[(intTimes[sInds] <= maxIntTime)]  # Filters targets exceeding end of OB
                endTimes = startTimes + intTimes + Obs.settlingTime + det_modes[0]['syst']['ohTime']
                
                if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)

            # 5.2 find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(occ_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                try: # endTimes may exist past koTimes so we have an exception to hand this case
                    tmpIndsbool = list()
                    for i in np.arange(len(occ_sInds)):
                        koTimeInd = np.where(np.round(occ_endTimes[occ_sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                        tmpIndsbool.append(occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                    occ_sInds = occ_sInds[tmpIndsbool]
                    del tmpIndsbool
                except:
                    occ_sInds = np.asarray([],dtype=int)

            if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                try: # endTimes may exist past koTimes so we have an exception to hand this case
                    tmpIndsbool = list()
                    for i in np.arange(len(sInds)):
                        koTimeInd = np.where(np.round(endTimes[sInds[i]].value)-self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                        tmpIndsbool.append(koMap[sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                    sInds = sInds[tmpIndsbool]
                    del tmpIndsbool
                except:
                    sInds = np.asarray([],dtype=int)

            # 5.3 Filter off current occulter target star from detection list
            if old_occ_sInd is not None:
                sInds = sInds[(sInds != old_occ_sInd)]
                occ_sInds = occ_sInds[(occ_sInds != old_occ_sInd)]

            # 6.1 Filter off any stars visited by the occulter 3 or more times
            if np.any(occ_sInds):
                occ_sInds = occ_sInds[(self.occ_starVisits[occ_sInds] < self.occ_max_visits)]

            # 6.2 Filter off coronograph stars with > 3 visits and no detections
            no_dets = np.logical_and((self.starVisits[sInds] > self.n_det_remove), (self.sInd_detcounts[sInds] == 0))
            sInds = sInds[np.where(np.invert(no_dets))[0]]

            max_dets = np.where(self.sInd_detcounts[sInds] < self.max_successful_dets)[0]
            sInds = sInds[max_dets]

            # 7 Filter off cornograph stars with too-long inttimes
            available_time = None
            if self.occ_arrives > TK.currentTimeAbs:
                available_time = self.occ_arrives - TK.currentTimeAbs.copy()
                if np.any(sInds[intTimes[sInds] < available_time]):
                    sInds = sInds[intTimes[sInds] < available_time]

            # 8 remove occ targets on ignore_stars list
            occ_sInds = np.setdiff1d(occ_sInds, np.intersect1d(occ_sInds, self.ignore_stars))

            tmpIndsbool = list()
            for i in np.arange(len(occ_sInds)):
                koTimeInd = np.where(np.round(occ_startTimes[occ_sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                tmpIndsbool.append(occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind

            t_det = 0*u.d
            det_mode = copy.deepcopy(det_modes[0])
            occ_sInd = old_occ_sInd

            # 9 Choose best target from remaining
            # if the starshade has arrived at its destination, or it is the first observation
            if np.any(occ_sInds):
                if old_occ_sInd is None or ((TK.currentTimeAbs.copy() + t_det) >= self.occ_arrives and self.ready_to_update):
                    occ_sInd = self.choose_next_occulter_target(old_occ_sInd, occ_sInds, occ_intTimes)
                    if old_occ_sInd is None:
                        self.occ_arrives = TK.currentTimeAbs.copy()
                    else:
                        self.occ_arrives = occ_startTimes[occ_sInd]
                        self.occ_slewTime = slewTimes[occ_sInd]
                        self.occ_sd = sd[occ_sInd]
                        self.inttime_predict = occ_intTimes[occ_sInd]
                    self.ready_to_update = False
                elif not np.any(sInds):
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + 1*u.d)
                    continue

            if occ_sInd is not None:
                sInds = sInds[np.where(sInds != occ_sInd)]

            if self.tot_det_int_cutoff < self.tot_dettime:
                sInds = np.array([])

            if np.any(sInds):

                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, intTimes[sInds])

                # Perform dual band detections if necessary
                if self.WAint[sInd] > det_modes[1]['IWA'] and self.WAint[sInd] < det_modes[1]['OWA']:
                    det_mode['BW'] = det_mode['BW'] + det_modes[1]['BW']
                    det_mode['inst']['sread'] = det_mode['inst']['sread'] + det_modes[1]['inst']['sread']
                    det_mode['inst']['idark'] = det_mode['inst']['idark'] + det_modes[1]['inst']['idark']
                    det_mode['inst']['CIC'] = det_mode['inst']['CIC'] + det_modes[1]['inst']['CIC']
                    det_mode['syst']['optics'] = np.mean((det_mode['syst']['optics'], det_modes[1]['syst']['optics']))
                    det_mode['instName'] = 'combined'

                t_det = self.calc_targ_intTime(np.array([sInd]), startTimes[sInd], det_mode)[0]

                if t_det > maxIntTime and maxIntTime > 0*u.d:
                    t_det = maxIntTime
                if available_time is not None and available_time > 0*u.d:
                    if t_det > available_time:
                        t_det = available_time.copy().value * u.d

            # if no observable target, call the TimeKeeping.wait() method
            if not np.any(sInds) and not np.any(occ_sInds):
                self.vprint('No Observable Targets at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
                return DRM, None, None, None, None, None, None
            break

        else:
            self.logger.info('Mission complete: no more time available')
            self.vprint('Mission complete: no more time available')
            return DRM, None, None, None, None, None, None

        if TK.mission_is_over(OS, Obs, det_mode):
            self.logger.info('Mission complete: no more time available')
            self.vprint('Mission complete: no more time available')
            return DRM, None, None, None, None, None, None

        return DRM, sInd, occ_sInd, t_det, sd, occ_sInds, det_mode

