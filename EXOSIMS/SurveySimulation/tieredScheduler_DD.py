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

        self.phase1_end = TK.missionStart + 365*u.d
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        detModes = filter(lambda mode: 'imag' in mode['inst']['name'], OS.observingModes)
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        if np.any(spectroModes):
            charMode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            charMode = OS.observingModes[0]
        
        # Begin Survey, and loop until mission is finished
        self.logger.info('OB%s: survey beginning.'%(TK.OBnumber+1))
        print 'OB%s: survey beginning.'%(TK.OBnumber+1)
        t0 = time.time()
        sInd = None
        occ_sInd = None
        cnt = 0
        self.occ_arrives = TK.currentTimeAbs
        while not TK.mission_is_over():
             
            # Acquire the NEXT TARGET star index and create DRM
            prev_occ_sInd = occ_sInd
            DRM, sInd, occ_sInd, t_det, sd, occ_sInds, dmode = self.next_target(sInd, occ_sInd, detModes, charMode)
            if sInd != occ_sInd:
                assert t_det !=0, "Integration time can't be 0."

            if sInd is not None and (TK.currentTimeAbs + t_det) >= self.occ_arrives and np.any(occ_sInds):
                sInd = occ_sInd
                self.ready_to_update = True

            time2arrive = self.occ_arrives - TK.currentTimeAbs
            
            if sInd is not None:
                cnt += 1

                # clean up revisit list when one occurs to prevent repeats
                if np.any(self.starRevisit) and np.any(np.where(self.starRevisit[:,0] == float(sInd))):
                    s_revs = np.where(self.starRevisit[:,0] == float(sInd))[0]
                    dt_max = 1.*u.week
                    t_revs = np.where(self.starRevisit[:,1]*u.day - TK.currentTimeNorm < dt_max)[0]
                    self.starRevisit = np.delete(self.starRevisit, np.intersect1d(s_revs,t_revs),0)

                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)
                
                # Beginning of observation, start to populate DRM
                DRM['OB#'] = TK.OBnumber+1
                DRM['Obs#'] = cnt
                DRM['star_ind'] = sInd
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()

                if sInd == occ_sInd:
                    # wait until expected arrival time is observed
                    if time2arrive > 0*u.d:
                        TK.allocate_time(time2arrive.to('day'))
                        if time2arrive > 1*u.d:
                            self.GAtime = self.GAtime + time2arrive.to('day')

                TK.obsStart = TK.currentTimeNorm.to('day')

                self.logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))
                print '  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2))
                
                if sInd != occ_sInd:
                    # PERFORM DETECTION and populate revisit list attribute.
                    # First store fEZ, dMag, WA
                    if np.any(pInds):
                        DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                        DRM['det_dMag'] = SU.dMag[pInds].tolist()
                        DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, dmode)

                    # update GAtime
                    self.GAtime = self.GAtime + t_det.to('day')*.07

                    DRM['det_time'] = t_det.to('day')

                    if np.any(detected):
                        print '  Det. results are: %s'%(detected)
                    # populate the DRM with detection results

                    DRM['det_status'] = detected
                    DRM['det_SNR'] = det_SNR
                    DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                    DRM['det_params'] = det_systemParams
                    DRM['det_mode'] = dict(dmode)
                    DRM['FA_det_status'] = int(FA)

                    det_comp = Comp.comp_per_intTime(t_det, TL, sInd, det_fZ, self.ZodiacalLight.fEZ0, self.WAint[sInd], detMode)[0]
                    DRM['det_comp'] = det_comp

                    del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                
                elif sInd == occ_sInd:
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    # First store fEZ, dMag, WA, and characterization mode

                    # clean up revisit list when one occurs to prevent repeats
                    if np.any(self.occ_starRevisit) and np.any(np.where(self.occ_starRevisit[:,0] == float(occ_sInd))):
                        s_revs = np.where(self.occ_starRevisit[:,0] == float(occ_sInd))[0]
                        dt_max = 1.*u.week
                        t_revs = np.where(self.occ_starRevisit[:,1]*u.day - TK.currentTimeNorm < dt_max)[0]
                        self.occ_starRevisit = np.delete(self.occ_starRevisit, np.intersect1d(s_revs, t_revs),0)

                    occ_pInds = np.where(SU.plan2star == occ_sInd)[0]
                    sInd = occ_sInd

                    DRM['slew_time'] = self.occ_slewTime.to('day').value
                    DRM['slew_angle'] = self.occ_sd.to('deg').value
                    slew_mass_used = self.occ_slewTime*Obs.defburnPortion*Obs.flowRate
                    DRM['slew_dV'] = (self.occ_slewTime*self.ao*Obs.defburnPortion).to('m/s').value
                    DRM['slew_mass_used'] = slew_mass_used.to('kg')
                    Obs.scMass = Obs.scMass - slew_mass_used
                    DRM['scMass'] = Obs.scMass.to('kg')

                    self.logger.info('  Starshade and telescope aligned at target star')
                    print '  Starshade and telescope aligned at target star'
                    if np.any(occ_pInds):
                        DRM['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                        DRM['char_dMag'] = SU.dMag[occ_pInds].tolist()
                        DRM['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                    DRM['char_mode'] = dict(charMode)
                    del DRM['char_mode']['inst'], DRM['char_mode']['syst']

                     # PERFORM CHARACTERIZATION and populate spectra list attribute
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                            self.observation_characterization(sInd, charMode)
                    if np.any(characterized):
                        print '  Char. results are: %s'%(characterized)
                    assert char_intTime != 0, "Integration time can't be 0."
                    # update the occulter wet mass
                    if OS.haveOcculter == True and char_intTime is not None:
                        DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
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

                    char_comp = Comp.comp_per_intTime(char_intTime, TL, occ_sInd, char_fZ, self.ZodiacalLight.fEZ0, self.WAint[occ_sInd], charMode)[0]
                    DRM['char_comp'] = char_comp

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
                        t_rev = TK.currentTimeNorm + T/2.
                        revisit = np.array([sInd, t_rev.to('day').value])
                        if self.occ_starRevisit.size == 0:
                            self.occ_starRevisit = np.array([revisit])
                        else:
                            self.occ_starRevisit = np.vstack((self.occ_starRevisit, revisit))

                self.goal_GAtime = self.GA_percentage * TK.currentTimeNorm.to('day')
                goal_GAdiff = self.goal_GAtime - self.GAtime

                # allocate extra time to GA if we are falling behind
                if goal_GAdiff > 1*u.d and goal_GAdiff < time2arrive.to('day'):
                    print 'Allocating time %s to general astrophysics'%(goal_GAdiff)
                    self.GAtime = self.GAtime + goal_GAdiff
                    TK.allocate_time(goal_GAdiff)

                # Append result values to self.DRM
                self.DRM.append(DRM)

                # Calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.to('day')

                # With prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration):
                    obsLength = (TK.obsEnd-TK.obsStart).to('day')
                    TK.next_observing_block(dt=obsLength)
                
                # With occulter, if spacecraft fuel is depleted, exit loop
                if Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %s' %TK.obsEnd.round(2)
                    break

        else:
            dtsim = (time.time()-t0)*u.s
            mission_end = "Mission complete: no more time available.\n"\
                    + "Simulation duration: %s.\n" %dtsim.astype('int')\
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."

            self.logger.info(mission_end)
            print mission_end

            return mission_end

    def next_target(self, old_sInd, old_occ_sInd, detmode, charmode):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star for the telescope
            old_occ_sInd (integer):
                Index of the previous target star for the occulter
            detmode (dict array):
                Selected observing mode for detection
            charmode (dict):
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
        
        # Create DRM
        DRM = {}

        # Allocate settling time + overhead time
        if old_sInd == old_occ_sInd and old_occ_sInd is not None:
            TK.allocate_time(Obs.settlingTime + charmode['syst']['ohTime'])
        else:
            TK.allocate_time(0.0 + detmode[0]['syst']['ohTime'])
        
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        assert OS.haveOcculter == True
        self.ao = Obs.thrust/Obs.scMass
        slewTime_fac = (2.*Obs.occulterSep/np.abs(self.ao)/(Obs.defburnPortion/2. \
                - Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        cnt = 0
        # Now, start to look for available targets
        while not TK.mission_is_over():
            # 0/ initialize arrays
            slewTime = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            intTimes = np.zeros(TL.nStars)*u.d
            occ_intTimes = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            occ_tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)

            # 1/ Find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            sd = None
            # find angle between old and new stars, default to pi/2 for first target
            if old_occ_sInd is None:
                sd = np.zeros(TL.nStars)*u.rad
                r_old = TL.starprop(np.where(np.in1d(TL.Name, self.occHIPs))[0][0], TK.currentTimeAbs)[0]
            else:
                # position vector of previous target star
                r_old = TL.starprop(old_occ_sInd, TK.currentTimeAbs)[0]
                u_old = r_old.value/np.linalg.norm(r_old)
                # position vector of new target stars
                r_new = TL.starprop(sInds, TK.currentTimeAbs)
                u_new = (r_new.value.T/np.linalg.norm(r_new,axis=1)).T
                # angle between old and new stars
                sd = np.arccos(np.clip(np.dot(u_old,u_new.T),-1,1))*u.rad
                # calculate slew time
                slewTime = np.sqrt(slewTime_fac*np.sin(sd/2.))
            
            occ_startTimes = TK.currentTimeAbs + slewTime
            occ_startTimesNorm = TK.currentTimeNorm + slewTime
            kogoodStart = Obs.keepout(TL, sInds, occ_startTimes, charmode)
            occ_sInds = sInds[np.where(kogoodStart)[0]]
            HIP_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
            occ_sInds = occ_sInds[np.where(np.in1d(occ_sInds, HIP_sInds))[0]]

            startTimes = TK.currentTimeAbs + np.zeros(TL.nStars)*u.d
            startTimesNorm = TK.currentTimeNorm
            kogoodStart = Obs.keepout(TL, sInds, startTimes, detmode[0])
            sInds = sInds[np.where(kogoodStart)[0]]

            # 2a/ If we are in detection phase two, start adding new targets to occulter target list
            if TK.currentTimeAbs > self.phase1_end:
                if self.is_phase1 is True:
                    print 'Entering detection phase 2: target list for occulter expanded'
                    self.is_phase1 = False
                occ_sInds = np.setdiff1d(occ_sInds, sInds[np.where((self.starVisits[sInds] > self.nVisitsMax) & 
                                                                   (self.occ_starVisits[sInds] == 0))[0]])

            fEZ = ZL.fEZ0
            WA = self.WAint[0]

            # 2/ calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            if len(occ_sInds) > 0:  
                occ_intTimes[occ_sInds] = self.calc_int_inflection(occ_sInds, fEZ, occ_startTimes, 
                                                                    WA, charmode, ischar=True)
                #occ_intTimes[occ_sInds] = self.calc_targ_intTime(occ_sInds, occ_startTimes[occ_sInds], charmode)
                totTimes = occ_intTimes*charmode['timeMultiplier']
                # end times
                occ_endTimes = occ_startTimes + totTimes
                occ_endTimesNorm = occ_startTimesNorm + totTimes
                # indices of observable stars
                occ_sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                            (occ_endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]

            if len(sInds) > 0:  
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], detmode[0])

                totTimes = intTimes*detmode[0]['timeMultiplier']
                # end times
                endTimes = startTimes + totTimes
                endTimesNorm = startTimesNorm + totTimes
                # indices of observable stars
                sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 3/ Find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(occ_sInds) > 0 and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, occ_sInds, occ_endTimes[occ_sInds], charmode)
                occ_sInds = occ_sInds[np.where(kogoodEnd)[0]]

            if len(sInds) > 0 and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, sInds, endTimes[sInds], detmode[0])
                sInds = sInds[np.where(kogoodEnd)[0]]

            # 4/ Filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            sInds = self.revisitFilter(sInds,TK.currentTimeNorm)

            # revisit list, with time after start
            if np.any(occ_sInds):
                occ_tovisit[occ_sInds] = (self.occ_starVisits[occ_sInds] == self.occ_starVisits[occ_sInds].min())
                if self.occ_starRevisit.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = TK.currentTimeNorm - self.occ_starRevisit[:,1]*u.day
                    ind_rev = [int(x) for x in self.occ_starRevisit[dt_rev > 0, 0] if x in occ_sInds]
                    occ_tovisit[ind_rev] = True
                occ_sInds = np.where(occ_tovisit)[0]

            # 5/ Filter off current occulter target star from detection list
            if old_occ_sInd is not None:
                sInds = sInds[np.where(sInds != old_occ_sInd)[0]]
                occ_sInds = occ_sInds[np.where(occ_sInds != old_occ_sInd)[0]]

            # 6/ Filter off previously visited occ_sInds
            #occ_sInds = occ_sInds[np.where(self.occ_starVisits[occ_sInds] == 0)[0]]

            #6a/ Filter off any stars visited by the occulter 3 or more times
            occ_sInds = occ_sInds[np.where(self.occ_starVisits[occ_sInds] < 3)[0]]

            # 7a/ Filter off stars with too-long inttimes
            if self.occ_arrives > TK.currentTimeAbs:
                available_time = self.occ_arrives - TK.currentTimeAbs
                if np.any(sInds[intTimes[sInds] < available_time]):
                    sInds = sInds[intTimes[sInds] < available_time]

            t_det = 0*u.d
            dmode = copy.deepcopy(detmode[0])

            # 7b/ Choose best target from remaining
            if np.any(sInds):

                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, intTimes[sInds])
                occ_sInd = old_occ_sInd

                # store relevant values
                # intTime_by_mode = np.zeros(len(detmode))*u.d
                # for m_i, mode in enumerate(detmode):
                #     intTime_by_mode[m_i] = self.calc_targ_intTime(sInd, startTimes[sInd], mode)
                # t_det = max(intTime_by_mode)

                # Perform dual band detections if necessary
                if self.WAint[sInd] > detmode[1]['IWA'] and self.WAint[sInd] < detmode[1]['OWA']:
                    dmode['BW'] = dmode['BW'] + detmode[1]['BW']
                    dmode['inst']['sread'] = dmode['inst']['sread'] + detmode[1]['inst']['sread']
                    dmode['inst']['idark'] = dmode['inst']['idark'] + detmode[1]['inst']['idark']
                    dmode['inst']['CIC'] = dmode['inst']['CIC'] + detmode[1]['inst']['CIC']
                    dmode['syst']['optics'] = np.mean((dmode['syst']['optics'], detmode[1]['syst']['optics']))
                    dmode['instName'] = 'combined'

                t_det = self.calc_targ_intTime(sInd, startTimes[sInd], dmode)[0]

                # update visited list for current star
                self.starVisits[sInd] += 1

            # if the starshade has arrived at its destination, or it is the first observation
            if np.any(occ_sInds):
                if old_occ_sInd is None or ((TK.currentTimeAbs + t_det) >= self.occ_arrives and self.ready_to_update):
                    occ_sInd = self.choose_next_occulter_target(old_occ_sInd, occ_sInds, intTimes)
                    if old_occ_sInd is None:
                        self.occ_arrives = TK.currentTimeAbs
                    else:
                        self.occ_arrives = occ_startTimes[occ_sInd]
                        self.occ_slewTime = slewTime[occ_sInd]
                        self.occ_sd = sd[occ_sInd]
                    if not np.any(sInds):
                        sInd = occ_sInd
                    self.ready_to_update = False
                    self.occ_starVisits[occ_sInd] += 1
                elif not np.any(sInds):
                    TK.allocate_time(1*u.d)
                    cnt += 1
                    continue

            # if no observable target, call the TimeKeeping.wait() method
            if not np.any(sInds) and not np.any(occ_sInds):
                TK.allocate_time(TK.waitTime*TK.waitMultiple**cnt)
                cnt += 1
                continue

            break

        else:
            self.logger.info('Mission complete: no more time available')
            print 'Mission complete: no more time available'
            return DRM, None, None, None, None, None, None

        if TK.mission_is_over():
            self.logger.info('Mission complete: no more time available')
            print 'Mission complete: no more time available'
            return DRM, None, None, None, None, None, None

        return DRM, sInd, occ_sInd, t_det, sd, occ_sInds, dmode

