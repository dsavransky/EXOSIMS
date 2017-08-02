from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import EXOSIMS, os
import astropy.units as u
import astropy.constants as const
import numpy as np
import time

class SS_det_only(SurveySimulation):
    """
    SS_det_only is a variant of survey scheduler that performs only detections
    """

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
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber + 1)
        self.logger.info(log_begin)
        print log_begin
        t0 = time.time()
        sInd = None
        cnt = 0
        while not TK.mission_is_over():
            
            # save the start time of this observation (BEFORE any OH/settling/slew time)
            TK.obsStart = TK.currentTimeNorm.to('day')
            
            # acquire the NEXT TARGET star index and create DRM
            DRM, sInd, det_intTime = self.next_target(sInd, det_mode)
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
                log_obs = ('  Observation #%s, target #%s/%s with %s planet(s), ' \
                        + 'mission time: %s')%(cnt, sInd+1, TL.nStars, len(pInds), 
                        TK.obsStart.round(2))
                self.logger.info(log_obs)
                print log_obs
                
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
                
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                # if char_mode['SNR'] not in [0, np.inf]:
                #     characterized = np.zeros(det.size, dtype=int).tolist()
                #     # characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                #     #         self.observation_characterization(sInd, char_mode)
                # else:
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
                # DRM['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                # DRM['char_status'] = characterized[:-1] if FA else characterized
                # DRM['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                # DRM['char_fZ'] = char_fZ.to('1/arcsec2')
                # DRM['char_params'] = char_systemParams
                # # populate the DRM with FA results
                # DRM['FA_det_status'] = int(FA)
                # DRM['FA_char_status'] = characterized[-1] if FA else 0
                # DRM['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                # DRM['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 \
                #         if FA else 0./u.arcsec**2
                # DRM['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                # DRM['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec \
                #         if FA else 0.*u.arcsec
                
                # populate the DRM with observation modes
                DRM['det_mode'] = dict(det_mode)
                del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                # DRM['char_mode'] = dict(char_mode)
                # del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                
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
                    print 'Total fuel mass exceeded at %s'%TK.obsEnd.round(2)
                    break
        
        else:
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            print log_end


    def choose_next_target(self, old_sInd, sInds, slewTime, t_dets):
        """Choose next telescope target based on star completeness and integration time.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTime (float array):
                slew times to all stars (must be indexed by sInds)
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping

        nStars = len(sInds)

        # reshape sInds
        sInds = np.array(sInds,ndmin=1)

        # 1/ Choose next telescope target
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], TK.currentTimeNorm)

        # add weight for star revisits
        ind_rev = []
        if self.starRevisit.size != 0:
            dt_max = 1.*u.week
            dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
            ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]

        f2_uv = np.where((self.starVisits[sInds] > 0) & (self.starVisits[sInds] < 6), 
                          self.starVisits[sInds], 0) * (1 - (np.in1d(sInds, ind_rev, invert=True)))

        weights = (comps + f2_uv/6.)/t_dets
        sInd = np.random.choice(sInds[weights == max(weights)])

        return sInd
