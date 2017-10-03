from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
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
from EXOSIMS.util.deltaMag import deltaMag

class tieredScheduler(SurveySimulation):
    """tieredScheduler 
    
    This class implements a tiered scheduler that independantly schedules the observatory
    while the starshade slews to its next target.
    
        Args:
        as (iterable 2x1):
            Cost function coefficients: slew distance, completeness
        as (iterable nx1)
            List of star HIP numbers to initialize occulter target list.
        
        \*\*specs:
            user specified values
    """

    def __init__(self, coeffs=[2,1], occHIPs=[], **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 2x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 2):
            raise TypeError("coeffs must be a 3 element iterable")

        #Add to outspec
        self._outspec['coeffs'] = coeffs
        self._outspec['occHIPs'] = occHIPs
        
        #normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs)
        
        self.coeffs = coeffs
        if occHIPs is not []:
            occHIPs_path = os.path.join(EXOSIMS.__path__[0],'Scripts',occHIPs)
            assert os.path.isfile(occHIPs_path), "%s is not a file."%occHIPs_path
            HIPsfile = open(occHIPs_path, 'r').read()
            self.occHIPs = HIPsfile.split(',')
        else:
            self.occHIPs = occHIPs

        TL = self.TargetList
        self.occ_arrives = None # The timestamp at which the occulter finishes slewing
        self.occ_starVisits = np.zeros(TL.nStars,dtype=int) # The number of times each star was visited by the occulter
        self.phase1_end = None # The designated end time for the first observing phase
        self.is_phase1 = True
        self.FA_status = np.zeros(TL.nStars,dtype=bool)
        self.GA_percentage = .25
        self.GAtime = 0.*u.d
        self.goal_GAtime = None
        self.EVPOC = None
        self.ao = None

        self.ready_to_update = False


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

        self.phase1_end = TK.missionStart + 365*u.d
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
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
            TK.obsStart = TK.currentTimeNorm.to('day')
            prev_occ_sInd = occ_sInd
            DRM, sInd, occ_sInd, t_det, slewTime, sd, occ_sInds = self.next_target(sInd, occ_sInd, detMode, charMode)
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
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, detMode)
                    if np.any(detected):
                        print '  Det. results are: %s'%(detected)
                    # update GAtime
                    self.GAtime = self.GAtime + t_det.to('day')*.07
                    # populate the DRM with detection results
                    DRM['det_time'] = t_det.to('day')
                    DRM['det_status'] = detected
                    DRM['det_SNR'] = det_SNR
                    DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                    DRM['det_params'] = det_systemParams
                    DRM['FA_det_status'] = int(FA)
                
                elif sInd == occ_sInd:
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    # First store fEZ, dMag, WA, and characterization mode
                    occ_pInds = np.where(SU.plan2star == occ_sInd)[0]
                    sInd = occ_sInd

                    # update GAtime
                    if time2arrive > 0*u.d:
                        self.GAtime = self.GAtime + time2arrive.to('day')

                    DRM['slew_time'] = slewTime[sInd].to('day').value
                    DRM['slew_angle'] = sd[sInd].to('deg').value
                    slew_mass_used = slewTime[sInd]*Obs.defburnPortion*Obs.flowRate
                    DRM['slew_dV'] = (slewTime[sInd]*self.ao*Obs.defburnPortion).to('m/s').value
                    DRM['slew_mass_used'] = slew_mass_used.to('kg').value
                    Obs.scMass = Obs.scMass - slew_mass_used
                    DRM['scMass'] = Obs.scMass.to('kg').value

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
                        t_rev = TK.currentTimeNorm + T/2.
                        revisit = np.array([sInd, t_rev.to('day').value])
                        if self.starRevisit.size == 0:
                            self.starRevisit = np.array([revisit])
                        else:
                            self.starRevisit = np.vstack((self.starRevisit, revisit))

                self.goal_GAtime = self.GA_percentage * TK.currentTimeNorm.to('day')
                goal_GAdiff = self.goal_GAtime - self.GAtime

                # allocate extra time to GA if we are falling behind
                if goal_GAdiff > 1*u.d and goal_GAdiff < time2arrive:
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
            detmode (dict):
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
            TK.allocate_time(0.0 + detmode['syst']['ohTime'])
        
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        assert OS.haveOcculter == True
        self.ao = Obs.thrust/Obs.scMass
        slewTime_fac = (2.*Obs.occulterSep/np.abs(self.ao)/(Obs.defburnPortion/2. \
                - Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        # Now, start to look for available targets
        while not TK.mission_is_over():
            # 0/ initialize arrays
            slewTime = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            t_dets = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)
            
            # 1/ Find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            sd = None
            # find angle between old and new stars, default to pi/2 for first target
            if old_occ_sInd is None:
                sd = np.zeros(TL.nStars)*u.rad
                r_old = TL.starprop(np.where(TL.Name == self.occHIPs[0])[0][0], TK.currentTimeAbs)[0]
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
            
            occ_startTime = TK.currentTimeAbs + slewTime
            kogoodStart = Obs.keepout(TL, sInds, occ_startTime, charmode)
            occ_sInds = sInds[np.where(kogoodStart)[0]]
            HIP_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
            occ_sInds = occ_sInds[np.where(np.in1d(occ_sInds, HIP_sInds))[0]]

            startTime = TK.currentTimeAbs + np.zeros(TL.nStars)*u.d
            kogoodStart = Obs.keepout(TL, sInds, startTime, detmode)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 2/ Calculate integration times for the preselected targets, 
            # and filter out t_tots > integration cutoff
            fEZ = ZL.fEZ0
            dMag = self.dMagint[occ_sInds]
            WAc = OS.starlightSuppressionSystems[1]['IWA']*2
            if np.any(occ_sInds):
                fZ = ZL.fZ(Obs, TL, occ_sInds, occ_startTime[occ_sInds], charmode)
                t_dets[occ_sInds] = OS.calc_intTime(TL, occ_sInds, fZ, fEZ, dMag, WAc, charmode)
                # include integration time multiplier
                occ_t_tots = t_dets*charmode['timeMultiplier']
                # total time must be positive, shorter than integration cut-off,
                # and it must not exceed the Observing Block end time 
                startTimeNorm = (occ_startTime - TK.missionStart).jd*u.day
                occ_sInds = np.where((occ_t_tots > 0) & (occ_t_tots <= OS.intCutoff) & \
                            (startTimeNorm + occ_t_tots <= TK.OBendTimes[TK.OBnumber]))[0]

            dMag = self.dMagint[sInds]
            WA = self.WAint[sInds]
            if np.any(sInds):
                fZ = ZL.fZ(Obs, TL, sInds, startTime[sInds], detmode)
                t_dets[sInds] = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, detmode)
                # include integration time multiplier
                t_tots = t_dets*detmode['timeMultiplier']
                # total time must be positive, shorter than integration cut-off,
                # and it must not exceed the Observing Block end time 
                startTimeNorm = (startTime - TK.missionStart).jd*u.day
                sInds = np.where((t_tots > 0) & (t_tots <= OS.intCutoff) & \
                            (startTimeNorm + t_tots <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 3/ Find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if np.any(occ_sInds):
                endTime = occ_startTime[occ_sInds] + occ_t_tots[occ_sInds]
                kogoodEnd = Obs.keepout(TL, occ_sInds, endTime, charmode)
                occ_sInds = occ_sInds[np.where(kogoodEnd)[0]]

            if np.any(sInds):
                endTime = startTime[sInds] + t_tots[sInds]
                kogoodEnd = Obs.keepout(TL, sInds, endTime, detmode)
                sInds = sInds[np.where(kogoodEnd)[0]]

            # 3a/ If we are in detection phase two, start adding new targets to occulter target list
            if TK.currentTimeAbs > self.phase1_end:
                if self.is_phase1 is True:
                    print 'Entering detection phase 2: target list for occulter expanded'
                    self.is_phase1 = False
                occ_sInds = np.setdiff1d(occ_sInds, sInds[np.where((self.starVisits[sInds] > 5) & 
                                                                   (self.occ_starVisits[sInds] == 0))[0]])
            
            # 4/ Filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if np.any(sInds):
                tovisit[sInds] = (self.starVisits[sInds] == self.starVisits[sInds].min())
                if self.starRevisit.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]
                    tovisit[ind_rev] = True
                sInds = np.where(tovisit)[0]

            # 5/ Filter off current occulter target star from detection list
            if old_occ_sInd is not None:
                sInds = sInds[np.where(sInds != old_occ_sInd)[0]]

            # 6/ Filter off previously visited occ_sInds
            occ_sInds = occ_sInds[np.where(self.occ_starVisits[occ_sInds] == 0)[0]]

            # 7/ Choose best target from remaining
            if np.any(sInds):
                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, slewTime, t_dets[sInds])
                occ_sInd = old_occ_sInd
                # if it is the first target or if there is not enough time to make another detection
                # store relevant values
                t_det = t_dets[sInd]
                fZ = ZL.fZ(Obs, TL, sInd, startTime[sInd], detmode)
                WA = self.WAint[sInd]
                # update visited list for current star
                self.starVisits[sInd] += 1
                int_time = self.calc_int_inflection(sInd, fEZ, fZ, WA, detmode)

                if int_time < t_det:
                    t_det = int_time

                # if the starshade has arrived at its destination, or it is the first observation
                if np.any(occ_sInds) or old_occ_sInd is None:
                    if old_occ_sInd is None or ((TK.currentTimeAbs + t_det) >= self.occ_arrives and self.ready_to_update): 
                        occ_sInd = self.choose_next_occulter_target(old_occ_sInd, occ_sInds, t_dets)
                        self.occ_arrives = occ_startTime[occ_sInd]
                        self.ready_to_update = False
                        self.occ_starVisits[occ_sInd] += 1
                break

            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.wait()

        else:
            self.logger.info('Mission complete: no more time available')
            print 'Mission complete: no more time available'
            return DRM, None, None, None, None, None, None

        if TK.mission_is_over():
            self.logger.info('Mission complete: no more time available')
            print 'Mission complete: no more time available'
            return DRM, None, None, None, None, None, None

        return DRM, sInd, occ_sInd, t_dets[sInd], slewTime, sd, occ_sInds

    def choose_next_occulter_target(self, old_occ_sInd, occ_sInds, t_dets):
        """Choose next target for the occulter based on truncated 
        depth first search of linear cost function.
        
        Args:
            old_occ_sInd (integer):
                Index of the previous target star
            occ_sInds (integer array):
                Indices of available targets
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """

        # Choose next Occulter target

        OS = self.OpticalSystem
        Obs = self.Observatory
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        nStars = len(occ_sInds)

        # reshape sInds
        occ_sInds = np.array(occ_sInds,ndmin=1)

        # current stars have to be in the adjmat
        if (old_occ_sInd is not None) and (old_occ_sInd not in occ_sInds):
            occ_sInds = np.append(occ_sInds, old_occ_sInd)

        # get completeness values
        comps = Comp.completeness_update(TL, occ_sInds, self.starVisits[occ_sInds], TK.currentTimeNorm)
        
        # if first target, or if only 1 available target, choose highest available completeness
        nStars = len(occ_sInds)
        if (old_occ_sInd is None) or (nStars == 1):
            occ_sInd = np.where(TL.Name == self.occHIPs[0])[0][0]
            #occ_sInd = np.random.choice(occ_sInds[comps == max(comps)])
            return occ_sInd
        
        # define adjacency matrix
        A = np.zeros((nStars,nStars))

        # consider slew distance when there's an occulter
        r_ts = TL.starprop(occ_sInds, TK.currentTimeAbs)
        u_ts = (r_ts.value.T/np.linalg.norm(r_ts,axis=1)).T
        angdists = np.arccos(np.clip(np.dot(u_ts,u_ts.T),-1,1))
        A[np.ones((nStars),dtype=bool)] = angdists
        A = self.coeffs[0]*(A)/np.pi

        # add factor due to completeness
        A = A + self.coeffs[1]*(1-comps)

        # kill diagonal
        A = A + np.diag(np.ones(nStars)*np.Inf)
        
        # take two traversal steps
        step1 = np.tile(A[occ_sInds==old_occ_sInd,:],(nStars,1)).flatten('F')
        step2 = A[np.array(np.ones((nStars,nStars)),dtype=bool)]
        tmp = np.argmin(step1+step2)
        occ_sInd = occ_sInds[int(np.floor(tmp/float(nStars)))]

        return occ_sInd

    def choose_next_telescope_target(self, old_sInd, sInds, slewTime, t_dets):
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

    def calc_int_inflection(self, sInd, fEZ, fZ, WA, mode, ischar=False):
        """Calculate integration time based on inflection point of Completeness as a function of int_time
        
        Args:
            sInd (integer):
                Index of the target star
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode

        Returns:
            int_time (float):
                The suggested integration time
        
        """

        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        ZL = self.ZodiacalLight
        Obs = self.Observatory

        Cpath = os.path.join(Comp.classpath, Comp.filename+'_curves.comp')
        intTimes = np.logspace(-5, 2, num_points)*u.d

        if os.path.exists(Cpath):
            print 'Loading cached completeness file from "%s".' % Cpath
            curves = pickle.load(open(Cpath, 'rb'))
            print 'Completeness curves loaded from cache.'
        else:
            sInds = np.arange(TL.nStars)
            num_points = 500
            fZ = np.ones(intTimes.shape) * fZ
            fEZ = np.ones(intTimes.shape) * ZL.fEZ0
            WA = np.ones(intTimes.shape) * WA

            curves = np.zeros([2, sInds.shape, intTime.shape])

            # calculate completeness curves for all sInds
            print 'Cached completeness file not found at "%s".' % Cpath
            print 'Beginning completeness curve calculations.'

            for i_sInd in sInds:
                sInd_array = np.ones(intTimes.shape) * i_sInd
                curves[0,i_sInd,:] = OS.calc_dMag_per_intTime(intTimes, TL, sInd_array, fZ, fEZ, WA, mode)[:,0]
                curves[1,i_sInd,:] = Comp.comp_per_intTime(intTimes, TL, i_sInd, fZ, fEZ, WA[0], mode)

            pickle.dump(curves, open(Cpath, 'wb'))
            print 'completeness curves stored in %r' % Cpath

        dm_v_t = curves[0,sInd,:]
        c_v_t = curves[1,sInd,:]
        dcdt = np.diff(c_v_t)/np.diff(intTimes)

        #=========================================================

        dMagmin = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[1]*\
                  Comp.PlanetPopulation.Rprange[1]/Comp.PlanetPopulation.rrange[0])**2))
        dMagmax = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[0]*\
                  Comp.PlanetPopulation.Rprange[0]/Comp.PlanetPopulation.rrange[1])**2))
        # dMagmax = OS.dMagLim
        num_points = 250

        dMags = np.linspace(dMagmin, dMagmax, num_points)

        # calculate t_det as a function of dMag
        t_dets = OS.calc_intTime(TL, sInd, fZ, fEZ, dMags, WA, mode)

        # calculate comp as a function of dMag
        smin = TL.dist[sInd] * np.tan(mode['IWA'])
        if np.isinf(mode['OWA']):
            smax = Comp.PlanetPopulation.rrange[1].to('AU').value
        else:
            smax = (np.tan(OWA)*TL.dist[sInds]).to('AU').value

        if self.EVPOC is None:
            self.calc_EVPOC()

        comps = self.EVPOC(smin.to('AU').value, smax.to('AU').value, dMagmin, dMags)

        # find the inflection point of the completeness graph
        if ischar is False:
            int_time = t_dets[np.where(np.gradient(comps) == max(np.gradient(comps)))[0]][0]
            int_time = int_time*self.starVisits[sInd]

            # update star completeness
            idx = (np.abs(t_dets-int_time)).argmin()
            comp = comps[idx]
            TL.comp[sInd] = comp
        else:
            idt = np.abs(t_dets - max(t_dets)).argmin()
            idx = np.abs(comps - comps[idt]*.9).argmin()

            # idx = np.abs(comps - max(comps)*.9).argmin()
            int_time = t_dets[idx]
            comp = comps[idx]

        return int_time

    def calc_EVPOC(self):
        Comp = self.Completeness

        bins = 1000
        # xedges is array of separation values for interpolant
        xedges = np.linspace(0., Comp.PlanetPopulation.rrange[1].value, bins)*\
                Comp.PlanetPopulation.arange.unit
        xedges = xedges.to('AU').value

        # yedges is array of delta magnitude values for interpolant
        ymin = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[1]*\
                Comp.PlanetPopulation.Rprange[1]/Comp.PlanetPopulation.rrange[0])**2))
        ymax = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[0]*\
                Comp.PlanetPopulation.Rprange[0]/Comp.PlanetPopulation.rrange[1])**2*1e-11))
        yedges = np.linspace(ymin, ymax, bins)

        # number of planets for each Monte Carlo simulation
        nplan = int(np.min([1e6,Comp.Nplanets]))
        # number of simulations to perform (must be integer)
        steps = int(Comp.Nplanets/nplan)
        
        Cpath = os.path.join(Comp.classpath, Comp.filename+'.comp')
        H, xedges, yedges = self.genC(Cpath, nplan, xedges, yedges, steps)
        EVPOCpdf = interpolate.RectBivariateSpline(xedges, yedges, H.T)
        EVPOC = np.vectorize(EVPOCpdf.integral)

        self.EVPOC = EVPOC

    def genC(self, Cpath, nplan, xedges, yedges, steps):
        """Gets completeness interpolant for initial completeness
        
        This function either loads a completeness .comp file based on specified
        Planet Population module or performs Monte Carlo simulations to get
        the 2D completeness values needed for interpolation.
        
        Args:
            Cpath (string):
                path to 2D completeness value array
            nplan (float):
                number of planets used in each simulation
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
            steps (integer):
                number of simulations to perform
                
        Returns:
            H (float ndarray):
                2D numpy ndarray containing completeness probability density values
        
        """
        
        # if the 2D completeness pdf array exists as a .comp file load it
        if os.path.exists(Cpath):
            print 'Loading cached completeness file from "%s".' % Cpath
            H = pickle.load(open(Cpath, 'rb'))
            print 'Completeness loaded from cache.'
        else:
            # run Monte Carlo simulation and pickle the resulting array
            print 'Cached completeness file not found at "%s".' % Cpath
            print 'Beginning Monte Carlo completeness calculations.'
            
            t0, t1 = None, None # keep track of per-iteration time
            for i in xrange(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = '' # no message
                else:
                    delta_t_msg = '[%.3f s/iteration]' % (t1 - t0)
                print 'Completeness iteration: %5d / %5d %s' % (i+1, steps, delta_t_msg)
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
            
            H = H/(self.Nplanets*(xedges[1]-xedges[0])*(yedges[1]-yedges[0]))
                        
            # store 2D completeness pdf array as .comp file
            pickle.dump(H, open(Cpath, 'wb'))
            print 'Monte Carlo completeness calculations finished'
            print '2D completeness array stored in %r' % Cpath
        
        return H, xedges, yedges

    def hist(self, nplan, xedges, yedges):
        """Returns completeness histogram for Monte Carlo simulation
        
        This function uses the inherited Planet Population module.
        
        Args:
            nplan (float):
                number of planets used
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
        
        Returns:
            h (ndarray):
                2D numpy ndarray containing completeness histogram
        
        """
        
        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=1000,
                range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
        return h, xedges, yedges

    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation
        
        Args:
            nplan (integer):
                Number of planets
                
        Returns:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness
        
        """
        
        PPop = self.PlanetPopulation
        
        nplan = int(nplan)
        
        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.0*np.pi,size=nplan)
        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(nplan)
        # check if circular orbits
        if np.sum(PPop.erange) == 0:
            r = a
            e = 0.0
            E = M
        else:
            E = eccanom(M,e)
            # orbital radius
            r = a*(1.0-e*np.cos(E))

        beta = np.arccos(1.0-2.0*np.random.uniform(size=nplan))*u.rad
        s = r*np.sin(beta)
        # phase function
        Phi = self.PlanetPhysicalModel.calc_Phi(beta)
        # calculate dMag
        dMag = deltaMag(p,Rp,r,Phi)
        
        return s, dMag

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
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        # get the last detected planets, and check if there was a FA
        #det = self.lastDetected[sInd,0]
        det = np.ones(pInds.size, dtype=bool)
        fEZs = SU.fEZ[pInds].to('1/arcsec2').value
        dMags = SU.dMag[pInds]
        WAs = SU.WA[pInds].to('arcsec').value

        FA = (det.size == pInds.size + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
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
        
        # look for last detected planets that have not been fully characterized
        tochar = np.zeros(len(det), dtype=bool)
        if (FA == False):
            tochar[det] = (self.fullSpectra[pInds[det]] != 1)
        elif pInds[det].size > 1:
            tochar[det] = np.append((self.fullSpectra[pInds[det][:-1]] != 1), True)
        else:
            tochar[det] = np.array([True])
        
        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs
            startTimeNorm = TK.currentTimeNorm
            # planets to characterize
            tochar[tochar] = Obs.keepout(TL, sInd, startTime, mode)

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            # fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            # dMag = self.lastDetected[sInd,2][tochar]
            # WA = self.lastDetected[sInd,3][tochar]*u.mas
            fEZ = fEZs[tochar]/u.arcsec**2
            dMag = dMags[tochar]
            WAp = WAs[tochar]*u.arcsec

            intTimes = np.zeros(len(pInds))*u.d
            # t_chars[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            for i,j in enumerate(WAp):
                if tochar[i]:
                    intTimes[i] = self.calc_int_inflection(sInd, fEZ[i], fZ, j, mode, ischar=True)

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
            tochar[tochar] = Obs.keepout(TL, sInd, endTimes[tochar], mode)
        
        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            intTime = np.max(intTimes[tochar])
            pIndsChar = pIndsDet[tochar]
            log_char = '   - Charact. planet(s) %s (%s/%s detected)'%(pIndsChar, 
                    len(pIndsChar), len(pIndsDet))
            self.logger.info(log_char)
            print log_char
            
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
                dt = intTime/self.ntFlux
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    TK.allocate_time(dt/2.)
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                            fZ=fZs[i])
                    # allocate second half of dt
                    TK.allocate_time(dt/2.)
                
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
                extraTime = intTime*(mode['timeMultiplier'] - 1)
                TK.allocate_time(extraTime)
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                TK.allocate_time(totTime/2.)
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime/2.)
            
            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = fEZs[-1]/u.arcsec**2
                dMag = dMags[-1]
                WA = WAs[-1]*u.arcsec
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
            WAchar = WAs[char]*u.arcsec
            # find the current WAs of characterized planets
            WA = WAs*u.arcsec
            if FA:
                WAs = np.append(WAs, WAs[-1]*u.arcsec)
            # check for partial spectra
            IWA_max = mode['IWA']*(1 + mode['BW']/2.)
            OWA_min = mode['OWA']*(1 - mode['BW']/2.)
            char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        return characterized.astype(int), fZ, systemParams, SNR, intTime

