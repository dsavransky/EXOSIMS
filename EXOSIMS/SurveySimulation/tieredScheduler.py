from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools
from scipy import interpolate
try:
    import cPickle as pickle
except:
    import pickle

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

    def __init__(self, coeffs=[1,1], occ_HIPs=[], **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 6x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 4):
            raise TypeError("coeffs must be a 3 element iterable")
        
        #normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs)
        
        self.coeffs = coeffs
        self.occ_HIPs = occ_HIPs
        self.occ_arrives = None # The timestamp at which the occulter finishes slewing
        self.occ_starVisits = np.zeros(TL.nStars,dtype=int) # The number of times each star was visited by the occulter
        self.phase1_end = None # The designated end time for the first observing phase
        self.FA_status = np.zeros(TL.nStars,dtype=bool)

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

        self.phase1_end = TK.currentTimeNorm.to('day') + 365*u.day
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter == True:
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
        Logger.info('OB%s: survey beginning.'%(TK.OBnumber+1))
        print 'OB%s: survey beginning.'%(TK.OBnumber+1)
        t0 = time.time()
        sInd = None
        occ_sInd = None
        cnt = 0
        while not TK.mission_is_over():
             
            # Acquire the NEXT TARGET star index and create DRM
            TK.obsStart = TK.currentTimeNorm.to('day')
            DRM, sInd, occ_sInd, t_det = self.next_target(sInd, occ_sInd, detMode)
            assert t_det !=0, "Integration time can't be 0."
            
            if sInd is not None:
                cnt += 1
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
                occ_pInds = np.where(SU.plan2star == occ_sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()
                Logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
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
                    detected, detSNR, FA = self.observation_detection(sInd, t_det, detMode)
                    # Update the occulter wet mass
                    # if OS.haveOcculter == True:
                    #     DRM = self.update_occulter_mass(DRM, sInd, t_det, 'det')
                    # Populate the DRM with detection results
                    self.FA_status[sInd] = FA
                    DRM['det_time'] = t_det.to('day').value
                    DRM['det_status'] = detected
                    DRM['det_SNR'] = detSNR
                
                elif sInd == occ_sInd:
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    # First store fEZ, dMag, WA, and characterization mode
                    if np.any(occ_pInds):
                        DRM['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                        DRM['char_dMag'] = SU.dMag[occ_pInds].tolist()
                        DRM['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                    DRM['char_mode'] = dict(charMode)
                    del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                    characterized, charSNR, t_char = self.observation_characterization(sInd, charMode)
                    assert t_char !=0, "Integration time can't be 0."
                    # Update the occulter wet mass
                    if OS.haveOcculter == True and t_char is not None:
                        DRM = self.update_occulter_mass(DRM, sInd, t_char, 'char')
                    # if any false alarm, store its characterization status, fEZ, dMag, and WA
                    if self.FA_status[sInd]:
                        DRM['FA_status'] = characterized.pop()
                        DRM['FA_SNR'] = charSNR.pop()
                        DRM['FA_fEZ'] = self.lastDetected[sInd,1][-1]
                        DRM['FA_dMag'] = self.lastDetected[sInd,2][-1]
                        DRM['FA_WA'] = self.lastDetected[sInd,3][-1]
                    # add star back into the revisit list
                    if 1 in characterized or -1 in characterized:
                        sp = SU.s.mean() # XXX use smin here
                        Mp = SU.Mp.mean()
                        mu = const.G*(Mp + Ms)
                        T = 2.*np.pi*np.sqrt(sp**3/mu)
                        t_rev = TK.currentTimeNorm + 0.75*T
                        revisit = np.array([sInd, t_rev.to('day').value])
                        if self.starRevisit.size == 0:
                            self.starRevisit = np.array([revisit])
                        else:
                            self.starRevisit = np.vstack((self.starRevisit, revisit))
                    # Populate the DRM with characterization results
                    DRM['char_time'] = t_char.to('day').value if t_char else 0.
                    DRM['char_status'] = characterized
                    DRM['char_SNR'] = charSNR

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
                if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %s' %TK.obsEnd.round(2)
                    break

        else:
            dtsim = (time.time()-t0)*u.s
            mission_end = "Mission complete: no more time available.\n"\
                    + "Simulation duration: %s.\n" %dtsim.astype('int')\
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."

            Logger.info(mission_end)
            print mission_end

            return mission_end

    def next_target(self, old_sInd, old_occ_sInd, mode):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star for the telescope
            old_occ_sInd (integer):
                Index of the previous target star for the occulter
            mode (dict):
                Selected observing mode for detection
                
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
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        if OS.haveOcculter == True:
            ao = Obs.thrust/Obs.scMass
            slewTime_fac = (2.*Obs.occulterSep/np.abs(ao)/(Obs.defburnPortion/2. \
                    - Obs.defburnPortion**2/4.)).decompose().to('d2')

        # initialize occulter target list
        occ_sInds = np.where(np.in1d(TL.Name, self.occ_HIPs))
        
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
            if OS.haveOcculter == True:
                # find angle between old and new stars, default to pi/2 for first target
                if old_occ_sInd is None:
                    sd = np.zeros(TL.nStars)*u.rad
                else:
                    # position vector of previous target star
                    r_old = TL.starprop(old_occ_sInd, TK.currentTimeAbs)[0]
                    u_old = r_old.value/np.linalg.norm(r_old)
                    # position vector of new target stars
                    r_new = TL.starprop(occ_sInds, TK.currentTimeAbs)
                    u_new = (r_new.value.T/np.linalg.norm(r_new,axis=1)).T
                    # angle between old and new stars
                    sd = np.arccos(np.clip(np.dot(u_old,u_new.T),-1,1))*u.rad
                # calculate slew time
                slewTime = np.sqrt(slewTime_fac*np.sin(sd/2.))
            
            occ_startTime = TK.currentTimeAbs + slewTime
            kogoodStart = Obs.keepout(TL, occ_sInds, occ_startTime, OS.telescopeKeepout)
            occ_sInds = occ_sInds[np.where(kogoodStart)[0]]

            startTime = TK.currentTimeAbs + np.zeros(TL.nStars)*u.d
            kogoodStart = Obs.keepout(TL, sInds, startTime, OS.telescopeKeepout)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 2/ Calculate integration times for the preselected targets, 
            # and filter out t_tots > integration cutoff
            fEZ = ZL.fEZ0
            dMag = OS.dMagint
            WA = OS.WAint
            if np.any(occ_sInds):
                fZ = ZL.fZ(TL, occ_sInds, mode['lam'], Obs.orbit(occ_startTime[occ_sInds]))
                t_dets[occ_sInds] = OS.calc_intTime(TL, occ_sInds, fZ, fEZ, dMag, WA, mode)
                # include integration time multiplier
                t_tots = t_dets*mode['timeMultiplier']
                # total time must be positive, shorter than integration cut-off,
                # and it must not exceed the Observing Block end time 
                startTimeNorm = (occ_startTime - TK.missionStart).jd*u.day
                occ_sInds = np.where((t_tots > 0) & (t_tots <= OS.intCutoff) & \
                            (startTimeNorm + t_tots <= TK.OBendTimes[TK.OBnumber]))[0]

            if np.any(sInds):
                fZ = ZL.fZ(TL, sInds, mode['lam'], Obs.orbit(startTime[sInds]))
                t_dets[sInds] = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
                # include integration time multiplier
                t_tots = t_dets*mode['timeMultiplier']
                # total time must be positive, shorter than integration cut-off,
                # and it must not exceed the Observing Block end time 
                startTimeNorm = (startTime - TK.missionStart).jd*u.day
                sInds = np.where((t_tots > 0) & (t_tots <= OS.intCutoff) & \
                            (startTimeNorm + t_tots <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 3/ Find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if np.any(occ_sInds):
                endTime = occ_startTime[occ_sInds] + t_tots[occ_sInds]
                kogoodEnd = Obs.keepout(TL, occ_sInds, endTime, OS.telescopeKeepout)
                occ_sInds = occ_sInds[np.where(kogoodEnd)[0]]

            if np.any(sInds):
                endTime = startTime[sInds] + t_tots[sInds]
                kogoodEnd = Obs.keepout(TL, sInds, endTime, OS.telescopeKeepout)
                sInds = sInds[np.where(kogoodEnd)[0]]

            # If we are in detection phase two, start adding new targets to occulter target list
            if TK.currentTimeAbs > self.phase1_end:
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
            sInds = sInds[np.where(sInds != old_occ_sInd)[0]]

            # 6/ Filter off previously visited occ_sInds
            occ_sInds = occ_sInds[np.where(self.occ_starVisits[occ_sInds] == 0)[0]]

            # 7/ Choose best target from remaining
            if np.any(sInds):
                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, slewTime, t_dets[sInds])
                occ_sInd = old_occ_sInd
                # if it is the first target or if there is not enough time to make another detection
                if old_occ_sInd is None or (TK.currentTimeAbs + t_dets[sInd]) >= self.occ_arrives: 
                    occ_sInd = self.choose_next_occulter_target(old_occ_sInd, occ_sInds, 
                                                                slewTime, t_dets[occ_sInds])
                    sInd = occ_sInd
                    self.occ_arrives = occ_startTime[occ_sInd]
                    self.occ_starVisits[occ_sInd] += 1

                # update visited list for current star
                self.starVisits[sInd] += 1
                # update visited list for Completeness for current star
                Comp.visits[sInd] += 1
                # store relevant values
                t_det = t_dets[sInd]
                fZ = ZL.fZ(TL, sInds, mode['lam'], Obs.orbit(startTime[sInds]))
                int_time = calc_int_inflection(sInd, fEZ, WA, mode)
                if int_time < t_det:
                    t_det = int_time
                break

            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.wait()

        else:
            Logger.info('Mission complete: no more time available')
            print 'Mission complete: no more time available'
            return DRM, None, None
        
        if OS.haveOcculter == True:
            # find values related to slew time
            DRM['slew_time'] = slewTime[occ_sInd].to('day').value
            DRM['slew_angle'] = sd[occ_sInd].to('deg').value
            slew_mass_used = slewTime[occ_sInd]*Obs.defburnPortion*Obs.flowRate
            DRM['slew_dV'] = (slewTime[occ_sInd]*ao*Obs.defburnPortion).to('m/s').value
            DRM['slew_mass_used'] = slew_mass_used.to('kg').value
            Obs.scMass = Obs.scMass - slew_mass_used
            DRM['scMass'] = Obs.scMass.to('kg').value
            # update current time by adding slew time for the chosen target XXX not needed
            # TK.allocate_time(slewTime[occ_sInd])
            if TK.mission_is_over():
                Logger.info('Mission complete: no more time available')
                print 'Mission complete: no more time available'
                return DRM, None, None
        
        return DRM, sInd, occ_sInd, t_det

    def choose_next_occulter_target(self, old_occ_sInd, occ_sInds, slewTime, t_dets):
        """Choose next target for the occulter based on truncated 
        depth first search of linear cost function.
        
        Args:
            old_occ_sInd (integer):
                Index of the previous target star
            occ_sInds (integer array):
                Indices of available targets
            slewTime (float array):
                slew times to all stars (must be indexed by sInds)
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

        nStars = len(sInds)

        # get available occulter target list
        startTime = TK.currentTimeAbs + slewTime
        kogoodStart = Obs.keepout(TL, occ_sInds, startTime, OS.telescopeKeepout)
        occ_sInds = occ_sInds[np.where(kogoodStart)[0]]

        # reshape sInds
        occ_sInds = np.array(occ_sInds,ndmin=1)

        # current stars have to be in the adjmat
        if (old_occ_sInd is not None) and (old_occ_sInd not in occ_sInds):
            occ_sInds = np.append(occ_sInds, old_occ_sInd)

        # get completeness values
        comps = TL.comp0[occ_sInds]
        updated = (self.starVisits[occ_sInds] > 0)
        comps[updated] = Comp.completeness_update(TL, occ_sInds[updated], TK.currentTimeNorm)
        
        # if first target, or if only 1 available target, choose highest available completeness
        nStars = len(occ_sInds)
        if (old_occ_sInds is None) or (nStars == 1):
            occ_sInd = np.random.choice(occ_sInds[comps == max(comps)])
            return occ_sInd
        
        # define adjacency matrix
        A = np.zeros((nStars,nStars))

        # only consider slew distance when there's an occulter
        if OS.haveOcculter:
            r_ts = TL.starprop(occ_sInds, TK.currentTimeAbs)
            u_ts = (r_ts.value.T/np.linalg.norm(r_ts,axis=1)).T
            angdists = np.arccos(np.clip(np.dot(u_ts,u_ts.T),-1,1))
            A[np.ones((nStars),dtype=bool)] = angdists
            A = self.coeffs[0]*(A)/np.pi

        # add factor due to completeness
        A = A + self.coeffs[1]*(1-comps)
        
        # add factor due to unvisited ramp
        # f_uv = np.zeros(nStars)
        # f_uv[self.starVisits[occ_sInds] == 0] = ((TK.currentTimeNorm / TK.missionFinishNorm)\
        #         .decompose().value)**2
        # A = A - self.coeffs[2]*f_uv

        # # add factor due to revisited ramp
        # f2_uv = np.where(self.starVisits[occ_sInds] > 0, self.starVisits[occ_sInds], 0) *\
        #         (1 - (np.in1d(occ_sInds, self.starRevisit[:,0],invert=True)))
        # A = A + self.coeffs[3]*f2_uv
        
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
        
        OS = self.OpticalSystem
        Obs = self.Observatory
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        nStars = len(sInds)

        # reshape sInds
        sInds = np.array(sInds,ndmin=1)

        # current stars have to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds, old_sInd)

        # 1/ Choose next telescope target
        comps = TL.comp0[sInds] # completeness of each star in TargetList
        updated = (self.starVisits[sInds] > 0)
        comps[updated] =  self.Completeness.completeness_update(self.TargetList, \
                sInds[updated], self.TimeKeeping.currentTimeNorm)

        # add weight for star revisits
        if self.starRevisit.size != 0:
            dt_max = 1.*u.week
            dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
            ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]

        f2_uv = np.where((self.starVisits[sInds] > 0) & (SSim.starVisits[sInds] < 6), 
                          self.starVisits[sInds], 0) * (1 - (np.in1d(sInds, ind_rev,invert=True)))

        weights = (comps + f2_uv/6.)/t_dets[sInds]
        sInd = np.random.choice(sInds[weights == max(weights)])
        
        return sInd

    def calc_int_inflection(self, sInd, fEZ, WA, mode):
        """Calculate integration time based on inflection point of Completeness as a function of int_time
        
        Args:
            sInd (integer):
                Index of the target star
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode

        Returns:
            int_time (float):
                The suggested integration time
        
        """

        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList

        dMagmin = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[1]*\
                  Comp.PlanetPopulation.Rprange[1]/Comp.PlanetPopulation.rrange[0])**2))
        dMagmax = OS.dMagLim
        num_points = 250

        dMags = np.linspace(dMagmin, dMagmax, num_points)

        # calculate t_det as a function of dMag
        fZ = ZL.fZ(Obs, TL, sInd, startTime[sInd], mode)
        t_dets = OS.calc_intTime(TL, sInd, fZ, fEZ, dMags, WA, mode)

        # calculate comp as a function of dMag
        smin = TL.dist[sInd] * np.tan(mode['IWA'])
        smax = TL.dist[sInd] * np.tan(mode['OWA'])

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
        
        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(Comp.classpath, Comp.filename+'.comp')
        Cpdf = pickle.load(open(Cpath, 'rb'))

        EVPOCpdf = interpolate.RectBivariateSpline(xedges, yedges, Cpdf.T)
        EVPOC = np.vectorize(EVPOCpdf.integral)

        comps = EVPOC(smin.to('AU').value, smax.to('AU').value, dMagmin, dMags)

        # find the inflection point of the completeness graph
        int_time = t_dets[np.where(np.gradient(comps) == max(np.gradient(comps)))[0]]

        # update star completeness
        idx = (np.abs(t_dets-int_time*self.starVisits[sInd])).argmin()
        comp = comps[idx]
        TL.comp[sInd] = comp

        return int_time


