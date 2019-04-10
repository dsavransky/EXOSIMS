from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import os
import astropy.units as u
import numpy as np
import time
import sys

# Python 3 compatibility:
if sys.version_info[0] > 2:
    xrange = range

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
        det_mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]
        
        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber + 1)
        self.logger.info(log_begin)

        self.vprint(log_begin)
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
                    self.vprint('Total fuel mass exceeded at %s'%TK.obsEnd.round(2))
                    break
        
        else:
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            self.vprint(log_end)

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
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # create DRM
        DRM = {}
        
        # allocate settling time + overhead time
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        
        # in case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        if OS.haveOcculter == True:
            ao = Obs.thrust/Obs.scMass
            slewTime_fac = (2.*Obs.occulterSep/np.abs(ao)/(Obs.defburnPortion/2. - 
                    Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        # now, start to look for available targets
        while not TK.mission_is_over():
            # 1/ initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            intTimes = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)
            
            # 2/ find spacecraft orbital START positions (if occulter, positions 
            # differ for each star) and filter out unavailable targets
            sd = None
            if OS.haveOcculter == True:
                # find angle between old and new stars, default to pi/2 for first target
                if old_sInd is None:
                    sd = np.array([np.radians(90)]*TL.nStars)*u.rad
                else:
                    # position vector of previous target star
                    r_old = TL.starprop(old_sInd, TK.currentTimeAbs)[0]
                    u_old = r_old.value/np.linalg.norm(r_old)
                    # position vector of new target stars
                    r_new = TL.starprop(sInds, TK.currentTimeAbs)
                    u_new = (r_new.value.T/np.linalg.norm(r_new, axis=1)).T
                    # angle between old and new stars
                    sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1))*u.rad
                # calculate slew time
                slewTimes = np.sqrt(slewTime_fac*np.sin(sd/2.))
            # start times, including slew times
            startTimes = TK.currentTimeAbs + slewTimes
            startTimesNorm = TK.currentTimeNorm + slewTimes
            # indices of observable stars
            kogoodStart = Obs.keepout(TL, sInds, startTimes, mode)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 3/ calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            if len(sInds) > 0:
                # assumed values for detection
                fZ = ZL.fZ(Obs, TL, sInds, startTimes[sInds], mode)
                fEZ = ZL.fEZ0
                dMag = self.dMagint[sInds]
                WA = self.WAint[sInds]
                intTimes[sInds] = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
                totTimes = intTimes*mode['timeMultiplier']
                # end times
                endTimes = startTimes + totTimes
                endTimesNorm = startTimesNorm + totTimes
                # indices of observable stars
                sInds = np.where((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))[0]
            
            # 4/ find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(sInds) > 0 and Obs.checkKeepoutEnd:
                kogoodEnd = Obs.keepout(TL, sInds, endTimes[sInds], mode)
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 5/ filter out all previously (more-)visited targets, unless in 
            # revisit list, with time within some dt of start (+- 1 week)
            if len(sInds) > 0:
                tovisit[sInds] = (self.starVisits[sInds] == min(self.starVisits[sInds]))
                if self.starRevisit.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] 
                            if x in sInds]
                    tovisit[ind_rev] = True
                sInds = np.where(tovisit)[0]
            
            # 6/ choose best target from remaining
            if len(sInds) > 0:
                # choose sInd of next target
                sInd = self.choose_next_target(old_sInd, sInds, slewTimes, intTimes[sInds])
                # store selected star integration time
                intTime = intTimes[sInd]
                break
            
            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.wait()
            
        else:
            return DRM, None, None
        
        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]
        
        # populate DRM with occulter related values
        if OS.haveOcculter == True:
            # find values related to slew time
            DRM['slew_time'] = slewTimes[sInd].to('day')
            DRM['slew_angle'] = sd[sInd].to('deg')
            slew_mass_used = slewTimes[sInd]*Obs.defburnPortion*Obs.flowRate
            DRM['slew_dV'] = (slewTimes[sInd]*ao*Obs.defburnPortion).to('m/s')
            DRM['slew_mass_used'] = slew_mass_used.to('kg')
            Obs.scMass = Obs.scMass - slew_mass_used
            DRM['scMass'] = Obs.scMass.to('kg')
            # update current time by adding slew time for the chosen target
            TK.allocate_time(slewTimes[sInd])
            if TK.mission_is_over():
                return DRM, None, None
        
        return DRM, sInd, intTime


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

    def calc_int_inflection(self, sInd, fEZ, fZ, WA, mode, ischar=False):
        """Calculate integration time based on inflection point of Completeness as a function of int_time
        
        Args:
            sInd (integer):
                Index of the target star
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            fZ (astropy Quantity):
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

        dMagmin = np.round(-2.5*np.log10(float(Comp.PlanetPopulation.prange[1]*\
                  Comp.PlanetPopulation.Rprange[1]/Comp.PlanetPopulation.rrange[0])**2))
        dMagmax = OS.dMagLim
        num_points = 250

        dMags = np.linspace(dMagmin, dMagmax, num_points)

        # calculate t_det as a function of dMag
        t_dets = OS.calc_intTime(TL, sInd, fZ, fEZ, dMags, WA, mode)

        # calculate comp as a function of dMag
        smin = TL.dist[sInd] * np.tan(mode['IWA'])
        smax = TL.dist[sInd] * np.tan(mode['OWA'])

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
            idx = np.abs(comps - max(comps)*.9).argmin()
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
            with open(Cpath, 'rb') as cfile:
                H = pickle.load(cfile)
        else:
            # run Monte Carlo simulation and pickle the resulting array
            self.vprint('Cached completeness file not found at "%s".' % Cpath)
            self.vprint('Beginning Monte Carlo completeness calculations.')
            
            t0, t1 = None, None # keep track of per-iteration time
            for i in xrange(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = '' # no message
                else:
                    delta_t_msg = '[%.3f s/iteration]' % (t1 - t0)

                self.vprint('Completeness iteration: %5d / %5d %s' % (i+1, steps, delta_t_msg))
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h

            Nplanets = 1e8
            
            H = H/(Nplanets*(xedges[1]-xedges[0])*(yedges[1]-yedges[0]))
                        
            # store 2D completeness pdf array as .comp file

            with open(Cpath, 'wb') as cfile:
                pickle.dump(H, cfile)
            self.vprint('Monte Carlo completeness calculations finished')
            self.vprint('2D completeness array stored in %r' % Cpath)
        
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
        # sample semi-major axis
        a = PPop.gen_sma(nplan).to('AU').value
        # sample other necessary orbital parameters
        if np.sum(PPop.erange) == 0:
            # all circular orbits
            r = a
            e = 0.0
            E = M
        else:
            # sample eccentricity
            if PPop.constrainOrbits:
                e = PPop.gen_eccen_from_sma(nplan,a*u.AU)
            else:
                e = PPop.gen_eccen(nplan)   
            # Newton-Raphson to find E
            E = eccanom(M,e)
            # orbital radius
            r = a*(1.0-e*np.cos(E))

        beta = np.arccos(1.0-2.0*np.random.uniform(size=nplan))*u.rad
        s = r*np.sin(beta)*u.AU
        # sample albedo, planetary radius, phase function
        p = PPop.gen_albedo(nplan)
        Rp = PPop.gen_radius(nplan)
        Phi = self.Completeness.PlanetPhysicalModel.calc_Phi(beta)
        
        # calculate dMag
        dMag = deltaMag(p,Rp,r*u.AU,Phi)
        return s, dMag
