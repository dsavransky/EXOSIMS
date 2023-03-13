from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
from astropy.time import Time
import time


class multiSS(SurveySimulation):
    def __init__(
        self, coeffs=[-1, -2, np.e, np.pi], count=0, count_1=0, ko=1, ko_2=1, **specs
    ):

        SurveySimulation.__init__(self, **specs)

        # verify that coefficients input is iterable 4x1
        if not (isinstance(coeffs, (list, tuple, np.ndarray))) or (len(coeffs) != 4):
            raise TypeError("coeffs must be a 4 element iterable")
        self.count = count
        self.count_1 = count_1
        self.ko = ko
        self.ko_2 = ko_2
        self.coeff = coeffs
        # Add to outspec
        self._outspec["coeffs"] = coeffs

        # normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs / np.linalg.norm(coeffs)

        # initialize the second target star
        self.second_target = None
        # to handle first two target case
        self.counter_2 = None

    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter:
            self.currentSep = Obs.occulterSep

        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(
            filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
        )
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s: survey beginning." % (TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        while not TK.mission_is_over(OS, Obs, det_mode):

            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd  # used to save sInd if returned sInd is None
            DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)
            

            if sInd is not None:
                ObsNum += (
                    1  # we're making an observation so increment observation number
                )

                if OS.haveOcculter:
                    # advance to start of observation (add slew time for selected target
                    _ = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)

                # beginning of observation, start to populate DRM
                DRM["star_ind"] = sInd
                DRM["star_name"] = TL.Name[sInd]
                DRM["arrival_time"] = TK.currentTimeNorm.to("day").copy()
                DRM["OB_nb"] = TK.OBnumber
                DRM["ObsNum"] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM["plan_inds"] = pInds.astype(int)
                log_obs = (
                    "  Observation #%s, star ind %s (of %s) with %s planet(s), "
                    + "mission time at Obs start: %s, exoplanetObsTime: %s"
                ) % (
                    ObsNum,
                    sInd,
                    TL.nStars,
                    len(pInds),
                    TK.currentTimeNorm.to("day").copy().round(2),
                    TK.exoplanetObsTime.to("day").copy().round(2),
                )
                self.logger.info(log_obs)
                self.vprint(log_obs)

                # PERFORM DETECTION and populate revisit list attribute
                (
                    detected,
                    det_fZ,
                    det_systemParams,
                    det_SNR,
                    FA,
                ) = self.observation_detection(sInd, det_intTime.copy(), det_mode)
                # update the occulter wet mass
                if OS.haveOcculter:
                    DRM = self.update_occulter_mass(
                        DRM, sInd, det_intTime.copy(), "det"
                    )
                # populate the DRM with detection results
                DRM["det_time"] = det_intTime.to("day")
                DRM["det_status"] = detected
                DRM["det_SNR"] = det_SNR
                DRM["det_fZ"] = det_fZ.to("1/arcsec2")
                DRM["det_params"] = det_systemParams

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                if char_mode["SNR"] not in [0, np.inf]:
                    (
                        characterized,
                        char_fZ,
                        char_systemParams,
                        char_SNR,
                        char_intTime,
                    ) = self.observation_characterization(sInd, char_mode)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if FA else len(pInds)
                    characterized = np.zeros(lenChar, dtype=float)
                    char_SNR = np.zeros(lenChar, dtype=float)
                    char_fZ = 0.0 / u.arcsec**2
                    char_systemParams = SU.dump_system_params(sInd)
                assert char_intTime != 0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter and (char_intTime is not None):
                    DRM = self.update_occulter_mass(DRM, sInd, char_intTime, "char")
                # populate the DRM with characterization results
                DRM["char_time"] = (
                    char_intTime.to("day") if char_intTime else 0.0 * u.day
                )
                DRM["char_status"] = characterized[:-1] if FA else characterized
                DRM["char_SNR"] = char_SNR[:-1] if FA else char_SNR
                DRM["char_fZ"] = char_fZ.to("1/arcsec2")
                DRM["char_params"] = char_systemParams
                # populate the DRM with FA results
                DRM["FA_det_status"] = int(FA)
                DRM["FA_char_status"] = characterized[-1] if FA else 0
                DRM["FA_char_SNR"] = char_SNR[-1] if FA else 0.0
                DRM["FA_char_fEZ"] = (
                    self.lastDetected[sInd, 1][-1] / u.arcsec**2
                    if FA
                    else 0.0 / u.arcsec**2
                )
                DRM["FA_char_dMag"] = self.lastDetected[sInd, 2][-1] if FA else 0.0
                DRM["FA_char_WA"] = (
                    self.lastDetected[sInd, 3][-1] * u.arcsec if FA else 0.0 * u.arcsec
                )

                # populate the DRM with observation modes
                DRM["det_mode"] = dict(det_mode)
                del DRM["det_mode"]["inst"], DRM["det_mode"]["syst"]
                DRM["char_mode"] = dict(char_mode)
                del DRM["char_mode"]["inst"], DRM["char_mode"]["syst"]
                DRM["exoplanetObsTime"] = TK.exoplanetObsTime.copy()

                # append result values to self.DRM
                self.DRM.append(DRM)

                # handle case of inf OBs and missionPortion < 1
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1.0):
                    self.arbitrary_time_advancement(
                        TK.currentTimeNorm.to("day").copy() - DRM["arrival_time"]
                    )

            else:  # sInd == None
                sInd = old_sInd  # Retain the last observed star
                if (
                    TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]
                ):  # currentTime is at end of OB
                    # Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(
                        OS, Obs, det_mode
                    ):  # as long as the mission is not over
                        TK.advancetToStartOfNextOB()  # Advance To Start of Next OB
                elif waitTime is not None:
                    # CASE 1: Advance specific wait time
                    _ = TK.advanceToAbsTime(
                        TK.currentTimeAbs.copy() + waitTime,
                        self.defaultAddExoplanetObsTime,
                    )
                    self.vprint("waitTime is not None")
                else:
                    startTimes = (
                        TK.currentTimeAbs.copy() + np.zeros(TL.nStars) * u.d
                    )  # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(
                        TL,
                        np.arange(TL.nStars),
                        startTimes,
                        self.koMaps,
                        self.koTimes,
                        det_mode,
                    )[0]
                    # CASE 2 If There are no observable targets for the rest of the
                    # mission
                    if (
                        observableTimes[
                            (
                                TK.missionFinishAbs.copy().value * u.d
                                > observableTimes.value * u.d
                            )
                            * (
                                observableTimes.value * u.d
                                >= TK.currentTimeAbs.copy().value * u.d
                            )
                        ].shape[0]
                    ) == 0:
                        self.vprint(
                            (
                                "No Observable Targets for Remainder of mission at "
                                "currentTimeNorm = {}"
                            ).format(TK.currentTimeNorm)
                        )
                        # Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife
                        TK.currentTimeAbs = TK.missionFinishAbs
                    # CASE 3 nominal wait time if at least 1 target is still in list
                    # and observable
                    else:
                        # TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[
                            observableTimes.value * u.d
                            > TK.currentTimeAbs.copy().value * u.d
                        ]
                        # apply intTime filter
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1)
                        # apply revisit Filter #NOTE this means stars you added to the
                        # revisit list
                        inds3 = self.revisitFilter(
                            inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)
                        )
                        self.vprint(
                            "Filtering %d stars from advanceToAbsTime"
                            % (TL.nStars - len(inds3))
                        )
                        oTnowToEnd = observableTimes[inds3]
                        # there is at least one observableTime between now and the end
                        # of the mission
                        if not oTnowToEnd.value.shape[0] == 0:
                            # advance to that observable time
                            tAbs = np.min(oTnowToEnd)
                        else:
                            tAbs = (
                                TK.missionStart + TK.missionLife
                            )  # advance to end of mission
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        # Advance Time to this time OR start of next
                        # OB following this time
                        _ = TK.advanceToAbsTime(tAbs, self.defaultAddExoplanetObsTime)
                        self.vprint(
                            (
                                "No Observable Targets a currentTimeNorm= {:.2f}. "
                                "Advanced To {:.2f}"
                            ).format(
                                tmpcurrentTimeNorm.to("day"),
                                TK.currentTimeNorm.to("day"),
                            )
                        )
        else:  # TK.mission_is_over()
            dtsim = (time.time() - t0) * u.s
            log_end = (
                "Mission complete: no more time available.\n"
                + "Simulation duration: %s.\n" % dtsim.astype("int")
                + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            )
            self.logger.info(log_end)
            self.vprint(log_end)

    def next_target(self, old_sInd, mode):
        """Finds index of next target star and calculates its integration time.

        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.

        Args:
            old_sInd (int):
                Index of the previous target star
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                DRM (dict):
                    Design Reference Mission, contains the results of one complete
                    observation (detection and characterization)
                sInd (int):
                    Index of next target star. Defaults to None.
                intTime (astropy.units.Quantity):
                    Selected star integration time for detection in units of day.
                    Defaults to None.
                waitTime (astropy.units.Quantity):
                    a strategically advantageous amount of time to wait in the case
                    of an occulter for slew times

                    change the surveySim method to handle the case when observations are not yet done by assigning slewTime/ sd/ waitTime to be None or acceptable outputs
        """
        OS = self.OpticalSystem
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        Comp = self.Completeness
        allModes = OS.observingModes

        # create DRM
        DRM = {}

        """self.DRM.append(DRM)"""  # change this

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # create appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        # 1.2 Initialize array for slewTime array for second occulter
        slewTimes_2 = np.zeros(TL.nStars) * u.d
        fZs = np.zeros(TL.nStars) / u.arcsec**2.0
        # dV for both StarShades
        dV = np.zeros(TL.nStars) * u.m / u.s
        dV_2 = np.zeros(TL.nStars) * u.m / u.s

        intTimes = np.zeros(TL.nStars) * u.d
        obsTimes = np.zeros([2, TL.nStars]) * u.d
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        sd = None
        sd_2 = None

        # calculate the angular separation and slew times for both starshades, now that 2 targets have been observed, this logic takes actual past 2 targets

        if OS.haveOcculter and self.count_1 == 1:
            sd = Obs.star_angularSep(
                TL, self.DRM[-1]["star_ind"], sInds, tmpCurrentTimeAbs
            )
            sd_2 = Obs.star_angularSep(
                TL, self.DRM[-2]["star_ind"], sInds, tmpCurrentTimeAbs
            )
            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, self.DRM[-1]["star_ind"], sInds, sd, 0, None
            )
            slewTimes_2 = Obs.calculate_slewTimes(
                TL, self.DRM[-2]["star_ind"], sInds, sd_2, 1, None
            )

            self.slewTimes_2 = slewTimes_2
        
        # take first 2 observations (first check if these are first two..) assign all attributes relating to starshade be zero
        # (Considering for first two targets, Starsahdes don't slew from a prior position. This logic can be edited later)
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # 3. filter out all previously (more-)visited targets, unless in
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # 4.1 calculate integration times for ALL preselected targets
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
        )  # Maximum intTime allowed

        

        """if len(sInds.tolist()) > 0:
            if OS.haveOcculter and (old_sInd is not None) and self.count_1 == 1:
                (
                    sInds,
                    slewTimes[sInds],
                    intTimes[sInds],
                    dV[sInds],
                ) = self.refineOcculterSlews(
                    self.DRM[-1]["star_ind"], sInds, slewTimes, obsTimes, sd, mode
                )
                (
                    sInds,
                    slewTimes_2[sInds],
                    intTimes[sInds],
                    dV_2[sInds],
                ) = self.refineOcculterSlews(
                    self.DRM[-2]["star_ind"],
                    sInds,
                    slewTimes_2,
                    obsTimes,
                    sd_2,
                    mode,
                )
                endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
            else:
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)
                sInds = sInds[
                    np.where(intTimes[sInds] <= maxIntTime)
                ]  # Filters targets exceeding end of OB
                endTimes = tmpCurrentTimeAbs.copy() + intTimes

                if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)"""
        intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode) 

        sInds = sInds[
                    np.where(intTimes[sInds] <= maxIntTime)
                ]  # Filters targets exceeding end of OB
        
        if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)

        #filter out the slews which exceed max int time for targets 
        #The int time for second target is assumed to be less than maxInt time based on slewTime for first target
        if self.count_1 == 1:
            slewTimes = slewTimes[sInds]
            self.slewTimes_2 = self.slewTimes_2[sInds]
                
        dt = TK.currentTimeNorm.copy()

        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        [X, Y] = np.meshgrid(comps, comps)
        c_mat = X + Y

        # kill diagonal with arbitrary low number
        np.fill_diagonal(c_mat, 1e-9)

        #kill the upper half because the elements are symmetrical (eg. comp(a,b), comp(b,a), 
        # completeness assumed to be constant in Time for one set of observation)
        np.tril(c_mat)

        

        # 5. choose best target from remaining
        """if len(sInds.tolist()) > 0 and old_sInd is None:

            # calculating the first target star based on maximum completeness value (THIS LOGIC WILL BE ELIMINATED ONCE THE FIRST_TWO_TARGET IS IMPLEMENTED)

            # calculate dt since previous observation
            dt = TK.currentTimeNorm.copy()
            # get dynamic completeness values
            comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
            # choose target with maximum completeness
            sInd = np.random.choice(sInds[comps == max(comps)])

            # Check if exoplanetObsTime would be exceeded
            mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
            (
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
            ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
            maxIntTime = min(
                maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
            )  # Maximum intTime allowed
            intTimes2 = self.calc_targ_intTime(
                np.array([sInd]), TK.currentTimeAbs.copy(), mode
            )
            if (
                intTimes2 > maxIntTime
            ):  # check if max allowed integration time would be exceeded
                self.vprint("max allowed integration time would be exceeded")
                sInd = None
                waitTime = 1.0 * u.d
                return sInd, waitTime
            DRM = Obs.log_occulterResults(
                DRM, slewTimes[sInd], sInd, sd[sInd], dV[sInd]
            )
            waitTime = slewTimes[sInd]
            return (DRM, sInd, slewTimes[sInd], waitTime)"""

        if self.count_1 ==1:
            if len(sInds.tolist()) > 0 and old_sInd is not None:
                # choose sInd of next target
                sInd, waitTime = self.choose_next_target(
                    old_sInd, sInds, slewTimes, intTimes[sInds]
                )

                # Should Choose Next Target decide there are no stars it wishes to
                # observe at this time.
                if sInd is None and (waitTime is not None):
                    self.vprint(
                        "There are no stars available to observe. Waiting {}".format(
                            waitTime
                        )
                    )
                    return DRM, None, None, waitTime
                elif (sInd is None) and (waitTime is None):
                    self.vprint(
                        "There are no stars available to observe and waitTime is None."
                    )
                    return DRM, None, None, waitTime
                # store selected star integration time
                intTime = intTimes[sInd]

            # if no observable target, advanceTime to next Observable Target
            else:
                self.vprint(
                    "No Observable Targets at currentTimeNorm= "
                    + str(TK.currentTimeNorm.copy())
                )
                return DRM, None, None, None


        # populate DRM with occulter related values
        if OS.haveOcculter and self.count_1 == 1:
            if self.count == 0:
                DRM = Obs.log_occulterResults(
                    DRM, slewTimes[sInd], sInd, sd[sInd], dV[sInd]
                )
                self.count = self.count + 1
                print("Done")
            else:
                DRM = Obs.log_occulterResults(
                    DRM, slewTimes_2[sInd], sInd, sd_2[sInd], dV_2[sInd]
                )
                self.count = 0
                print("Done_2")

            return DRM, sInd, intTime, slewTimes[sInd]
        
        if old_sInd is None and self.counter_2 is None:
            i = 0
            #change the int values to ceil to check for keepout
            while self.ko_2 == 1:
                H = np.unravel_index(c_mat.argmax(), c_mat.shape)
                first_target = H[0]
                second_target = H[1]
                self.ko_2 = np.prod(
                    koMap[
                        first_target,
                        int(TK.currentTimeNorm.copy().value) : int(
                            TK.currentTimeNorm.copy().value
                        )
                        + int(intTimes[first_target].value),
                    ].astype(int)
                )
                +np.prod(
                    koMap[
                        second_target,
                        int(TK.currentTimeNorm.copy().value)
                        + int(intTimes[first_target].value) : int(
                            TK.currentTimeNorm.copy().value
                        )
                        + int(intTimes[first_target].value)
                        + int(intTimes[second_target].value),
                    ].astype(int)
                )
                i = i+1
                print(i)
                if self.ko_2 == 0:
                    pass
                else:
                    c_mat[H] = 1e-9
                    self.ko_2 = 1

            slewTime = 0 * u.d
            sInd = first_target  
            intTime = intTimes[sInd]
            waitTime = slewTime
            self.counter_2 = second_target
            DRM = Obs.log_occulterResults(DRM, 0 * u.d, sInd, 0 * u.rad, 0 * u.d / u.s)
            
            #print(self.starVisits)

        else:

            if self.count_1 == 0:
                sInd = self.counter_2
                slewTime = 0 * u.d
                intTime = intTimes[sInd]
                waitTime = 2*u.d
                DRM = Obs.log_occulterResults(DRM, 0 * u.d, sInd, 0 * u.rad, 0 * u.d / u.s)
                self.count_1 = 1
                
                

                return DRM, sInd, intTime, waitTime 

            return DRM, sInd, intTime, waitTime
        
        # update visited list for selected star
        
        self.starVisits[sInd] += 1

        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]


        return DRM, sInd, intTime, waitTime
    

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):

        """Choose next target based on truncated depth first search
        of linear cost function.

        Args:
            old_sInd (int):
                Index of the previous target star
            sInds (int array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (int):
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

        mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # cast sInds to array (pre-filtered target list)
        sInds = np.array(sInds, ndmin=1, copy=False)

        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimes_2 = tmpCurrentTimeAbs.copy() + self.slewTimes_2
        
        """print(len(self.slewTimes_2))
        print(len(startTimes))
        print(len(startTimes_2))
        print(len(sInds))
"""
        if len(startTimes) == len(sInds) == len(startTimes_2):
            pass
        else:
            print("No")
    
    

        intTimes_2 = np.zeros(len(sInds))
        # integration time for all the possible target first 
        intTimes = self.calc_targ_intTime(sInds, startTimes, mode)

        # integration time for all the targets that will be observed second
        intTimes_2 = self.calc_targ_intTime(sInds, startTimes_2, mode)
        # appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # calculate dt since previous observation
        "uncomment these lines later"
        #dt = TK.currentTimeNorm.copy() + slewTimes - self.lastObsTimes
        # get dynamic completeness values (use it for later purposes)
        #comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)

        # defining a dummy cost matrix for random walk scheduler
        """cost_matrix = np.ones([len(sInds), len(sInds)])
        print(cost_matrix)
        A = np.random.randint(1000,size = (len(sInds),len(sInds)))
        print(A)
        cost_matrix = cost_matrix*A
        # kill diagonal
        print(cost_matrix)
        np.fill_diagonal(cost_matrix, 1e9)
        print(cost_matrix)"""
        dt = TK.currentTimeNorm.copy()

        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)

        [X, Y] = np.meshgrid(comps, comps)
        c_mat = X + Y

        # kill diagonal with arbitrary low number
        np.fill_diagonal(c_mat, 1e-9)

        #kill the upper half because the elements are symmetrical (eg. comp(a,b), comp(b,a), 
        # completeness assumed to be constant in Time for one set of observation)
        np.tril(c_mat)
        if self.second_target is None:
            i = 0
            
            while self.ko == 1:
                # figure out the next two steps, edit method to select a random index instead of an element from array.
                #edit this logic as done above for first two targets
                h = np.unravel_index(c_mat.argmax(), c_mat.shape)
                first_target_sInd = h[0]
                second_target_sInd = h[1]
                np.prod(
                    koMap[
                        first_target_sInd,
                        int(TK.currentTimeNorm.copy().value) : int(TK.currentTimeNorm.copy().value)
                        +int(intTimes[first_target_sInd].value) + int(slewTimes[first_target_sInd].value),
                    ]
                ) + np.prod(
                    koMap[
                        second_target_sInd,
                        int(TK.currentTimeNorm.copy().value) + int(intTimes[first_target_sInd].value)
                        + int(slewTimes[first_target_sInd].value) : int(TK.currentTimeNorm.copy().value)
                        + int(intTimes[first_target_sInd].value)
                        + int(slewTimes[first_target_sInd].value)
                        + int(self.slewTimes_2[second_target_sInd].value)
                        + int(intTimes_2[second_target_sInd].value),
                    ]
                )
                break
                if self.ko == 0:
                    print("working")
                    pass
                else:
                    print(h)
                    """i = i+1
                    print(i)"""
                    c_mat[h] = 1e-9
                    self.ko = 1

            # get the current target
            sInd = first_target_sInd

            self.second_target = second_target_sInd
            waittime = slewTimes[sInd]
        else:
            sInd = self.second_target
            waittime = self.slewTimes_2[sInd]
            self.second_target = None
        return sInd, waittime

    def update_occulter_mass(self, DRM, sInd, t_int, skMode):
        """Updates the occulter wet mass in the Observatory module, and stores all
        the occulter related values in the DRM array.

        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            sInd (int):
                Integer index of the star of interest
            t_int (~astropy.units.Quantity(~numpy.ndarray(float))):
                Selected star integration time (for detection or characterization)
                in units of day
            skMode (str):
                Station keeping observing mode type ('det' or 'char')

        Returns:
            dict:
                Design Reference Mission dictionary, contains the results of one
                complete observation

        """

        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        assert skMode in ("det", "char"), "Observing mode type must be 'det' or 'char'."

        if self.second_target is None:

            # decrement mass for station-keeping
            dF_lateral, dF_axial, intMdot, mass_used, deltaV = Obs.mass_dec_sk(
                TL, sInd, TK.currentTimeAbs.copy(), t_int
            )

            DRM[skMode + "_dV"] = deltaV.to("m/s")
            DRM[skMode + "_mass_used"] = mass_used.to("kg")
            DRM[skMode + "_dF_lateral"] = dF_lateral.to("N")
            DRM[skMode + "_dF_axial"] = dF_axial.to("N")
            # update current spacecraft mass
            Obs.scMass[:, 0] = Obs.scMass[:, 0] - mass_used
            DRM["scMass_first"] = Obs.scMass[0].to("kg")
            if Obs.twotanks:
                Obs.skMass = Obs.skMass - mass_used
                DRM["skMass"] = Obs.skMass.to("kg")

        else:
            # decrement mass for station-keeping
            dF_lateral, dF_axial, intMdot, mass_used, deltaV = Obs.mass_dec_sk(
                TL, sInd, TK.currentTimeAbs.copy(), t_int
            )

            DRM[skMode + "_dV"] = deltaV.to("m/s")
            DRM[skMode + "_mass_used"] = mass_used.to("kg")
            DRM[skMode + "_dF_lateral"] = dF_lateral.to("N")
            DRM[skMode + "_dF_axial"] = dF_axial.to("N")
            # update current spacecraft mass
            Obs.scMass[:, 1] = Obs.scMass[:, 1] - mass_used
            DRM["scMass_second"] = Obs.scMass[:, 1].to("kg")
            if Obs.twotanks:
                Obs.skMass = Obs.skMass - mass_used
                DRM["skMass"] = Obs.skMass.to("kg")

        return DRM
