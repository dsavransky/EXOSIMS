from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import time


class multiSS(SurveySimulation):
    def __init__(
        self,
        coeffs=[5, 1, 12, 10 * np.pi, 30],
        count=0,
        count_1=0,
        ko=0,
        ko_2=0,
        counter_3=0,
        **specs
    ):

        SurveySimulation.__init__(self, **specs)

        # verify that coefficients input is iterable 4x1
        if not (isinstance(coeffs, (list, tuple, np.ndarray))) or (len(coeffs) != 5):
            raise TypeError("coeffs must be a 4 element iterable")
        self.count = count
        self.count_1 = count_1
        self.ko = ko
        self.ko_2 = ko_2
        self.coeff = coeffs
        self.counter_3 = counter_3
        # Add to outspec
        self._outspec["coeffs"] = coeffs

        # normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs / np.linalg.norm(coeffs)

        # initialize the second target star
        self.second_target = None
        # to handle first two target case
        self.counter_2 = None
        self.oldInd = None

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

            # some new logic, subject to changes
            if (
                TK.currentTimeNorm.copy().value
                + waitTime.value
                + Obs.settlingTime.value
                + det_intTime.value
            ) > 1800:
                dtsim = (time.time() - t0) * u.s
                log_end = (
                    "Mission complete: no more time available.\n"
                    + "Simulation duration: %s.\n"
                    % dtsim.astype("int")
                    + "Results stored in SurveySimulation.DRM"
                      "(Design Reference Mission)."
                )
                self.logger.info(log_end)
                self.vprint(log_end)
                break

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
                DRM["waitTime"] = waitTime
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
                # observation end NormTime
                DRM["ObsEndTimeNorm"] = TK.currentTimeNorm.to("day").copy()
                DRM["ObsEndTimeAbs"] = TK.currentTimeAbs.copy()

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
                        if int(np.ceil(TK.currentTimeNorm.copy().value)) >= 1790:
                            tAbs = TK.currentTimeAbs.copy() + 20
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
        """
        OS = self.OpticalSystem
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        Comp = self.Completeness

        # create DRM
        DRM = {}

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # create appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # look for available targets

        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d

        # 1.2 Initialize array for slewTime array for second occulter
        slewTimes_2 = np.zeros(TL.nStars) * u.d

        # dV for both StarShades
        dV = np.zeros(TL.nStars) * u.m / u.s
        dV_2 = np.zeros(TL.nStars) * u.m / u.s

        intTimes = np.zeros(TL.nStars) * u.d
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        sd = None
        sd_2 = None

        # calculate the angular separation and slew times for both starshades,
        # now that 2 targets have been observed, this logic takes actual past 2 targets

        if OS.haveOcculter and self.count_1 == 1:
            sd = Obs.star_angularSep(
                TL, self.DRM[-1]["star_ind"], sInds, tmpCurrentTimeAbs
            )
            sd_2 = Obs.star_angularSep(
                TL, self.DRM[-2]["star_ind"], sInds, tmpCurrentTimeAbs
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, self.DRM[-1]["star_ind"], sInds, sd, 0, None
            )
            slewTimes_2 = Obs.calculate_slewTimes(
                TL, self.DRM[-2]["star_ind"], sInds, sd_2, 1, None
            )
            # print slewTimes
            self.slewTimes_2 = slewTimes_2

        self.sd = sd
        self.sd_2 = sd_2

        # take first 2 observations (first check if these are first two..)
        # assign all attributes relating to starshade be zero
        # (Considering for first two targets, Starsahdes don't slew from
        # a prior position. This logic can be edited later)
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)

        # filter targets which have NaN or 0 values
        sInds = sInds[np.where(intTimes[sInds] != 0)]
        sInds = sInds[
            np.where(intTimes[sInds] <= OS.intCutoff)
        ]  # Filters targets exceeding end of OB

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # filter out the slews which exceed max int time for targets
        # The int time for second target is assumed to be
        # less than maxInt time based on slewTime for first target

        comps = Comp.completeness_update(
            TL, sInds, self.starVisits[sInds], TK.currentTimeNorm.copy()
        )
        [X, Y] = np.meshgrid(comps, comps)
        c_mat = self.coeff[2] * (X + Y)

        # kill diagonal with arbitrary low number
        np.fill_diagonal(c_mat, 1e-9)

        # kill the upper half because the elements
        # are symmetrical (eg. comp(a,b), comp(b,a),
        # completeness assumed to be constant in
        # Time for one set of observation)
        np.tril(c_mat)

        if self.count_1 == 1:
            if len(sInds.tolist()) > 0 and old_sInd is not None:
                # choose sInd of next target

                sInd, waitTime = self.choose_next_target(
                    old_sInd, sInds, slewTimes[sInds], intTimes[sInds]
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
            else:
                DRM = Obs.log_occulterResults(
                    DRM, slewTimes_2[sInd], sInd, sd_2[sInd], dV_2[sInd]
                )
                self.count = 0
            return DRM, sInd, intTime, slewTimes[sInd]

        extraTimes = int(np.ceil(Obs.settlingTime.to("day").copy().value)) + int(
            np.ceil(mode["syst"]["ohTime"].copy().value)
        )

        if old_sInd is None and self.counter_2 is None:
            i = 0
            # change the int values to ceil to check for keepout
            while self.ko_2 == 0:
                H = np.unravel_index(c_mat.argmax(), c_mat.shape)
                first_target = sInds[H[0]]
                second_target = sInds[H[1]]

                # for generating a random schedule,
                # comment line 487,489 and use next 6 lines:
                # rng = np.random.default_rng()
                # a = rng.integers(0,len(sInds))
                # b = rng.integers(0,len(sInds))
                # H = [a,b]
                # irst_target = sInds[a]
                # second_target = sInds[b]

                # first target Obs start time
                t1 = int(np.ceil(TK.currentTimeNorm.copy().value))

                # first target Obs end time
                t2 = t1 + int(np.ceil(intTimes[first_target].value)) + extraTimes

                # second target Obs start time
                # this is t2 because, slewTime or waitTime is 0
                # for the second target in this sim

                # second target Obs end time
                t3 = t2 + int(np.ceil(intTimes[second_target].value)) + extraTimes
                self.ko_2 = np.all(koMap[first_target, t1:t2,].astype(int)) * np.all(
                    koMap[
                        second_target,
                        t2:t3,
                    ].astype(int)
                )
                i = i + 1
                if i % 5000 == 0:
                    print(i)
                if self.ko_2 == 1:
                    break
                else:
                    c_mat[H] = 0
                    self.ko_2 = 0

            slewTime = 0 * u.d
            sInd = first_target
            intTime = intTimes[sInd]
            waitTime = slewTime
            self.counter_2 = second_target
            self.starVisits[sInd] += 1
            DRM = Obs.log_occulterResults(DRM, 0 * u.d, sInd, 0 * u.rad, 0 * u.d / u.s)

        else:

            if self.count_1 == 0:
                sInd = self.counter_2
                slewTime = 0 * u.d
                intTime = intTimes[sInd]
                waitTime = slewTime
                self.starVisits[sInd] += 1
                DRM = Obs.log_occulterResults(
                    DRM, 0 * u.d, sInd, 0 * u.rad, 0 * u.d / u.s
                )
                self.count_1 = 1

                return DRM, sInd, intTime, waitTime

            return DRM, sInd, intTime, waitTime

        # store normalized start time for future completeness update

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

        extraTimes = int(np.ceil(Obs.settlingTime.copy().value)) + int(
            np.ceil(mode["syst"]["ohTime"].copy().value)
        )

        # cast sInds to array (pre-filtered target list)
        sInds = np.array(sInds, ndmin=1, copy=False)

        # ObsStartTime = self.DRM[-1]["ObsEndTimeNorm"]+ slewTimes.to("day")
        # ObsStartTime_2 = self.DRM[-2]["ObsEndTimeNorm"]+..
        # self.slewTimes_2[sInds].to("day")

        if self.counter_3 == 0:
            ObsStartTime = self.DRM[-1]["ObsEndTimeNorm"] + slewTimes.to("day")
            ObsStartTime_2 = self.DRM[-2]["ObsEndTimeNorm"] + self.slewTimes_2[
                sInds
            ].to("day")
            self.ObsStartTime = ObsStartTime
            self.ObsStartTime_2 = ObsStartTime_2
            self.counter_3 = self.counter_3 + 1
        else:
            ObsStartTime = self.ObsStartTime
            ObsStartTime_2 = self.ObsStartTime_2
            self.counter_3 = 0

        # appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # integration time for all the targets that will be observed second
        intTimes_2 = intTimes

        ang_sep = abs(self.sd[sInds].value)
        ang_sep2 = abs(self.sd_2[sInds].value)

        a, b = np.meshgrid(ang_sep2, ang_sep)
        ang_cost = -self.coeff[3] * (a + b)

        SV = self.starVisits[sInds]
        x, y = np.meshgrid(SV, SV)
        Star_visit_cost = self.coeff[1] * (x + y)

        comps = Comp.completeness_update(
            TL, sInds, self.starVisits[sInds], TK.currentTimeNorm.copy()
        )
        [X, Y] = np.meshgrid(comps, comps)
        compcost = self.coeff[2] * (X + Y) * TK.currentTimeNorm.value.copy()

        m, n = np.meshgrid(self.slewTimes_2[sInds], slewTimes)
        slew_cost = -self.coeff[0] * ((m + n) / np.linalg.norm(m + n))

        P, Q = np.meshgrid(intTimes, intTimes)
        intcost = -self.coeff[4] * ((P + Q) / np.linalg.norm(P + Q))

        c_mat = (
            Star_visit_cost
            + slew_cost * np.e ** (1 / (TK.currentTimeNorm.value.copy()))
            + intcost
            + ang_cost * np.e ** (1 / (TK.currentTimeNorm.value.copy()))
            + compcost
        )

        # kill diagonal with 0

        np.fill_diagonal(c_mat, 0)

        # kill the upper half because the elements are symmetrical
        # (eg. comp(a,b), comp(b,a),
        # completeness assumed to be constant in Time for one set of observation)
        c_mat = np.tril(c_mat)

        if self.second_target is None:
            i = 0
            j = 0

            while self.ko == 0:
                # for using random scheduler, comment/uncomment lines 641--646
                h = np.unravel_index(c_mat.argmax(), c_mat.shape)
                first_target_sInd = [h[0]]
                second_target_sInd = [h[1]]

                # rng = np.random.default_rng()
                # A = rng.integers(0,len(sInds))
                # B = rng.integers(0,len(sInds))
                # h = [A,B]
                # first_target_sInd = A
                # second_target_sInd = B

                if int(np.ceil(TK.currentTimeNorm.copy().value)) > int(
                    np.ceil(ObsStartTime_2[first_target_sInd].value)
                ):
                    T1 = int(np.ceil(TK.currentTimeNorm.copy().value))
                else:
                    T1 = int(np.ceil(ObsStartTime_2[first_target_sInd].value))

                DT1 = extraTimes + int(np.ceil(intTimes[first_target_sInd].value)) + T1

                tempT2 = DT1 + T1

                if tempT2 > int(np.ceil(ObsStartTime[second_target_sInd].value)):
                    T2 = tempT2
                else:
                    T2 = int(np.ceil(ObsStartTime[second_target_sInd].value))

                DT2 = (
                    T2 + extraTimes + int(np.ceil(intTimes_2[second_target_sInd].value))
                )

                self.ko = np.all(koMap[sInds[h[0]], T1:DT1,]) * np.all(
                    koMap[
                        sInds[h[1]],
                        T2:DT2,
                    ]
                )

                if self.ko == 1:
                    self.ko = 0
                    # set ko to be 0 again for second set of target search
                    break
                else:
                    # print(h)
                    i = i + 1
                    j = j + 1
                    c_mat[h] = 0

                    if i % 10000 == 0:
                        print(i)

                    # advance by 10 days if no set found, check
                    # for 50,000 elements and then increment the time again
                    if i >= 50000 and j == 50000:
                        _ = TK.allocate_time(10 * u.d)
                        # check mission time
                        dt = TK.currentTimeNorm.copy()
                        comps = Comp.completeness_update(
                            TL, sInds, self.starVisits[sInds], dt
                        )
                        [X, Y] = np.meshgrid(comps, comps)
                        c_mat = self.coeff[2] * (X + Y)
                        """+ ang_cost + Star_visit_cost"""
                        np.fill_diagonal(c_mat, 0)
                        c_mat = np.tril(c_mat)
                        print(i)
                        print(TK.currentTimeNorm.copy())
                        j = 0

                    self.ko = 0

            # get the current target
            sInd = sInds[h[0]]

            self.second_target = h[1]
            self.oldInd = sInds[self.second_target]
            # checking if starshade reached target before the
            # observation or there is waittime before observation starts

            if ObsStartTime[first_target_sInd] > TK.currentTimeNorm.copy():
                waittime = ObsStartTime[first_target_sInd] - TK.currentTimeNorm.copy()
            else:
                waittime = 0 * u.d
            self.starVisits[sInd] += 1
        else:

            if ObsStartTime_2[self.second_target] > TK.currentTimeNorm.copy():
                waittime = (
                    ObsStartTime_2[self.second_target] - TK.currentTimeNorm.copy()
                )
            else:
                waittime = 0 * u.d
            # uncomment this line or not?
            # sInd = sInds[self.second_target]
            sInd = self.oldInd
            self.oldInd = None
            self.second_target = None
            self.starVisits[sInd] += 1
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
            DRM["scMass_first"] = Obs.scMass[:, 0].to("kg")
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
