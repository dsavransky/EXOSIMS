from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import astropy.constants as const
import numpy as np
import time
import copy
from EXOSIMS.util.deltaMag import deltaMag


class coroOnlyScheduler(SurveySimulation):
    """coroOnlyScheduler - Coronograph Only Scheduler

    This scheduler inherits directly from the prototype SurveySimulation module.

    The coronlyScheduler operates using only a coronograph. The scheduler makes
    detections until stars can be promoted into a characterization list, at which
    point they are charcterized.

    Args:
        revisit_wait (float):
            Wait time threshold for star revisits. The value given is the fraction of a
            characterized planet's period that must be waited before scheduling a
            revisit.
        revisit_weight (float):
            Weight used to increase preference for coronograph revisits.
        n_det_remove (integer):
            Number of failed detections before a star is removed from the target list.
        n_det_min (integer):
            Minimum number of detections required for promotion to char target.
        max_successful_chars (integer):
            Maximum number of successful characterizions before star is taken off
            target list.
        max_successful_dets (integer):
            Maximum number of successful detections before star is taken off target
            list.
        lum_exp (int):
            The exponent to use for luminosity weighting on coronograph targets.
        promote_by_time (bool):
            Only promote stars that have had detections that span longer than half
            a period.
        detMargin (float):
            Acts in the same way a charMargin. Adds a multiplyer to the calculated
            detection time.
        **specs:
            user specified values

    """

    def __init__(
        self,
        revisit_wait=0.5,
        revisit_weight=1.0,
        n_det_remove=3,
        n_det_min=3,
        max_successful_chars=1,
        max_successful_dets=4,
        lum_exp=1,
        promote_by_time=False,
        detMargin=0.0,
        **specs
    ):

        SurveySimulation.__init__(self, **specs)

        TL = self.TargetList
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse

        # Add to outspec
        self._outspec["revisit_wait"] = revisit_wait
        self._outspec["revisit_weight"] = revisit_weight
        self._outspec["n_det_remove"] = n_det_remove
        self._outspec["max_successful_chars"] = max_successful_chars
        self._outspec["lum_exp"] = lum_exp
        self._outspec["n_det_min"] = n_det_min

        self.FA_status = np.zeros(TL.nStars, dtype=bool)  # False Alarm status array
        # The exponent to use for luminosity weighting on coronograph targets
        self.lum_exp = lum_exp

        self.sInd_charcounts = {}  # Number of characterizations by star index
        self.sInd_detcounts = np.zeros(
            TL.nStars, dtype=int
        )  # Number of detections by star index
        self.sInd_dettimes = {}
        # Minimum number of visits with no detections required to filter off star
        self.n_det_remove = n_det_remove
        # Minimum number of detections required for promotio
        self.n_det_min = n_det_min
        # Maximum allowed number of successful chars of deep dive targets before
        # removal from target list
        self.max_successful_chars = max_successful_chars
        self.max_successful_dets = max_successful_dets
        self.char_starRevisit = np.array([])  # Array of star revisit times
        # The number of times each star was visited by the occulter
        self.char_starVisits = np.zeros(TL.nStars, dtype=int)
        self.promote_by_time = promote_by_time
        self.detMargin = detMargin

        # self.revisit_wait = revisit_wait * u.d
        EEID = 1 * u.AU * np.sqrt(TL.L)
        mu = const.G * (TL.MsTrue)
        T = (2.0 * np.pi * np.sqrt(EEID**3 / mu)).to("d")
        self.revisit_wait = revisit_wait * T

        self.revisit_weight = revisit_weight
        self.no_dets = np.ones(self.TargetList.nStars, dtype=bool)
        # list of stars promoted from the detection list to the characterization list
        self.promoted_stars = self.known_rocky
        # list of stars that have been removed from the occ_sInd list
        self.ignore_stars = []
        self.t_char_earths = np.array([])  # corresponding integration times for earths

        allModes = OS.observingModes
        num_char_modes = len(
            list(filter(lambda mode: "spec" in mode["inst"]["name"], allModes))
        )
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)

        # Promote all stars assuming they have known earths
        char_sInds_with_earths = []
        if TL.earths_only:

            OS = self.OpticalSystem
            TL = self.TargetList
            SU = self.SimulatedUniverse
            # char_modes = list(
            #    filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
            # )

            # check for earths around the available stars
            for sInd in np.arange(TL.nStars):
                pInds = np.where(SU.plan2star == sInd)[0]
                pinds_earthlike = self.is_earthlike(pInds, sInd)
                if np.any(pinds_earthlike):
                    self.known_earths = np.union1d(
                        self.known_earths, pInds[pinds_earthlike]
                    ).astype(int)
                    char_sInds_with_earths.append(sInd)
            self.promoted_stars = np.union1d(
                self.promoted_stars, char_sInds_with_earths
            ).astype(int)

    def initializeStorageArrays(self):
        """
        Initialize all storage arrays based on # of stars and targets
        """

        self.DRM = []
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        allModes = OS.observingModes
        num_char_modes = len(
            list(filter(lambda mode: "spec" in mode["inst"]["name"], allModes))
        )
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.propagTimes = np.zeros(self.TargetList.nStars) * u.d
        self.lastObsTimes = np.zeros(self.TargetList.nStars) * u.d
        self.starVisits = np.zeros(
            self.TargetList.nStars, dtype=int
        )  # contains the number of times each star was visited
        self.starRevisit = np.array([])
        self.starExtended = np.array([], dtype=int)
        self.lastDetected = np.empty((self.TargetList.nStars, 4), dtype=object)

    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        Comp = self.Completeness

        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_modes = list(
            filter(lambda mode: "imag" in mode["inst"]["name"], OS.observingModes)
        )
        base_det_mode = list(
            filter(lambda mode: mode["detectionMode"], OS.observingModes)
        )[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(
            filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
        )
        if np.any(spectroModes):
            char_modes = spectroModes
        # if no spectro mode, default char mode is first observing mode
        else:
            char_modes = [allModes[0]]

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s: survey beginning." % (TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        while not TK.mission_is_over(OS, Obs, det_modes[0]):

            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd  # used to save sInd if returned sInd is None
            DRM, sInd, det_intTime, waitTime, det_mode = self.next_target(
                sInd, det_modes, char_modes
            )

            if sInd is not None:

                # beginning of observation, start to populate DRM
                pInds = np.where(SU.plan2star == sInd)[0]
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

                FA = False
                if sInd not in self.promoted_stars:
                    ObsNum += (
                        1  # we're making an observation so increment observation number
                    )
                    pInds = np.where(SU.plan2star == sInd)[0]
                    DRM["star_ind"] = sInd
                    DRM["star_name"] = TL.Name[sInd]
                    DRM["arrival_time"] = TK.currentTimeNorm.to("day").copy()
                    DRM["OB_nb"] = TK.OBnumber
                    DRM["ObsNum"] = ObsNum
                    DRM["plan_inds"] = pInds.astype(int)

                    # update visited list for selected star
                    self.starVisits[sInd] += 1
                    # PERFORM DETECTION and populate revisit list attribute
                    (
                        detected,
                        det_fZ,
                        det_systemParams,
                        det_SNR,
                        FA,
                    ) = self.observation_detection(sInd, det_intTime.copy(), det_mode)

                    if np.any(detected > 0):
                        self.sInd_detcounts[sInd] += 1
                        self.sInd_dettimes[sInd] = (
                            self.sInd_dettimes.get(sInd) or []
                        ) + [TK.currentTimeNorm.copy().to("day")]
                        self.vprint("  Det. results are: %s" % (detected))

                    if (
                        np.any(self.is_earthlike(pInds.astype(int), sInd))
                        and self.sInd_detcounts[sInd] >= self.n_det_min
                    ):
                        good_2_promote = False
                        if not self.promote_by_time:
                            good_2_promote = True
                        else:
                            sp = SU.s[pInds]
                            Ms = TL.MsTrue[sInd]
                            Mp = SU.Mp[pInds]
                            mu = const.G * (Mp + Ms)
                            T = (2.0 * np.pi * np.sqrt(sp**3 / mu)).to("d")
                            # star must have detections that span longer than half a
                            # period and be in the habitable zone
                            # and have a smaller radius that a sub-neptune
                            if np.any(
                                (
                                    T / 2.0
                                    < (
                                        self.sInd_dettimes[sInd][-1]
                                        - self.sInd_dettimes[sInd][0]
                                    )
                                )
                            ):
                                good_2_promote = True
                        if sInd not in self.promoted_stars and good_2_promote:
                            self.promoted_stars = np.union1d(
                                self.promoted_stars, sInd
                            ).astype(int)
                            self.known_earths = np.union1d(
                                self.known_earths,
                                pInds[self.is_earthlike(pInds.astype(int), sInd)],
                            ).astype(int)

                    # populate the DRM with detection results
                    DRM["det_time"] = det_intTime.to("day")
                    DRM["det_status"] = detected
                    DRM["det_SNR"] = det_SNR
                    DRM["det_fZ"] = det_fZ.to("1/arcsec2")
                    if np.any(pInds):
                        DRM["det_fEZ"] = SU.fEZ[pInds].to("1/arcsec2").value.tolist()
                        DRM["det_dMag"] = SU.dMag[pInds].tolist()
                        DRM["det_WA"] = SU.WA[pInds].to("mas").value.tolist()
                    DRM["det_params"] = det_systemParams
                    DRM["det_mode"] = dict(det_mode)

                    if det_intTime is not None:
                        det_comp = Comp.comp_per_intTime(
                            det_intTime,
                            TL,
                            sInd,
                            det_fZ,
                            self.ZodiacalLight.fEZ0,
                            TL.int_WA[sInd],
                            det_mode,
                        )[0]
                        DRM["det_comp"] = det_comp
                    else:
                        DRM["det_comp"] = 0.0
                    del DRM["det_mode"]["inst"], DRM["det_mode"]["syst"]
                    # append result values to self.DRM
                    self.DRM.append(DRM)
                    # handle case of inf OBs and missionPortion < 1
                    if np.isinf(TK.OBduration) and (TK.missionPortion < 1.0):
                        self.arbitrary_time_advancement(
                            TK.currentTimeNorm.to("day").copy() - DRM["arrival_time"]
                        )
                else:
                    self.char_starVisits[sInd] += 1
                    # PERFORM CHARACTERIZATION and populate spectra list attribute
                    do_char = True
                    for mode_index, char_mode in enumerate(char_modes):
                        (
                            characterized,
                            char_fZ,
                            char_systemParams,
                            char_SNR,
                            char_intTime,
                        ) = self.test_observation_characterization(
                            sInd, char_mode, mode_index
                        )
                        if char_intTime is None:
                            char_intTime = 0.0 * u.d
                        if char_intTime == 0.0 * u.d:
                            do_char = False
                            TK.advanceToAbsTime(TK.currentTimeAbs.copy() + 0.5 * u.d)

                    if do_char is True:
                        # we're making an observation so increment observation number
                        ObsNum += 1
                        pInds = np.where(SU.plan2star == sInd)[0]
                        DRM["star_ind"] = sInd
                        DRM["star_name"] = TL.Name[sInd]
                        DRM["arrival_time"] = TK.currentTimeNorm.to("day").copy()
                        DRM["OB_nb"] = TK.OBnumber
                        DRM["ObsNum"] = ObsNum
                        DRM["plan_inds"] = pInds.astype(int)
                        DRM["char_info"] = []
                        for mode_index, char_mode in enumerate(char_modes):
                            char_data = {}
                            if char_mode["SNR"] not in [0, np.inf]:
                                (
                                    characterized,
                                    char_fZ,
                                    char_systemParams,
                                    char_SNR,
                                    char_intTime,
                                ) = self.observation_characterization(
                                    sInd, char_mode, mode_index
                                )
                                if np.any(characterized):
                                    self.vprint(
                                        "  Char. results are: %s" % (characterized.T)
                                    )
                            else:
                                char_intTime = None
                                lenChar = len(pInds) + 1 if FA else len(pInds)
                                characterized = np.zeros(lenChar, dtype=float)
                                char_SNR = np.zeros(lenChar, dtype=float)
                                char_fZ = 0.0 / u.arcsec**2
                                char_systemParams = SU.dump_system_params(sInd)
                            assert char_intTime != 0, "Integration time can't be 0."

                            # populate the DRM with characterization results
                            char_data["char_time"] = (
                                char_intTime.to("day")
                                if char_intTime is not None
                                else 0.0 * u.day
                            )
                            char_data["char_status"] = (
                                characterized[:-1] if FA else characterized
                            )
                            char_data["char_SNR"] = char_SNR[:-1] if FA else char_SNR
                            char_data["char_fZ"] = char_fZ.to("1/arcsec2")
                            char_data["char_params"] = char_systemParams

                            if char_intTime is not None and np.any(characterized):
                                char_comp = Comp.comp_per_intTime(
                                    char_intTime,
                                    TL,
                                    sInd,
                                    char_fZ,
                                    self.ZodiacalLight.fEZ0,
                                    TL.int_WA[sInd],
                                    char_mode,
                                )[0]
                                DRM["char_comp"] = char_comp
                            else:
                                DRM["char_comp"] = 0.0
                            # populate the DRM with FA results
                            char_data["FA_det_status"] = int(FA)
                            char_data["FA_char_status"] = characterized[-1] if FA else 0
                            char_data["FA_char_SNR"] = char_SNR[-1] if FA else 0.0
                            char_data["FA_char_fEZ"] = (
                                self.lastDetected[sInd, 1][-1] / u.arcsec**2
                                if FA
                                else 0.0 / u.arcsec**2
                            )
                            char_data["FA_char_dMag"] = (
                                self.lastDetected[sInd, 2][-1] if FA else 0.0
                            )
                            char_data["FA_char_WA"] = (
                                self.lastDetected[sInd, 3][-1] * u.arcsec
                                if FA
                                else 0.0 * u.arcsec
                            )

                            # populate the DRM with observation modes
                            char_data["char_mode"] = dict(char_mode)
                            del (
                                char_data["char_mode"]["inst"],
                                char_data["char_mode"]["syst"],
                            )

                            char_data["exoplanetObsTime"] = TK.exoplanetObsTime.copy()
                            DRM["char_info"].append(char_data)

                        # do not revisit partial char if lucky_planets
                        if SU.lucky_planets:
                            self.char_starVisits[sInd] = self.nVisitsMax

                        # append result values to self.DRM
                        self.DRM.append(DRM)

                        # handle case of inf OBs and missionPortion < 1
                        if np.isinf(TK.OBduration) and (TK.missionPortion < 1.0):
                            self.arbitrary_time_advancement(
                                TK.currentTimeNorm.to("day").copy()
                                - DRM["arrival_time"]
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
                    _ = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
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
                        base_det_mode,
                    )[0]
                    # CASE 2 If There are no observable targets for the rest
                    # of the mission
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
                            ).format(TK.currentTimeNorm.copy())
                        )
                        # Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife
                        TK.currentTimeAbs = TK.missionFinishAbs
                    else:
                        # CASE 3 nominal wait time if at least 1 target is still in
                        # list and observable
                        # TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[
                            observableTimes.value * u.d
                            > TK.currentTimeAbs.copy().value * u.d
                        ]
                        # apply intTime filter
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1)
                        # apply revisit Filter #NOTE this means stars you added to
                        # the revisit list
                        inds3 = self.revisitFilter(
                            inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)
                        )
                        self.vprint(
                            "Filtering %d stars from advanceToAbsTime"
                            % (TL.nStars - len(inds3))
                        )
                        oTnowToEnd = observableTimes[inds3]
                        # there is at least one observableTime between now and the
                        # end of the mission
                        if not oTnowToEnd.value.shape[0] == 0:
                            tAbs = np.min(oTnowToEnd)  # advance to that observable time
                        else:
                            tAbs = (
                                TK.missionStart + TK.missionLife
                            )  # advance to end of mission
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        # Advance Time to this time OR start of next OB following
                        # this time
                        _ = TK.advanceToAbsTime(tAbs)
                        self.vprint(
                            (
                                "No Observable Targets a currentTimeNorm = {:.2f} "
                                "Advanced To currentTimeNorm= {:.2f}"
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

    def next_target(self, old_sInd, det_modes, char_modes):
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
            tuple:
                DRM (dict):
                    Design Reference Mission, contains the results of one complete
                    observation (detection and characterization)
                sInd (integer):
                    Index of next target star. Defaults to None.
                intTime (astropy Quantity):
                    Selected star integration time for detection in units of day.
                    Defaults to None.
                waitTime (astropy Quantity):
                    a strategically advantageous amount of time to wait in the case
                    of an occulter for slew times

        """
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse

        # create DRM
        DRM = {}

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + det_modes[0]["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy()
            + Obs.settlingTime
            + det_modes[0]["syst"]["ohTime"]
        )

        # create appropriate koMap
        koMap = self.koMaps[det_modes[0]["syst"]["name"]]
        char_koMap = self.koMaps[char_modes[0]["syst"]["name"]]

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        # fZs = np.zeros(TL.nStars) / u.arcsec**2.0
        # dV = np.zeros(TL.nStars) * u.m / u.s
        intTimes = np.zeros(TL.nStars) * u.d
        char_intTimes = np.zeros(TL.nStars) * u.d
        char_intTimes_no_oh = np.zeros(TL.nStars) * u.d
        # obsTimes = np.zeros([2, TL.nStars]) * u.d
        char_tovisit = np.zeros(TL.nStars, dtype=bool)
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        # sd = None

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            char_sInds = np.intersect1d(sInds, self.promoted_stars)
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # start times, including slew times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        # 2.5 Filter stars not observable at startTimes
        try:
            tmpIndsbool = list()
            for i in np.arange(len(sInds)):
                koTimeInd = np.where(
                    np.round(startTimes[sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    koMap[sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            sInds = sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722  # If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        try:
            tmpIndsbool = list()
            for i in np.arange(len(char_sInds)):
                koTimeInd = np.where(
                    np.round(startTimes[char_sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    char_koMap[char_sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            char_sInds = char_sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722 If there are no target stars to observe
            char_sInds = np.asarray([], dtype=int)

        # 3. filter out all previously (more-)visited targets, unless in
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # revisit list, with time after start
        if np.any(char_sInds):

            char_tovisit[char_sInds] = (self.char_starVisits[char_sInds] == 0) & (
                self.char_starVisits[char_sInds] < self.nVisitsMax
            )
            if self.char_starRevisit.size != 0:
                dt_rev = TK.currentTimeNorm.copy() - self.char_starRevisit[:, 1] * u.day
                ind_rev = [
                    int(x)
                    for x in self.char_starRevisit[dt_rev > 0 * u.d, 0]
                    if x in char_sInds
                ]
                char_tovisit[ind_rev] = self.char_starVisits[ind_rev] < self.nVisitsMax
            char_sInds = np.where(char_tovisit)[0]

        # 4.1 calculate integration times for ALL preselected targets
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, det_modes[0])
        maxIntTime = min(
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
            OS.intCutoff,
        )  # Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            intTimes[sInds] = self.calc_targ_intTime(
                sInds, startTimes[sInds], det_modes[0]
            ) * (1 + self.detMargin)
            sInds = sInds[
                (intTimes[sInds] <= maxIntTime)
            ]  # Filters targets exceeding end of OB
            endTimes = startTimes + intTimes

            if maxIntTime.value <= 0:
                sInds = np.asarray([], dtype=int)

        if len(char_sInds) > 0:
            for char_mode in char_modes:
                (
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                ) = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
                char_maxIntTime = min(
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                    OS.intCutoff,
                )  # Maximum intTime allowed

                char_mode_intTimes = np.zeros(TL.nStars) * u.d
                char_mode_intTimes[char_sInds] = self.calc_targ_intTime(
                    char_sInds, startTimes[char_sInds], char_mode
                ) * (1 + self.charMargin)
                char_mode_intTimes[np.isnan(char_mode_intTimes)] = 0 * u.d

                # Adjust integration time for stars with known earths around them
                for char_star in char_sInds:
                    char_earths = np.intersect1d(
                        np.where(SU.plan2star == char_star)[0], self.known_earths
                    ).astype(int)
                    if np.any(char_earths):
                        fZ = ZL.fZ(Obs, TL, char_star, startTimes[char_star], char_mode)
                        fEZ = SU.fEZ[char_earths].to("1/arcsec2").value / u.arcsec**2
                        if SU.lucky_planets:
                            phi = (1 / np.pi) * np.ones(len(SU.d))
                            dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[
                                char_earths
                            ]  # delta magnitude
                            WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to("arcsec")[
                                char_earths
                            ]  # working angle
                        else:
                            dMag = SU.dMag[char_earths]
                            WA = SU.WA[char_earths]

                        if np.all((WA < char_mode["IWA"]) | (WA > char_mode["OWA"])):
                            char_mode_intTimes[char_star] = 0.0 * u.d
                        else:
                            earthlike_inttimes = OS.calc_intTime(
                                TL, char_star, fZ, fEZ, dMag, WA, char_mode
                            ) * (1 + self.charMargin)
                            earthlike_inttimes[~np.isfinite(earthlike_inttimes)] = (
                                0 * u.d
                            )
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < char_maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                char_mode_intTimes[char_star] = np.max(
                                    earthlike_inttime
                                )
                char_intTimes_no_oh += char_mode_intTimes
                char_intTimes += char_mode_intTimes + char_mode["syst"]["ohTime"]
            char_endTimes = (
                startTimes
                + (char_intTimes * char_mode["timeMultiplier"])
                + Obs.settlingTime
            )

            char_sInds = char_sInds[
                (char_intTimes_no_oh[char_sInds] > 0.0 * u.d)
            ]  # Filters with an inttime of 0

            if char_maxIntTime.value <= 0:
                char_sInds = np.asarray([], dtype=int)

        # 5 remove char targets on ignore_stars list
        sInds = np.setdiff1d(
            sInds, np.intersect1d(sInds, self.promoted_stars).astype(int)
        )
        char_sInds = np.setdiff1d(
            char_sInds, np.intersect1d(char_sInds, self.ignore_stars)
        )

        # 6.2 Filter off coronograph stars with too many visits and no detections
        no_dets = np.logical_and(
            (self.starVisits[sInds] > self.n_det_remove),
            (self.sInd_detcounts[sInds] == 0),
        )
        sInds = sInds[np.where(np.invert(no_dets))[0]]

        max_dets = np.where(self.sInd_detcounts[sInds] < self.max_successful_dets)[0]
        sInds = sInds[max_dets]

        # 5.1 TODO Add filter to filter out stars entering and exiting keepout
        # between startTimes and endTimes

        # 5.2 find spacecraft orbital END positions (for each candidate target),
        # and filter out unavailable targets
        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            # endTimes may exist past koTimes so we have an exception to hand this case
            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(
                        np.round(endTimes[sInds[i]].value) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        koMap[sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                sInds = np.asarray([], dtype=int)

        if len(char_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            # try: # endTimes may exist past koTimes so we have an exception to
            # hand this case
            tmpIndsbool = list()
            for i in np.arange(len(char_sInds)):
                try:
                    koTimeInd = np.where(
                        np.round(char_endTimes[char_sInds[i]].value)
                        - self.koTimes.value
                        == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        char_koMap[char_sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                except:  # noqa: E722
                    tmpIndsbool.append(False)
            if np.any(tmpIndsbool):
                char_sInds = char_sInds[tmpIndsbool]
            else:
                char_sInds = np.asarray([], dtype=int)
            del tmpIndsbool

        # t_det = 0 * u.d
        det_mode = copy.deepcopy(det_modes[0])

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            if np.any(char_sInds):
                sInd, waitTime = self.choose_next_target(
                    old_sInd, char_sInds, slewTimes, char_intTimes[char_sInds]
                )
                # store selected star integration time
                intTime = char_intTimes[sInd]
            else:
                sInd, waitTime = self.choose_next_target(
                    old_sInd, sInds, slewTimes, intTimes[sInds]
                )
                # store selected star integration time
                intTime = intTimes[sInd]

            # Should Choose Next Target decide there are no stars it wishes to
            # observe at this time.
            if (sInd is None) and (waitTime is not None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe. "
                        "Waiting {}"
                    ).format(waitTime)
                )
                return DRM, None, None, waitTime, det_mode
            elif (sInd is None) and (waitTime is None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe "
                        "and waitTime is None"
                    )
                )
                return DRM, None, None, waitTime, det_mode

            # Perform dual band detections if necessary
            if (
                TL.int_WA[sInd] > det_modes[1]["IWA"]
                and TL.int_WA[sInd] < det_modes[1]["OWA"]
            ):
                det_mode["BW"] = det_mode["BW"] + det_modes[1]["BW"]
                det_mode["inst"]["sread"] = (
                    det_mode["inst"]["sread"] + det_modes[1]["inst"]["sread"]
                )
                det_mode["inst"]["idark"] = (
                    det_mode["inst"]["idark"] + det_modes[1]["inst"]["idark"]
                )
                det_mode["inst"]["CIC"] = (
                    det_mode["inst"]["CIC"] + det_modes[1]["inst"]["CIC"]
                )
                det_mode["syst"]["optics"] = np.mean(
                    (det_mode["syst"]["optics"], det_modes[1]["syst"]["optics"])
                )
                det_mode["instName"] = "combined"

            intTime = self.calc_targ_intTime(
                np.array([sInd]), startTimes[sInd], det_mode
            )[0] * (1 + self.detMargin)

            if intTime > maxIntTime and maxIntTime > 0 * u.d:
                intTime = maxIntTime

        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint(
                "No Observable Targets at currentTimeNorm= "
                + str(TK.currentTimeNorm.copy())
            )
            return DRM, None, None, None, det_mode

        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        return DRM, sInd, intTime, waitTime, det_mode

    def choose_next_target(self, old_sInd, sInds, slewTimes, t_dets):
        """Choose next telescope target based on star completeness and integration time.

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            t_dets (astropy Quantity array):
                Integration times for detection in units of day

        Returns:
            sInd (integer):
                Index of next target star

        """

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping

        # reshape sInds
        sInds = np.array(sInds, ndmin=1)

        # 1/ Choose next telescope target
        comps = Comp.completeness_update(
            TL, sInds, self.starVisits[sInds], TK.currentTimeNorm.copy()
        )

        # add weight for star revisits
        ind_rev = []
        if self.starRevisit.size != 0:
            dt_rev = self.starRevisit[:, 1] * u.day - TK.currentTimeNorm.copy()
            ind_rev = [
                int(x) for x in self.starRevisit[dt_rev < 0 * u.d, 0] if x in sInds
            ]

        f2_uv = np.where(
            (self.starVisits[sInds] > 0) & (self.starVisits[sInds] < self.nVisitsMax),
            self.starVisits[sInds],
            0,
        ) * (1 - (np.in1d(sInds, ind_rev, invert=True)))

        # f3_uv = np.where(
        #    (self.sInd_detcounts[sInds] > 0)
        #    & (self.sInd_detcounts[sInds] < self.max_successful_dets),
        #    self.sInd_detcounts[sInds],
        #    0,
        # ) * (1 - (np.in1d(sInds, ind_rev, invert=True)))

        # L = TL.L[sInds]
        l_extreme = max(
            [
                np.abs(np.log10(np.min(TL.L[sInds]))),
                np.abs(np.log10(np.max(TL.L[sInds]))),
            ]
        )
        if l_extreme == 0.0:
            l_weight = 1
        else:
            l_weight = 1 - np.abs(np.log10(TL.L[sInds]) / l_extreme) ** self.lum_exp

        t_weight = t_dets / np.max(t_dets)
        weights = (
            (comps + self.revisit_weight * f2_uv / float(self.nVisitsMax)) / t_weight
        ) * l_weight
        # weights = ((comps + self.revisit_weight*f3_uv/float(self.max_successful_dets)
        #            *f2_uv/float(self.nVisitsMax))/t_weight)*l_weight

        sInd = np.random.choice(sInds[weights == max(weights)])

        return sInd, slewTimes[sInd]

    def observation_characterization(self, sInd, mode, mode_index):
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
        fEZs = SU.fEZ[pInds].to("1/arcsec2").value
        dMags = SU.dMag[pInds]
        if SU.lucky_planets:
            # used in the "partial char" check below
            WAs = np.arctan(SU.a[pInds] / TL.dist[sInd]).to("arcsec").value
        else:
            WAs = SU.WA[pInds].to("arcsec").value

        # get the detected status, and check if there was a FA
        # det = self.lastDetected[sInd,0]
        det = np.ones(pInds.size, dtype=bool)
        FA = len(det) == len(pInds) + 1
        if FA:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if there's anything (planet or FA)
        # to characterize
        characterized = np.zeros(len(det), dtype=int)
        fZ = 0.0 / u.arcsec**2.0
        systemParams = SU.dump_system_params(
            sInd
        )  # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0:  # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime

        # look for last detected planets that have not been fully characterized
        if not (FA):  # only true planets, no FA
            tochar = self.fullSpectra[mode_index][pIndsDet] == 0
        else:  # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[mode_index][truePlans] == 0), True)

        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = (
                TK.currentTimeAbs.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )
            startTimeNorm = (
                TK.currentTimeNorm.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[
                0
            ][
                0
            ]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            koMap = self.koMaps[mode["syst"]["name"]]
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            pinds_earthlike = np.logical_and(
                np.array([(p in self.known_earths) for p in pIndsDet]), tochar
            )

            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            fEZ = fEZs[tochar] / u.arcsec**2
            WAp = TL.int_WA[sInd] * np.ones(len(tochar))
            dMag = TL.int_dMag[sInd] * np.ones(len(tochar))

            # if lucky_planets, use lucky planet params for dMag and WA
            if SU.lucky_planets:
                phi = (1 / np.pi) * np.ones(len(SU.d))
                e_dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)  # delta magnitude
                e_WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to(
                    "arcsec"
                )  # working angle
            else:
                e_dMag = SU.dMag
                e_WA = SU.WA

            WAp[((pinds_earthlike) & (tochar))] = e_WA[pIndsDet[pinds_earthlike]]
            dMag[((pinds_earthlike) & (tochar))] = e_dMag[pIndsDet[pinds_earthlike]]

            intTimes = np.zeros(len(tochar)) * u.day
            intTimes[tochar] = OS.calc_intTime(
                TL, sInd, fZ, fEZ, dMag[tochar], WAp[tochar], mode
            )
            intTimes[~np.isfinite(intTimes)] = 0 * u.d

            # add a predetermined margin to the integration times
            intTimes = intTimes * (1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes * (mode["timeMultiplier"])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = (
                (totTimes > 0)
                & (totTimes <= OS.intCutoff)
                & (endTimesNorm <= TK.OBendTimes[TK.OBnumber])
            )

        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            koTimeInds = np.zeros(len(endTimes.value[tochar]), dtype=int)

            # find index in koMap where each endTime is closest to koTimes
            for t, endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(
                        np.floor(endTime) - self.koTimes.value == 0
                    )[0]
                    koTimeInds[t] = (
                        endTimeInBounds[0] if endTimeInBounds.size != 0 else -1
                    )
                else:
                    koTimeInds[t] = np.where(
                        np.round(endTime) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]

        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            # Save Current Time before attempting time allocation
            currentTimeNorm = TK.currentTimeNorm.copy()
            currentTimeAbs = TK.currentTimeAbs.copy()

            if np.any(np.logical_and(pinds_earthlike, tochar)):
                intTime = np.max(intTimes[np.logical_and(pinds_earthlike, tochar)])
            else:
                intTime = np.max(intTimes[tochar])
            extraTime = intTime * (mode["timeMultiplier"] - 1.0)  # calculates extraTime
            success = TK.allocate_time(
                intTime + extraTime + mode["syst"]["ohTime"] + Obs.settlingTime, True
            )  # allocates time
            if not (success):  # Time was not successfully allocated
                char_intTime = None
                lenChar = len(pInds) + 1 if FA else len(pInds)
                characterized = np.zeros(lenChar, dtype=int)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0.0 / u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)

                # finally, populate the revisit list (NOTE: sInd becomes a float)
                t_rev = TK.currentTimeNorm.copy() + self.revisit_wait[sInd]
                revisit = np.array([sInd, t_rev.to("day").value])
                if self.char_starRevisit.size == 0:
                    self.char_starRevisit = np.array([revisit])
                else:
                    revInd = np.where(self.char_starRevisit[:, 0] == sInd)[0]
                    if revInd.size == 0:
                        self.char_starRevisit = np.vstack(
                            (self.char_starRevisit, revisit)
                        )
                    else:
                        self.char_starRevisit[revInd, 1] = revisit[1]
                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

            pIndsChar = pIndsDet[tochar]
            log_char = "   - Charact. planet(s) %s (%s/%s detected)" % (
                pIndsChar,
                len(pIndsChar),
                len(pIndsDet),
            )
            self.logger.info(log_char)
            self.vprint(log_char)

            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux) / u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype="object")
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime / float(self.ntFlux)
                timePlus = (
                    Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
                )  # accounts for the time since the current time
                for i in range(self.ntFlux):
                    # calculate signal and noise (electron count rates)
                    if SU.lucky_planets:
                        fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs, mode)[0]
                        Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                            sInd, planinds, dt, mode, fZ=fZs[i]
                        )
                    # allocate first half of dt
                    timePlus += dt / 2.0
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(
                        sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                    )
                    self.propagTimes[sInd] = currentTimeNorm + timePlus
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    if not SU.lucky_planets:
                        Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                            sInd, planinds, dt, mode, fZ=fZs[i]
                        )
                    # allocate second half of dt
                    timePlus += dt / 2.0

                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {
                    key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                    / float(self.ntFlux)
                    for key in sorted(systemParamss[0])
                }
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0] / N[N > 0]
                # allocate extra time for timeMultiplier
            # if only a FA, just save zodiacal brightness in the middle of the
            # integration
            else:
                # totTime = intTime * (mode["timeMultiplier"])
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy(), mode)[0]

            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = fEZs[-1] / u.arcsec**2
                dMag = dMags[-1]
                WA = WAs[-1] * u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p * intTime).decompose().value
                N = np.sqrt((C_b * intTime + (C_sp * intTime) ** 2).decompose().value)
                SNRfa = S / N if N > 0 else 0.0

            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)

            # now, store characterization status: 1 for full spectrum,
            # -1 for partial spectrum, 0 for not characterized
            char = SNR >= mode["SNR"]
            # initialize with full spectra
            characterized = char.astype(int)
            WAchar = WAs[char] * u.arcsec
            # find the current WAs of characterized planets
            if SU.lucky_planets:
                # keep original WAs (note, the dump_system_params() above, whence comes
                # systemParams, does not understand lucky_planets)
                pass
            else:
                WAs = systemParams["WA"]
            if FA:
                WAs = np.append(WAs, WAs[-1] * u.arcsec)
            # check for partial spectra
            IWA_max = mode["IWA"] * (1.0 + mode["BW"] / 2.0)
            OWA_min = mode["OWA"] * (1.0 - mode["BW"] / 2.0)
            char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            all_full = np.copy(characterized)
            all_full[char] = 0
            if sInd not in self.sInd_charcounts.keys():
                self.sInd_charcounts[sInd] = all_full
            else:
                self.sInd_charcounts[sInd] = self.sInd_charcounts[sInd] + all_full

            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[mode_index][pInds[charplans == 1]] += 1
            self.partialSpectra[mode_index][pInds[charplans == -1]] += 1

        # in both cases (detection or false alarm), schedule a revisit
        smin = np.min(SU.s[pInds[det]])
        Ms = TL.MsTrue[sInd]

        # if target in promoted_stars list, schedule revisit based off of
        # semi-major axis
        if sInd in self.promoted_stars:
            sp = np.min(SU.a[pInds[det]]).to("AU")
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.a[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G * (Mp + Ms)
            T = 2.0 * np.pi * np.sqrt(sp**3 / mu)
            t_rev = TK.currentTimeNorm.copy() + T / 3.0
        # otherwise schedule revisit based off of seperation
        elif smin is not None:
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G * (Mp + Ms)
            T = 2.0 * np.pi * np.sqrt(sp**3 / mu)
            t_rev = TK.currentTimeNorm.copy() + T / 2.0
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G * (Mp + Ms)
            T = 2.0 * np.pi * np.sqrt(sp**3 / mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75 * T

        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to("day").value])
        if self.char_starRevisit.size == 0:
            self.char_starRevisit = np.array([revisit])
        else:
            revInd = np.where(self.char_starRevisit[:, 0] == sInd)[0]
            if revInd.size == 0:
                self.char_starRevisit = np.vstack((self.char_starRevisit, revisit))
            else:
                self.char_starRevisit[revInd, 1] = revisit[1]

        # add stars to filter list
        if np.any(characterized.astype(int) == 1):
            if np.any(self.sInd_charcounts[sInd] >= self.max_successful_chars):
                self.ignore_stars = np.union1d(self.ignore_stars, [sInd]).astype(int)

        return characterized.astype(int), fZ, systemParams, SNR, intTime

    def test_observation_characterization(self, sInd, mode, mode_index):
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
        fEZs = SU.fEZ[pInds].to("1/arcsec2").value
        dMags = SU.dMag[pInds]
        # WAs = SU.WA[pInds].to("arcsec").value

        # get the detected status, and check if there was a FA
        # det = self.lastDetected[sInd,0]
        det = np.ones(pInds.size, dtype=bool)
        FA = len(det) == len(pInds) + 1
        if FA:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if there's anything (planet or FA)
        # to characterize
        characterized = np.zeros(len(det), dtype=int)
        fZ = 0.0 / u.arcsec**2.0
        systemParams = SU.dump_system_params(
            sInd
        )  # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0:  # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime

        # look for last detected planets that have not been fully characterized
        if not (FA):  # only true planets, no FA
            tochar = self.fullSpectra[mode_index][pIndsDet] == 0
        else:  # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[mode_index][truePlans] == 0), True)

        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = (
                TK.currentTimeAbs.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )
            startTimeNorm = (
                TK.currentTimeNorm.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
            )
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[
                0
            ][
                0
            ]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            koMap = self.koMaps[mode["syst"]["name"]]
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            pinds_earthlike = np.logical_and(
                np.array([(p in self.known_earths) for p in pIndsDet]), tochar
            )

            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            fEZ = fEZs[tochar] / u.arcsec**2
            dMag = dMags[tochar]
            WAp = TL.int_WA[sInd] * np.ones(len(tochar))
            dMag = TL.int_dMag[sInd] * np.ones(len(tochar))

            # if lucky_planets, use lucky planet params for dMag and WA
            if SU.lucky_planets:
                phi = (1 / np.pi) * np.ones(len(SU.d))
                e_dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)  # delta magnitude
                e_WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to(
                    "arcsec"
                )  # working angle
            else:
                e_dMag = SU.dMag
                e_WA = SU.WA

            WAp[((pinds_earthlike) & (tochar))] = e_WA[pIndsDet[pinds_earthlike]]
            dMag[((pinds_earthlike) & (tochar))] = e_dMag[pIndsDet[pinds_earthlike]]

            intTimes = np.zeros(len(tochar)) * u.day
            intTimes[tochar] = OS.calc_intTime(
                TL, sInd, fZ, fEZ, dMag[tochar], WAp[tochar], mode
            )
            intTimes[~np.isfinite(intTimes)] = 0 * u.d

            # add a predetermined margin to the integration times
            intTimes = intTimes * (1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes * (mode["timeMultiplier"])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = (
                (totTimes > 0)
                & (totTimes <= OS.intCutoff)
                & (endTimesNorm <= TK.OBendTimes[TK.OBnumber])
            )

        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            koTimeInds = np.zeros(len(endTimes.value[tochar]), dtype=int)

            # find index in koMap where each endTime is closest to koTimes
            for t, endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(
                        np.floor(endTime) - self.koTimes.value == 0
                    )[0]
                    koTimeInds[t] = (
                        endTimeInBounds[0] if endTimeInBounds.size != 0 else -1
                    )
                else:
                    koTimeInds[t] = np.where(
                        np.round(endTime) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]

        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            if np.any(np.logical_and(pinds_earthlike, tochar)):
                intTime = np.max(intTimes[np.logical_and(pinds_earthlike, tochar)])
            else:
                intTime = np.max(intTimes[tochar])
            extraTime = intTime * (mode["timeMultiplier"] - 1.0)  # calculates extraTime

            dt = intTime + extraTime + mode["syst"]["ohTime"] + Obs.settlingTime
            if (
                (dt.value <= 0 or dt.value == np.inf)
                or (TK.currentTimeNorm.copy() + dt > TK.missionLife.to("day"))
                or (TK.currentTimeNorm.copy() + dt > TK.OBendTimes[TK.OBnumber])
            ):
                success = (
                    False  # The temporal block to allocate is not positive nonzero
                )
            else:
                success = True

            # success = TK.allocate_time(intTime + extraTime + mode['syst']['ohTime']
            #                               + Obs.settlingTime, True)#allocates time
            if not (success):  # Time was not successfully allocated
                char_intTime = None
                lenChar = len(pInds) + 1 if FA else len(pInds)
                characterized = np.zeros(lenChar, dtype=float)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0.0 / u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)

                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

            # pIndsChar = pIndsDet[tochar]

        return characterized.astype(int), fZ, systemParams, SNR, intTime

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

        t_rev = TK.currentTimeNorm.copy() + self.revisit_wait[sInd]
        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to("day").value])
        if self.starRevisit.size == 0:  # If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])  # initialize sterRevisit
        else:
            revInd = np.where(self.starRevisit[:, 0] == sInd)[
                0
            ]  # indices of the first column of the starRevisit list containing sInd
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd, 1] = revisit[1]  # over

    def revisitFilter(self, sInds, tmpCurrentTimeNorm):
        """Helper method for Overloading Revisit Filtering

        Args:
            sInds - indices of stars still in observation list
            tmpCurrentTimeNorm (MJD) - the simulation time after overhead was
            added in MJD form

        Returns:
            ~numpy.ndarray(int):
                sInds - indices of stars still in observation list
        """
        tovisit = np.zeros(
            self.TargetList.nStars, dtype=bool
        )  # tovisit is a boolean array containing the
        if len(sInds) > 0:  # so long as there is at least 1 star left in sInds
            tovisit[sInds] = (self.starVisits[sInds] == min(self.starVisits[sInds])) & (
                self.starVisits[sInds] < self.nVisitsMax
            )  # Checks that no star has exceeded the number of revisits
            if (
                self.starRevisit.size != 0
            ):  # There is at least one revisit planned in starRevisit
                dt_rev = (
                    self.starRevisit[:, 1] * u.day - tmpCurrentTimeNorm
                )  # absolute temporal spacing between revisit and now.

                # return indices of all revisits within a threshold dt_max of
                # revisit day and indices of all revisits with no detections
                # past the revisit time
                ind_rev2 = [
                    int(x)
                    for x in self.starRevisit[dt_rev < 0 * u.d, 0]
                    if (x in sInds)
                ]
                tovisit[ind_rev2] = self.starVisits[ind_rev2] < self.nVisitsMax
            sInds = np.where(tovisit)[0]

        return sInds
