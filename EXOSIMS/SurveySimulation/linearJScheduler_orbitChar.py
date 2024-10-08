from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import astropy.constants as const
import time
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util._numpy_compat import copy_if_needed


class linearJScheduler_orbitChar(SurveySimulation):
    """linearJScheduler_orbitChar

    This class implements a varient of the linear cost function scheduler described
    in Savransky et al. (2010).

    It inherits directly from the protoype SurveySimulation class.

    The LJS_orbitChar scheduler performs scheduled starshade visits to both detect
    and characterize targets. Once a target is detected, it will be subsequently
    characterized. If the characterization is successful, that taget will be marked
    for further detections to characeterize it's orbit.

    Args:
        coeffs (iterable 6x1):
            Cost function coefficients: slew distance, completeness, least visited known
            RV planet ramp, unvisited known RV planet ramp, least visited ramp,
            unvisited ramp
        revisit_wait (float):
            Wait time threshold for star revisits. The value given is the fraction of a
            characterized planet's period that must be waited before scheduling a
            revisit.
        n_det_remove (int):
            Number of failed detections before a star is removed from the target list.
        n_det_min (int):
            Minimum number of detections required for promotion to char target.
        max_successful_dets (int):
            Maximum number of successful detections before star is taken off target
            list.
        max_successful_chars (int):
            Maximum number of successful characterizations on a given star before
            it is removed from the target list.
        det_only (bool):
            Run the sim only performing detections and no chars.
        char_only (bool:
            Run the sim performing only chars, particularly for precursor RV using
            known_rocky.
        specs (dict):
            :ref:`sec:inputspec`

    """

    def __init__(
        self,
        coeffs=[1, 1, 1, 1, 2, 1],
        revisit_wait=0.5,
        n_det_remove=3,
        n_det_min=3,
        max_successful_dets=4,
        max_successful_chars=1,
        det_only=False,
        char_only=False,
        **specs
    ):

        SurveySimulation.__init__(self, **specs)
        TL = self.TargetList
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse

        # verify that coefficients input is iterable 6x1
        if not (isinstance(coeffs, (list, tuple, np.ndarray))) or (len(coeffs) != 6):
            raise TypeError("coeffs must be a 6 element iterable")

        # Add to outspec
        self._outspec["coeffs"] = coeffs
        self._outspec["revisit_wait"] = revisit_wait

        # normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs / np.linalg.norm(coeffs, ord=1)

        self.coeffs = coeffs

        EEID = 1 * u.AU * np.sqrt(TL.L)
        mu = const.G * (TL.MsTrue)
        T = (2.0 * np.pi * np.sqrt(EEID**3 / mu)).to("d")
        self.revisit_wait = revisit_wait * T

        self.sInd_detcounts = np.zeros(
            TL.nStars, dtype=int
        )  # Number of detections by star index
        self.sInd_charcounts = np.zeros(
            TL.nStars, dtype=int
        )  # Number of spectral characterizations by star index
        self.sInd_dettimes = {}
        self.det_prefer = []  # list of star indicies to be given detection preference
        self.ignore_stars = []  # list of stars that have already been chard
        self.no_dets = np.ones(self.TargetList.nStars, dtype=bool)
        self.promoted_stars = []  # actually just a list of characterized stars
        self.promotable_stars = self.known_rocky

        # Minimum number of visits with no detections required to filter off star
        self.n_det_remove = n_det_remove
        # Minimum number of detections required for promotion
        self.n_det_min = n_det_min
        self.max_successful_dets = max_successful_dets
        # max number of characterizations allowed before retiring target
        self.max_successful_chars = max_successful_chars
        self.det_only = det_only
        self.char_only = char_only

        occ_sInds_with_earths = []
        if TL.earths_only:
            char_mode = list(  # noqa: F841
                filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
            )[0]

            # check for earths around the available stars
            for sInd in np.arange(TL.nStars):
                pInds = np.where(SU.plan2star == sInd)[0]
                pinds_earthlike = self.is_earthlike(pInds, sInd)
                if np.any(pinds_earthlike):
                    self.known_earths = np.union1d(
                        self.known_earths, pInds[pinds_earthlike]
                    ).astype(int)
                    occ_sInds_with_earths.append(sInd)
            self.promotable_stars = np.union1d(
                self.promotable_stars, occ_sInds_with_earths
            ).astype(int)

        if self.find_known_RV or TL.earths_only:
            TL.int_comp[self.promotable_stars] = 1.0

    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        Comp = self.Completeness

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
            DRM, sInd, det_intTime, waitTime = self.next_target(
                sInd, det_mode, char_mode
            )
            # pdb.set_trace() ###Rhonda debug
            if sInd is not None:
                ObsNum += (
                    1  # we're making an observation so increment observation number
                )

                if OS.haveOcculter:
                    # advance to start of observation
                    # (add slew time for selected target)
                    _ = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)

                # beginning of observation, start to populate DRM
                DRM["star_ind"] = sInd
                DRM["star_name"] = TL.Name[sInd]
                DRM["arrival_time"] = TK.currentTimeNorm.to("day").copy()
                DRM["OB_nb"] = TK.OBnumber
                DRM["ObsNum"] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0].astype(int)
                DRM["plan_inds"] = pInds
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

                detected = np.array([])
                detection = False
                FA = False

                if not self.char_only:
                    # if sInd not promoted of (char'able and char'd)
                    if sInd not in self.promotable_stars or (
                        sInd in self.promotable_stars and sInd in self.promoted_stars
                    ):
                        # PERFORM DETECTION and populate revisit list attribute
                        (
                            detected,
                            det_fZ,
                            det_systemParams,
                            det_SNR,
                            FA,
                        ) = self.observation_detection(
                            sInd, det_intTime.copy(), det_mode
                        )

                        if 1 in detected:
                            detection = True
                            self.sInd_detcounts[sInd] += 1
                            self.sInd_dettimes[sInd] = (
                                self.sInd_dettimes.get(sInd) or []
                            ) + [TK.currentTimeNorm.copy().to("day")]
                            self.vprint("  Det. results are: %s" % (detected))

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
                        if det_intTime is not None:
                            det_comp = Comp.comp_per_intTime(
                                det_intTime,
                                TL,
                                sInd,
                                det_fZ,
                                self.ZodiacalLight.fEZ0,
                                self.int_WA[sInd],
                                det_mode,
                            )[0]
                            DRM["det_comp"] = det_comp
                        else:
                            DRM["det_comp"] = 0.0
                        if np.any(pInds):
                            DRM["det_fEZ"] = (
                                SU.fEZ[pInds].to("1/arcsec2").value.tolist()
                            )
                            DRM["det_dMag"] = SU.dMag[pInds].tolist()
                            DRM["det_WA"] = SU.WA[pInds].to("mas").value.tolist()
                        DRM["det_params"] = det_systemParams
                    # populate the DRM with observation modes
                    DRM["det_mode"] = dict(det_mode)  # moved to det_observation section
                    del DRM["det_mode"]["inst"], DRM["det_mode"]["syst"]

                if not self.det_only:
                    if (detection and sInd not in self.ignore_stars) or (
                        sInd in self.promotable_stars and sInd not in self.ignore_stars
                    ):
                        # PERFORM CHARACTERIZATION and populate spectra list attribute
                        TL.int_comp[sInd] = 1.0
                        do_char = True

                        if sInd not in self.promotable_stars:
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
                            startTime = TK.currentTimeAbs.copy()
                            pred_char_intTime = self.calc_targ_intTime(
                                np.array([sInd]), startTime, char_mode
                            )

                            # Adjust integration time for stars
                            # with known earths around them
                            fZ = ZL.fZ(Obs, TL, sInd, startTime, char_mode)
                            fEZ = SU.fEZ[pInds].to("1/arcsec2").value / u.arcsec**2

                            if SU.lucky_planets:
                                phi = (1 / np.pi) * np.ones(len(SU.d))
                                dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[
                                    pInds
                                ]  # delta magnitude
                                WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to(
                                    "arcsec"
                                )[
                                    pInds
                                ]  # working angle
                            else:
                                dMag = SU.dMag[pInds]
                                WA = SU.WA[pInds]
                            # dMag = SU.dMag[pInds]
                            # WA = SU.WA[pInds]
                            earthlike_inttimes = OS.calc_intTime(
                                TL, sInd, fZ, fEZ, dMag, WA, char_mode
                            ) * (1 + self.charMargin)
                            earthlike_inttimes[~np.isfinite(earthlike_inttimes)] = (
                                0 * u.d
                            )
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < char_maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                pred_char_intTime = np.max(earthlike_inttime)
                            else:
                                pred_char_intTime = np.max(earthlike_inttimes)
                            if not pred_char_intTime <= char_maxIntTime:
                                do_char = False

                        if do_char:
                            if char_mode["SNR"] not in [0, np.inf]:
                                (
                                    characterized,
                                    char_fZ,
                                    char_systemParams,
                                    char_SNR,
                                    char_intTime,
                                ) = self.observation_characterization(sInd, char_mode)
                                if np.any(characterized):
                                    self.promoted_stars.append(sInd)
                                    self.vprint(
                                        "  Char. results are: %s" % (characterized)
                                    )
                                if np.any(
                                    np.logical_and(
                                        self.is_earthlike(pInds, sInd),
                                        (characterized == 1),
                                    )
                                ):
                                    self.known_earths = np.union1d(
                                        self.known_earths,
                                        pInds[self.is_earthlike(pInds, sInd)],
                                    ).astype(int)
                                    if sInd not in self.det_prefer:
                                        self.det_prefer.append(sInd)
                                    if sInd not in self.ignore_stars:
                                        self.ignore_stars.append(sInd)
                                if 1 in characterized:
                                    self.sInd_charcounts[sInd] += 1

                            else:
                                char_intTime = None
                                lenChar = len(pInds) + 1 if FA else len(pInds)
                                characterized = np.zeros(lenChar, dtype=float)
                                char_SNR = np.zeros(lenChar, dtype=float)
                                char_fZ = 0.0 / u.arcsec**2
                                char_systemParams = SU.dump_system_params(sInd)
                            assert char_intTime != 0, "Integration time can't be 0."
                            # update the occulter wet mass
                            if OS.haveOcculter and char_intTime is not None:
                                DRM = self.update_occulter_mass(
                                    DRM, sInd, char_intTime, "char"
                                )
                            # populate the DRM with characterization results
                            DRM["char_time"] = (
                                char_intTime.to("day")
                                if char_intTime is not None
                                else 0.0 * u.day
                            )
                            DRM["char_status"] = (
                                characterized[:-1] if FA else characterized
                            )
                            DRM["char_SNR"] = char_SNR[:-1] if FA else char_SNR
                            DRM["char_fZ"] = char_fZ.to("1/arcsec2")
                            if char_intTime is not None:
                                char_comp = Comp.comp_per_intTime(
                                    char_intTime,
                                    TL,
                                    sInd,
                                    char_fZ,
                                    self.ZodiacalLight.fEZ0,
                                    self.int_WA[sInd],
                                    char_mode,
                                )[0]
                                DRM["char_comp"] = char_comp
                            else:
                                DRM["char_comp"] = 0.0
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
                            DRM["FA_char_dMag"] = (
                                self.lastDetected[sInd, 2][-1] if FA else 0.0
                            )
                            DRM["FA_char_WA"] = (
                                self.lastDetected[sInd, 3][-1] * u.arcsec
                                if FA
                                else 0.0 * u.arcsec
                            )

                            DRM["char_mode"] = dict(char_mode)
                            del DRM["char_mode"]["inst"], DRM["char_mode"]["syst"]

                # populate the DRM with observation modes
                # DRM['det_mode'] = dict(det_mode) #moved to det_observation section
                # del DRM['det_mode']['inst'], DRM['det_mode']['syst']

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
                        det_mode,
                    )[0]
                    # CASE 2 If There are no observable targets for the
                    # rest of the mission
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
                    else:
                        # CASE 3 nominal wait time if at least 1 target is still
                        # in list and observable
                        # TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[
                            observableTimes.value * u.d
                            > TK.currentTimeAbs.copy().value * u.d
                        ]
                        # apply intTime filter
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1)
                        # apply revisit Filter
                        # NOTE this means stars you added to the revisit list
                        inds3 = self.revisitFilter(
                            inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)
                        )
                        self.vprint(
                            "Filtering %d stars from advanceToAbsTime"
                            % (TL.nStars - len(inds3))
                        )
                        oTnowToEnd = observableTimes[inds3]
                        # there is at least one observableTime between now and
                        # the end of the mission
                        if not oTnowToEnd.value.shape[0] == 0:
                            # advance to that observable time
                            tAbs = np.min(oTnowToEnd)
                        else:
                            # advance to end of mission
                            tAbs = TK.missionStart + TK.missionLife
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        # Advance Time to this time
                        # OR start of next OB following this time
                        _ = TK.advanceToAbsTime(tAbs)
                        self.vprint(
                            (
                                "No Observable Targets a currentTimeNorm= {:.2f} "
                                "Advanced To currentTimeNorm = {:.2f}"
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

    def next_target(self, old_sInd, mode, char_mode):
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
                    a strategically advantageous amount of time to wait in the case of
                    an occulter for slew times

        """
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse

        # create DRM
        DRM = {}

        # create appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]
        char_koMap = self.koMaps[char_mode["syst"]["name"]]

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = TK.currentTimeAbs.copy()
        tmpCurrentTimeNorm = TK.currentTimeNorm.copy()

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        # fZs = np.zeros(TL.nStars) / u.arcsec**2
        dV = np.zeros(TL.nStars) * u.m / u.s
        intTimes = np.zeros(TL.nStars) * u.d
        char_intTimes = np.zeros(TL.nStars) * u.d
        obsTimes = np.zeros([2, TL.nStars]) * u.d
        sInds = np.arange(TL.nStars)
        detectable_sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        sd = None
        if OS.haveOcculter:
            sd = Obs.star_angularSep(TL, old_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs
            )

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
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
        except:  # noqa: E722 If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        # 2.7 Filter off all non-earthlike-planet-having stars
        if TL.earths_only or self.char_only:
            sInds = np.intersect1d(sInds, self.promotable_stars)

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
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
            OS.intCutoff,
        )  # Maximum intTime allowed

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

        if len(sInds.tolist()) > 0:
            intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)

            # Adjust integration time for stars with known earths around them
            for star in sInds:
                if star in self.promotable_stars:
                    earths = np.intersect1d(
                        np.where(SU.plan2star == star)[0], self.known_earths
                    ).astype(int)
                    if np.any(earths):
                        fZ = ZL.fZ(Obs, TL, star, startTimes[star], mode)
                        fEZ = SU.fEZ[earths].to("1/arcsec2").value / u.arcsec**2
                        if SU.lucky_planets:
                            phi = (1 / np.pi) * np.ones(len(SU.d))
                            dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[
                                earths
                            ]  # delta magnitude
                            WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to("arcsec")[
                                earths
                            ]  # working angle
                        else:
                            dMag = SU.dMag[earths]
                            WA = SU.WA[earths]

                        if np.all((WA < mode["IWA"]) | (WA > mode["OWA"])):
                            intTimes[star] = 0.0 * u.d
                        else:
                            earthlike_inttimes = OS.calc_intTime(
                                TL, star, fZ, fEZ, dMag, WA, mode
                            )
                            earthlike_inttimes[~np.isfinite(earthlike_inttimes)] = (
                                0 * u.d
                            )
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                intTimes[star] = np.max(earthlike_inttime)
                            else:
                                intTimes[star] = np.max(earthlike_inttimes)
            endTimes = (
                startTimes
                + (intTimes * mode["timeMultiplier"])
                + Obs.settlingTime
                + mode["syst"]["ohTime"]
            )

            sInds = sInds[
                (intTimes[sInds] <= maxIntTime)
            ]  # Filters targets exceeding maximum intTime
            sInds = sInds[(intTimes[sInds] > 0.0 * u.d)]  # Filters with an inttime of 0
            detectable_sInds = sInds  # Filters targets exceeding maximum intTime

            if maxIntTime.value <= 0:
                sInds = np.asarray([], dtype=int)

        if len(sInds.tolist()) > 0:
            # calculate characterization starttimes
            temp_intTimes = intTimes.copy()
            for sInd in sInds:
                if sInd in self.promotable_stars:
                    temp_intTimes[sInd] = 0 * u.d
                else:
                    temp_intTimes[sInd] = (
                        intTimes[sInd].copy()
                        + (intTimes[sInd] * (mode["timeMultiplier"] - 1.0))
                        + Obs.settlingTime
                        + mode["syst"]["ohTime"]
                    )
            char_startTimes = startTimes + temp_intTimes

            # characterization_start = char_startTimes
            char_intTimes[sInds] = self.calc_targ_intTime(
                sInds, char_startTimes[sInds], char_mode
            ) * (1 + self.charMargin)

            # Adjust integration time for stars with known earths around them
            for star in sInds:
                if star in self.promotable_stars:
                    char_earths = np.intersect1d(
                        np.where(SU.plan2star == star)[0], self.known_earths
                    ).astype(int)
                    if np.any(char_earths):
                        fZ = ZL.fZ(Obs, TL, star, char_startTimes[star], char_mode)
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
                            char_intTimes[star] = 0.0 * u.d
                        else:
                            earthlike_inttimes = OS.calc_intTime(
                                TL, star, fZ, fEZ, dMag, WA, char_mode
                            ) * (1 + self.charMargin)
                            earthlike_inttimes[~np.isfinite(earthlike_inttimes)] = (
                                0 * u.d
                            )
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < char_maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                char_intTimes[star] = np.max(earthlike_inttime)
                            else:
                                char_intTimes[star] = np.max(earthlike_inttimes)
            char_endTimes = (
                char_startTimes
                + (char_intTimes * char_mode["timeMultiplier"])
                + Obs.settlingTime
                + char_mode["syst"]["ohTime"]
            )

            sInds = sInds[
                (char_intTimes[sInds] <= char_maxIntTime)
            ]  # Filters targets exceeding maximum intTime
            sInds = sInds[
                (char_intTimes[sInds] > 0.0 * u.d)
            ]  # Filters with an inttime of 0

            if char_maxIntTime.value <= 0:
                sInds = np.asarray([], dtype=int)

        # 5.1 TODO Add filter to filter out stars entering and exiting keepout
        # between startTimes and endTimes
        try:
            tmpIndsbool = list()
            for i in np.arange(len(sInds)):
                koTimeInd = np.where(
                    np.round(char_startTimes[sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    char_koMap[sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            sInds = sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722 If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        # 5.2 find spacecraft orbital END positions (for each candidate target),
        # and filter out unavailable targets
        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    # find indices where koTime is endTime[0]
                    koTimeInd = np.where(
                        np.round(endTimes[sInds[i]].value) - self.koTimes.value == 0
                    )[0][0]
                    # Is star observable at time ind
                    tmpIndsbool.append(koMap[sInds[i]][koTimeInd].astype(bool))
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                sInds = np.asarray([], dtype=int)

        if len(detectable_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            try:
                tmpIndsbool = list()
                for i in np.arange(len(detectable_sInds)):
                    koTimeInd = np.where(
                        np.round(endTimes[detectable_sInds[i]].value)
                        - self.koTimes.value
                        == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        koMap[detectable_sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                detectable_sInds = detectable_sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                detectable_sInds = np.asarray([], dtype=int)

        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(
                        np.round(char_endTimes[sInds[i]].value) - self.koTimes.value
                        == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        char_koMap[sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                sInds = np.asarray([], dtype=int)

        # 6.2 Filter off coronograph stars with too many visits and no detections
        no_dets = np.logical_and(
            (self.sInd_charcounts[sInds] >= self.max_successful_chars),
            (self.sInd_charcounts[sInds] == 0),
        )
        sInds = sInds[np.where(np.invert(no_dets))[0]]

        # using starVisits here allows multiple charcounts
        # to count towards orbit determination detections
        no_dets = np.logical_and(
            (self.starVisits[detectable_sInds] >= self.n_det_remove),
            (self.sInd_detcounts[detectable_sInds] == 0),
        )
        detectable_sInds = detectable_sInds[np.where(np.invert(no_dets))[0]]

        # find stars that are available for detection revisits
        detectable_sInds_tmp = []
        for dsInd in detectable_sInds:
            # if dsInd not awaiting characterization or
            # (is char'able and already char'd)
            if dsInd not in self.promotable_stars or (
                dsInd in self.promotable_stars and dsInd in self.promoted_stars
            ):
                detectable_sInds_tmp.append(dsInd)
        detectable_sInds = np.array(detectable_sInds_tmp)

        if not np.any(sInds) and np.any(detectable_sInds):
            if not self.char_only:
                sInds = detectable_sInds
            # implied else is sInds = []

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            sInd, waitTime = self.choose_next_target(
                old_sInd, sInds, slewTimes, intTimes[sInds]
            )

            # Should Choose Next Target decide there are no stars it wishes
            # to observe at this time
            if (sInd is None) and (waitTime is not None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe. "
                        "Waiting {}"
                    ).format(waitTime)
                )
                return DRM, None, None, waitTime
            elif (sInd is None) and (waitTime is None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe "
                        "and waitTime is None"
                    )
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

        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        # populate DRM with occulter related values
        if OS.haveOcculter:
            DRM = Obs.log_occulterResults(
                DRM, slewTimes[sInd], sInd, sd[sInd], dV[sInd]
            )
            return DRM, sInd, intTime, slewTimes[sInd]

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
            intTimes (astropy.units.Quantity array):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (int):
                    Index of next target star
                waitTime (astropy.units.Quantity):
                    the amount of time to wait (this method returns None)

        """

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        known_sInds = np.intersect1d(sInds, self.promotable_stars)

        # current star has to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds, old_sInd)

        # calculate dt since previous observation
        dt = TK.currentTimeNorm.copy() + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        for idx, sInd in enumerate(sInds):
            if sInd in known_sInds or sInd in self.det_prefer:
                comps[idx] = 1.0

        # if first target, or if only 1 available target,
        # choose highest available completeness
        nStars = len(sInds)
        if (old_sInd is None) or (nStars == 1):
            sInd = np.random.choice(sInds[comps == max(comps)])
            return sInd, slewTimes[sInd]

        # define adjacency matrix
        A = np.zeros((nStars, nStars))

        # 0/ only consider slew distance when there's an occulter
        if OS.haveOcculter:
            r_ts = TL.starprop(sInds, TK.currentTimeAbs.copy())
            u_ts = (
                r_ts.to("AU").value.T / np.linalg.norm(r_ts.to("AU").value, axis=1)
            ).T
            angdists = np.arccos(np.clip(np.dot(u_ts, u_ts.T), -1, 1))
            A[np.ones((nStars), dtype=bool)] = angdists
            A = self.coeffs[0] * (A) / np.pi

        # 1/ add factor due to completeness
        A = A + self.coeffs[1] * (1 - comps)

        # add factor for unvisited ramp for known stars
        if np.any(known_sInds):
            # 2/ add factor for least visited known stars
            f_uv = np.zeros(nStars)
            u1 = np.in1d(sInds, known_sInds)
            u2 = self.starVisits[sInds] == min(self.starVisits[known_sInds])
            unvisited = np.logical_and(u1, u2)
            f_uv[unvisited] = (
                float(TK.currentTimeNorm.copy() / TK.missionLife.copy()) ** 2
            )
            A = A - self.coeffs[2] * f_uv

            # 3/ add factor for unvisited known stars
            no_visits = np.zeros(nStars)
            u2 = self.starVisits[sInds] == 0
            unvisited = np.logical_and(u1, u2)
            no_visits[unvisited] = 1.0
            A = A - self.coeffs[3] * no_visits

        # 4/ add factor due to unvisited ramp
        f_uv = np.zeros(nStars)
        unvisited = self.starVisits[sInds] == 0
        f_uv[unvisited] = float(TK.currentTimeNorm.copy() / TK.missionLife.copy()) ** 2
        A = A - self.coeffs[4] * f_uv

        # 5/ add factor due to revisited ramp
        if self.starRevisit.size != 0:
            f2_uv = 1 - (np.in1d(sInds, self.starRevisit[:, 0]))
            A = A + self.coeffs[5] * f2_uv

        # kill diagonal
        A = A + np.diag(np.ones(nStars) * np.inf)

        # take two traversal steps
        step1 = np.tile(A[sInds == old_sInd, :], (nStars, 1)).flatten("F")
        step2 = A[np.array(np.ones((nStars, nStars)), dtype=bool)]
        tmp = np.nanargmin(step1 + step2)
        sInd = sInds[int(np.floor(tmp / float(nStars)))]

        waitTime = slewTimes[sInd]

        return sInd, waitTime

    def revisitFilter(self, sInds, tmpCurrentTimeNorm):
        """Helper method for Overloading Revisit Filtering

        Args:
            sInds - indices of stars still in observation list
            tmpCurrentTimeNorm (MJD) - the simulation time after overhead was added
            in MJD form

        Returns:
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

    def scheduleRevisit(self, sInd, smin, det, pInds):
        """A Helper Method for scheduling revisits after observation detection

        Args:
            sInd - sInd of the star just detected
            smin - minimum separation of the planet to star of planet just detected
            det -
            pInds - Indices of planets around target star

        Returns:
            None

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

    def observation_characterization(self, sInd, mode):
        """Finds if characterizations are possible and relevant information

        Args:
            sInd (int):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization

        Returns:
            tuple:
                characterized (int list):
                    Characterization status for each planet orbiting the observed
                    target star including False Alarm if any, where 1 is full spectrum,
                    -1 partial spectrum, and 0 not characterized
                fZ (astropy.units.Quantity):
                    Surface brightness of local zodiacal light in units of 1/arcsec2
                systemParams (dict):
                    Dictionary of time-dependant planet properties averaged over the
                    duration of the integration
                SNR (float numpy.ndarray):
                    Characterization signal-to-noise ratio of the observable planets.
                    Defaults to None.
                intTime (astropy.units.Quantity):
                    Selected star characterization time in units of day.
                    Defaults to None.

        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # selecting appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        pinds_earthlike = np.array([])
        fEZs = SU.fEZ[pInds].to("1/arcsec2").value
        # dMags = SU.dMag[pInds]
        WAs = SU.WA[pInds].to("arcsec").value

        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd, 0]
        if det is None:
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
            tochar = self.fullSpectra[pIndsDet] < self.max_successful_chars
        else:  # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append(
                (self.fullSpectra[truePlans] < self.max_successful_chars), True
            )

        # 1/ find spacecraft orbital START position including overhead time,
        # and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs.copy()
            startTimeNorm = TK.currentTimeNorm.copy()
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[
                0
            ][
                0
            ]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times at the
        # detected fEZ, dMag, and WA
        if np.any(tochar):
            pinds_earthlike = np.logical_and(
                np.array([(p in self.known_earths) for p in pIndsDet]), tochar
            )

            if self.lastDetected[sInd, 0] is None:
                fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
                fEZ = fEZs[tochar] / u.arcsec**2
                dMag = self.int_dMag[sInd] * np.ones(len(tochar))
                WA = self.int_WA[sInd] * np.ones(len(tochar))
            else:
                fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
                fEZ = self.lastDetected[sInd, 1][det][tochar] / u.arcsec**2
                dMag = self.lastDetected[sInd, 2][det][tochar]
                WA = self.lastDetected[sInd, 3][det][tochar] * u.arcsec
            # dMag = self.int_dMag[sInd]*np.ones(len(tochar))
            # WA = self.int_WA[sInd]*np.ones(len(tochar))

            intTimes = np.zeros(len(tochar)) * u.day

            # if lucky_planets, use lucky planet params for dMag and WA
            if SU.lucky_planets or sInd in self.known_rocky:
                phi = (1 / np.pi) * np.ones(len(SU.d))
                e_dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)  # delta magnitude
                e_WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to(
                    "arcsec"
                )  # working angle
                WA[pinds_earthlike[tochar]] = e_WA[pIndsDet[pinds_earthlike]]
                dMag[pinds_earthlike[tochar]] = e_dMag[pIndsDet[pinds_earthlike]]
            # else:
            #    e_dMag = SU.dMag
            #    e_WA = SU.WA
            #    WA[pinds_earthlike[tochar]] = e_WA[pIndsDet[pinds_earthlike]]
            #    dMag[pinds_earthlike[tochar]] = e_dMag[pIndsDet[pinds_earthlike]]
            # pdb.set_trace() ###
            intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            intTimes[~np.isfinite(intTimes)] = 0 * u.d
            # add a predetermined margin to the integration times
            intTimes = intTimes * (1.0 + self.charMargin)
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

        # 4/ if yes, allocate the overhead time, and perform the characterization
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
                characterized = np.zeros(lenChar, dtype=float)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0.0 / u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)
                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

            pIndsChar = pIndsDet[tochar]
            log_char = "   - Charact. planet inds %s (%s/%s detected)" % (
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
                fZs = np.zeros(self.ntFlux) / u.arcsec**2.0
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

            # if only a FA, just save zodiacal brightness
            # in the middle of the integration
            else:
                totTime = intTime * (mode["timeMultiplier"])
                fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs.copy() + totTime / 2.0, mode)[
                    0
                ]

            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd, 1][-1] / u.arcsec**2.0
                dMag = self.lastDetected[sInd, 2][-1]
                WA = self.lastDetected[sInd, 3][-1] * u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p * intTime).decompose().value
                N = np.sqrt((C_b * intTime + (C_sp * intTime) ** 2.0).decompose().value)
                SNRfa = S / N if N > 0.0 else 0.0

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
            WAs = systemParams["WA"]
            if FA:
                WAs = np.append(WAs, self.lastDetected[sInd, 3][-1] * u.arcsec)
            # check for partial spectra (for coronagraphs only)
            if not (mode["syst"]["occulter"]):
                IWA_max = mode["IWA"] * (1.0 + mode["BW"] / 2.0)
                OWA_min = mode["OWA"] * (1.0 - mode["BW"] / 2.0)
                char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
                characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        # schedule target revisit
        self.scheduleRevisit(sInd, None, None, None)

        return characterized.astype(int), fZ, systemParams, SNR, intTime
