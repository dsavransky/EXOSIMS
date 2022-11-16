from EXOSIMS.SurveySimulation.tieredScheduler_DD import tieredScheduler_DD
import astropy.units as u
import numpy as np
from astropy.time import Time
import copy
from EXOSIMS.util.deltaMag import deltaMag


class tieredScheduler_DD_SS(tieredScheduler_DD):
    """tieredScheduler_DDSS - tieredScheduler Dual Detection with SotoStarshade

    This class implements a version of the tieredScheduler that performs dual-band
    detections with SotoStarshade
    """

    def __init__(self, **specs):

        tieredScheduler_DD.__init__(self, **specs)
        self.inttime_predict = 0 * u.d

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
            tuple:
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
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        # Create DRM
        DRM = {}

        # selecting appropriate koMap
        occ_koMap = self.koMaps[char_mode["syst"]["name"]]
        koMap = self.koMaps[det_modes[0]["syst"]["name"]]

        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        assert OS.haveOcculter
        self.ao = Obs.thrust / Obs.scMass

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
            occ_tmpCurrentTimeAbs = TK.currentTimeAbs.copy()

            # 0 initialize arrays
            slewTimes = np.zeros(TL.nStars) * u.d
            dV = np.zeros(TL.nStars) * u.m / u.s
            intTimes = np.zeros(TL.nStars) * u.d
            occ_intTimes = np.zeros(TL.nStars) * u.d
            occ_tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)

            # 1 Find spacecraft orbital START positions and filter out unavailable
            # targets. If occulter, each target has its own START position.
            sd = Obs.star_angularSep(TL, old_occ_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, char_mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, old_occ_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs
            )

            # 2.1 filter out totTimes > integration cutoff
            if len(sInds) > 0:
                occ_sInds = np.intersect1d(self.occ_intTimeFilterInds, sInds)
            if len(sInds) > 0:
                sInds = np.intersect1d(self.intTimeFilterInds, sInds)

            # Starttimes based off of slewtime
            occ_startTimes = occ_tmpCurrentTimeAbs.copy() + slewTimes
            startTimes = tmpCurrentTimeAbs.copy() + np.zeros(TL.nStars) * u.d

            # 2.5 Filter stars not observable at startTimes
            try:
                tmpIndsbool = list()
                for i in np.arange(len(occ_sInds)):
                    koTimeInd = np.where(
                        np.round(occ_startTimes[occ_sInds[i]].value)
                        - self.koTimes.value
                        == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds_occ_ko = occ_sInds[tmpIndsbool]
                occ_sInds = sInds_occ_ko[np.where(np.in1d(sInds_occ_ko, HIP_sInds))[0]]
                del tmpIndsbool
            except:  # noqa: E722 If there are no target stars to observe
                sInds_occ_ko = np.asarray([], dtype=int)
                occ_sInds = np.asarray([], dtype=int)

            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(
                        np.round(startTimes[sInds[i]].value) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        koMap[sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722 If there are no target stars to observe
                sInds = np.asarray([], dtype=int)

            # 2.9 Occulter target promotion step
            occ_sInds = self.promote_coro_targets(occ_sInds, sInds_occ_ko)

            # 3 Filter out all previously (more-)visited targets, unless in
            # revisit list, with time within some dt of start (+- 1 week)
            if len(sInds.tolist()) > 0:
                sInds = self.revisitFilter(sInds, TK.currentTimeNorm.copy())

            # revisit list, with time after start
            if np.any(occ_sInds):
                occ_tovisit[occ_sInds] = (
                    self.occ_starVisits[occ_sInds]
                    == self.occ_starVisits[occ_sInds].min()
                )
                if self.occ_starRevisit.size != 0:
                    dt_rev = (
                        TK.currentTimeNorm.copy() - self.occ_starRevisit[:, 1] * u.day
                    )
                    ind_rev = [
                        int(x)
                        for x in self.occ_starRevisit[dt_rev > 0, 0]
                        if x in occ_sInds
                    ]
                    occ_tovisit[ind_rev] = True
                occ_sInds = np.where(occ_tovisit)[0]

            # 4 calculate integration times for ALL preselected targets,
            # and filter out totTimes > integration cutoff
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

            (
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
            ) = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
            occ_maxIntTime = min(
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
                OS.intCutoff,
            )  # Maximum intTime allowed
            if len(occ_sInds) > 0:
                # adjustment of integration times due to known earths or inflection
                # point moved to self.refineOcculterIntTimes method
                (
                    occ_sInds,
                    slewTimes[occ_sInds],
                    occ_intTimes[occ_sInds],
                    dV[occ_sInds],
                ) = self.refineOcculterSlews(
                    old_occ_sInd, occ_sInds, slewTimes, obsTimes, sd, char_mode
                )
                occ_startTimes += slewTimes
                occ_endTimes = occ_startTimes + occ_intTimes

                if occ_maxIntTime.value <= 0:
                    occ_sInds = np.asarray([], dtype=int)

            if len(sInds.tolist()) > 0:
                intTimes[sInds] = self.calc_targ_intTime(
                    sInds, startTimes[sInds], det_modes[0]
                )
                sInds = sInds[
                    (intTimes[sInds] <= maxIntTime)
                ]  # Filters targets exceeding end of OB
                endTimes = (
                    startTimes
                    + intTimes
                    + Obs.settlingTime
                    + det_modes[0]["syst"]["ohTime"]
                )

                if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)

            # 5.2 find spacecraft orbital END positions (for each candidate target),
            # and filter out unavailable targets
            if len(occ_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                # endTimes may exist past koTimes so we have an exception to
                # hand this case
                try:
                    tmpIndsbool = list()
                    for i in np.arange(len(occ_sInds)):
                        koTimeInd = np.where(
                            np.round(occ_endTimes[occ_sInds[i]].value)
                            - self.koTimes.value
                            == 0
                        )[0][
                            0
                        ]  # find indice where koTime is endTime[0]
                        tmpIndsbool.append(
                            occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)
                        )  # Is star observable at time ind
                    occ_sInds = occ_sInds[tmpIndsbool]
                    del tmpIndsbool
                except:  # noqa: E722
                    occ_sInds = np.asarray([], dtype=int)

            if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                # endTimes may exist past koTimes so we have an exception to handle
                # this case
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

            # 5.3 Filter off current occulter target star from detection list
            if old_occ_sInd is not None:
                sInds = sInds[(sInds != old_occ_sInd)]
                occ_sInds = occ_sInds[(occ_sInds != old_occ_sInd)]

            # 6.1 Filter off any stars visited by the occulter 3 or more times
            if np.any(occ_sInds):
                occ_sInds = occ_sInds[
                    (self.occ_starVisits[occ_sInds] < self.occ_max_visits)
                ]

            # 6.2 Filter off coronograph stars with > 3 visits and no detections
            no_dets = np.logical_and(
                (self.starVisits[sInds] > self.n_det_remove),
                (self.sInd_detcounts[sInds] == 0),
            )
            sInds = sInds[np.where(np.invert(no_dets))[0]]

            max_dets = np.where(self.sInd_detcounts[sInds] < self.max_successful_dets)[
                0
            ]
            sInds = sInds[max_dets]

            # 7 Filter off cornograph stars with too-long inttimes
            available_time = None
            if self.occ_arrives > TK.currentTimeAbs:
                available_time = self.occ_arrives - TK.currentTimeAbs.copy()
                if np.any(sInds[intTimes[sInds] < available_time]):
                    sInds = sInds[intTimes[sInds] < available_time]

            # 8 remove occ targets on ignore_stars list
            occ_sInds = np.setdiff1d(
                occ_sInds, np.intersect1d(occ_sInds, self.ignore_stars)
            )

            tmpIndsbool = list()
            for i in np.arange(len(occ_sInds)):
                koTimeInd = np.where(
                    np.round(occ_startTimes[occ_sInds[i]].value) - self.koTimes.value
                    == 0
                )[0][
                    0
                ]  # find indice where koTime is endTime[0]
                tmpIndsbool.append(
                    occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind

            t_det = 0 * u.d
            det_mode = copy.deepcopy(det_modes[0])
            occ_sInd = old_occ_sInd

            # 9 Choose best target from remaining
            # if the starshade has arrived at its destination, or it is the first
            # observation
            if np.any(occ_sInds):
                if old_occ_sInd is None or (
                    (TK.currentTimeAbs.copy() + t_det) >= self.occ_arrives
                    and self.ready_to_update
                ):
                    occ_sInd = self.choose_next_occulter_target(
                        old_occ_sInd, occ_sInds, occ_intTimes, slewTimes
                    )
                    if old_occ_sInd is None:
                        self.occ_arrives = TK.currentTimeAbs.copy()
                    else:
                        self.occ_arrives = occ_startTimes[occ_sInd]
                        self.occ_slewTime = slewTimes[occ_sInd]
                        self.occ_sd = sd[occ_sInd]
                        self.inttime_predict = occ_intTimes[occ_sInd]
                    self.ready_to_update = False
                elif not np.any(sInds):
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + 1 * u.d)
                    continue

            if occ_sInd is not None:
                sInds = sInds[np.where(sInds != occ_sInd)]

            if self.tot_det_int_cutoff < self.tot_dettime:
                sInds = np.array([])

            if np.any(sInds):

                # choose sInd of next target
                sInd = self.choose_next_telescope_target(
                    old_sInd, sInds, intTimes[sInds]
                )

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

                t_det = self.calc_targ_intTime(
                    np.array([sInd]), startTimes[sInd], det_mode
                )[0]

                if t_det > maxIntTime and maxIntTime > 0 * u.d:
                    t_det = maxIntTime
                if available_time is not None and available_time > 0 * u.d:
                    if t_det > available_time:
                        t_det = available_time.copy().value * u.d

            # if no observable target, call the TimeKeeping.wait() method
            if not np.any(sInds) and not np.any(occ_sInds):
                self.vprint(
                    "No Observable Targets at currentTimeNorm= "
                    + str(TK.currentTimeNorm.copy())
                )
                return DRM, None, None, None, None, None, None
            break

        else:
            self.logger.info("Mission complete: no more time available")
            self.vprint("Mission complete: no more time available")
            return DRM, None, None, None, None, None, None

        if TK.mission_is_over(OS, Obs, det_mode):
            self.logger.info("Mission complete: no more time available")
            self.vprint("Mission complete: no more time available")
            return DRM, None, None, None, None, None, None

        return DRM, sInd, occ_sInd, t_det, sd, occ_sInds, det_mode

    def refineOcculterIntTimes(self, occ_sInds, occ_startTimes, char_mode):
        """Refines/filters/chooses occulter intTimes based on tieredScheduler criteria

        This method filters the intTimes for the remaining occ_sInds according to
        the tiered_Scheduler_DD criteria. Including int_inflection and adjusting for
        stars with known earths around them. Code was copied from tieredScheduler_DD
        in section #4 to this method so that it can be conducted multiple times for
        different occ_startTimes. This method was created for use with SotoStarshade.
        In refineOcculterSlews with SotoStarshade, the intTimes are calculated once
        at the start of the observable time window and again at the end of it.

        Args:
            occ_sInds (integer array):
                Indices of available targets for occulter
            occ_startTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            char_mode (dict):
                Selected observing mode for detection

        Returns:
            occ_intTimes (astropy Quantity):
                Filtered occulter integration times
        """
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        OS = self.OpticalSystem
        TL = self.TargetList
        ZL = self.ZodiacalLight
        TK = self.TimeKeeping

        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
        occ_maxIntTime = min(
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
            OS.intCutoff,
        )  # Maximum intTime allowed

        if self.int_inflection:
            fEZ = ZL.fEZ0
            WA = TL.int_WA
            occ_intTimes = self.calc_int_inflection(
                occ_sInds, fEZ, occ_startTimes, WA[occ_sInds], char_mode, ischar=True
            )
        else:
            # characterization_start = occ_startTimes
            occ_intTimes = self.calc_targ_intTime(
                occ_sInds, occ_startTimes[occ_sInds], char_mode
            ) * (1 + self.charMargin)

            # Adjust integration time for stars with known earths around them
            for i, occ_star in enumerate(occ_sInds):
                if occ_star in self.promoted_stars:
                    occ_earths = np.intersect1d(
                        np.where(SU.plan2star == occ_star)[0], self.known_earths
                    ).astype(int)
                    if np.any(occ_earths):
                        fZ = ZL.fZ(
                            Obs, TL, occ_star, occ_startTimes[occ_star], char_mode
                        )
                        fEZ = SU.fEZ[occ_earths].to("1/arcsec2").value / u.arcsec**2
                        if SU.lucky_planets:
                            phi = (1 / np.pi) * np.ones(len(SU.d))
                            dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[
                                occ_earths
                            ]  # delta magnitude
                            WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to("arcsec")[
                                occ_earths
                            ]  # working angle
                        else:
                            dMag = SU.dMag[occ_earths]
                            WA = SU.WA[occ_earths]

                        if np.all((WA < char_mode["IWA"]) | (WA > char_mode["OWA"])):
                            occ_intTimes[i] = 0.0 * u.d
                        else:
                            earthlike_inttimes = OS.calc_intTime(
                                TL, occ_star, fZ, fEZ, dMag, WA, char_mode
                            ) * (1 + self.charMargin)
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < occ_maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                occ_intTimes[i] = np.max(earthlike_inttime)
                            else:
                                occ_intTimes[i] = np.max(earthlike_inttimes)
        return occ_intTimes

    def refineOcculterSlews(self, old_sInd, sInds, slewTimes, obsTimes, sd, mode):
        """Refines/filters/chooses occulter slews based on time constraints

        Refines the selection of occulter slew times by filtering based on mission time
        constraints and selecting the best slew time for each star. This method calls on
        other occulter methods within SurveySimulation depending on how slew times were
        calculated prior to calling this function (i.e. depending on which
        implementation of the Observatory module is used).

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            obsTimes (astropy Quantity array):
                A binary array with TargetList.nStars rows and
                (missionFinishAbs-missionStart)/dt columns
                where dt is 1 day by default. A value of 1 indicates the star is in
                keepout for (and therefore cannot be observed). A value of 0 indicates
                the star is not in keepout and may be observed.
            sd (astropy Quantity):
                Angular separation between stars in rad
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                sInds (integer):
                    Indeces of next target star
                slewTimes (astropy Quantity array):
                    slew times to all stars (must be indexed by sInds)
                intTimes (astropy Quantity array):
                    Integration times for detection in units of day
                dV (astropy Quantity):
                    Delta-V used to transfer to new star line of sight in unis of m/s
        """

        Obs = self.Observatory
        TL = self.TargetList

        # initializing arrays
        obsTimeArray = np.zeros([TL.nStars, 50]) * u.d
        intTimeArray = np.zeros([TL.nStars, 2]) * u.d

        for n in sInds:
            obsTimeArray[n, :] = (
                np.linspace(obsTimes[0, n].value, obsTimes[1, n].value, 50) * u.d
            )

        intTimeArray[sInds, 0] = self.refineOcculterIntTimes(
            sInds, Time(obsTimeArray[:, 0], format="mjd", scale="tai"), mode
        )
        intTimeArray[sInds, 1] = self.refineOcculterIntTimes(
            sInds, Time(obsTimeArray[:, -1], format="mjd", scale="tai"), mode
        )

        # added this for tieredScheduler
        intTimeArray *= mode["timeMultiplier"]

        # determining which scheme to use to filter slews
        obsModName = Obs.__class__.__name__

        # slew times have not been calculated/decided yet (SotoStarshade)
        if obsModName == "SotoStarshade":
            sInds, intTimes, slewTimes, dV = self.findAllowableOcculterSlews(
                sInds,
                old_sInd,
                sd[sInds],
                slewTimes[sInds],
                obsTimeArray[sInds, :],
                intTimeArray[sInds, :],
                mode,
            )
        # slew times were calculated/decided beforehand (Observatory Prototype)
        else:
            sInds, intTimes, slewTimes = self.filterOcculterSlews(
                sInds,
                slewTimes[sInds],
                obsTimeArray[sInds, :],
                intTimeArray[sInds, :],
                mode,
            )
            dV = np.zeros(len(sInds)) * u.m / u.s

        return sInds, slewTimes, intTimes, dV

    def chooseOcculterSlewTimes(self, sInds, slewTimes, dV, intTimes, loTimes):
        """Selects the best slew time for each star

        This method searches through an array of permissible slew times for
        each star and chooses the best slew time for the occulter based on
        maximizing possible characterization time for that particular star (as
        a default).

        Args:
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in unis of m/s
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
            loTimes (astropy Quantity array):
                Time left over after integration which could be used for
                characterization in units of day

        Returns:
            tuple:
            sInds (integer):
                Indeces of next target star
            slewTimes (astropy Quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in unis of m/s
        """

        # selection criteria for each star slew
        tmpSlewTimes = slewTimes.copy()

        # filter any slews that are == 0 by default
        badSlew_i, badSlew_j = np.where(tmpSlewTimes <= 0)
        tmpSlewTimes[badSlew_i, badSlew_j] = np.inf

        # filter any slews that would use up too much fuel
        badDV_i, badDV_j = np.where(dV > self.Observatory.dVmax)
        tmpSlewTimes[badDV_i, badDV_j] = np.inf

        # minimum slew time possible -> get to the star QUICK!
        good_j = np.argmin(tmpSlewTimes, axis=1)
        good_i = np.arange(0, len(sInds))

        dV = dV[good_i, good_j]
        intTime = intTimes[good_i, good_j]
        slewTime = slewTimes[good_i, good_j]

        return sInds, intTime, slewTime, dV

    def choose_next_occulter_target(
        self, old_occ_sInd, occ_sInds, intTimes, slewTimes=None
    ):
        """Choose next target for the occulter based on truncated
        depth first search of linear cost function.

        Args:
            old_occ_sInd (integer):
                Index of the previous target star
            occ_sInds (integer array):
                Indices of available targets
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds). Input is
                set to None by default

        Returns:
            sInd (integer):
                Index of next target star

        """

        # Choose next Occulter target

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem

        # reshape sInds, store available top9 sInds
        occ_sInds = np.array(occ_sInds, ndmin=1)
        top_HIPs = self.occHIPs[: self.topstars]
        top_sInds = np.intersect1d(np.where(np.in1d(TL.Name, top_HIPs))[0], occ_sInds)

        # current stars have to be in the adjmat
        if (old_occ_sInd is not None) and (old_occ_sInd not in occ_sInds):
            occ_sInds = np.append(occ_sInds, old_occ_sInd)

        # get completeness values
        comps = Comp.completeness_update(
            TL, occ_sInds, self.occ_starVisits[occ_sInds], TK.currentTimeNorm.copy()
        )

        # if first target, or if only 1 available target, choose highest
        # available completeness
        nStars = len(occ_sInds)
        if (old_occ_sInd is None) or (nStars == 1):
            occ_sInd = np.random.choice(occ_sInds[comps == max(comps)])
            return occ_sInd

        # define adjacency matrix
        A = np.zeros((nStars, nStars))

        # consider slew distance when there's an occulter
        A[np.ones((nStars), dtype=bool)] = slewTimes[occ_sInds].to("d").value
        A = self.coeffs[0] * (A) / (1 * u.yr).to("d").value

        # add factor due to completeness
        A = A + self.coeffs[1] * (1 - comps)

        # add factor due to intTime
        intTimes[old_occ_sInd] = np.inf
        A = A + self.coeffs[2] * (intTimes[occ_sInds] / OS.intCutoff)

        # add factor for unvisited ramp for deep dive stars
        if np.any(top_sInds):
            # add factor for least visited deep dive stars
            f_uv = np.zeros(nStars)
            u1 = np.in1d(occ_sInds, top_sInds)
            u2 = self.occ_starVisits[occ_sInds] == min(self.occ_starVisits[top_sInds])
            unvisited = np.logical_and(u1, u2)
            f_uv[unvisited] = (
                float(TK.currentTimeNorm.copy() / TK.missionLife.copy()) ** 2
            )
            A = A - self.coeffs[3] * f_uv

            self.coeff_data_a3.append([occ_sInds, f_uv])

            # add factor for unvisited deep dive stars
            no_visits = np.zeros(nStars)
            # no_visits[u1] = np.ones(len(top_sInds))
            u2 = self.occ_starVisits[occ_sInds] == 0
            unvisited = np.logical_and(u1, u2)
            no_visits[unvisited] = 1.0
            A = A - self.coeffs[4] * no_visits

            self.coeff_data_a4.append([occ_sInds, no_visits])
            self.coeff_time.append(TK.currentTimeNorm.copy().value)

        # add factor due to unvisited ramp
        f_uv = np.zeros(nStars)
        unvisited = self.occ_starVisits[occ_sInds] == 0
        f_uv[unvisited] = float(TK.currentTimeNorm.copy() / TK.missionLife.copy()) ** 2
        A = A - self.coeffs[5] * f_uv

        # add factor due to revisited ramp
        if self.occ_starRevisit.size != 0:
            f2_uv = 1 - (np.in1d(occ_sInds, self.occ_starRevisit[:, 0]))
            A = A + self.coeffs[6] * f2_uv

        # kill diagonal
        A = A + np.diag(np.ones(nStars) * np.Inf)

        # take two traversal steps
        step1 = np.tile(A[occ_sInds == old_occ_sInd, :], (nStars, 1)).flatten("F")
        step2 = A[np.array(np.ones((nStars, nStars)), dtype=bool)]
        tmp = np.nanargmin(step1 + step2)
        occ_sInd = occ_sInds[int(np.floor(tmp / float(nStars)))]

        return occ_sInd
