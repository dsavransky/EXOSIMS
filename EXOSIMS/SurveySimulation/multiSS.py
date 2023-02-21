from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np


class multiSS(SurveySimulation):
    def __init__(self, coeffs=[-1, -2, np.e, np.pi], count=0, **specs):

        SurveySimulation.__init__(self, **specs)

        # verify that coefficients input is iterable 4x1
        if not (isinstance(coeffs, (list, tuple, np.ndarray))) or (len(coeffs) != 4):
            raise TypeError("coeffs must be a 4 element iterable")
        self.count = count
        # Add to outspec
        self._outspec["coeffs"] = coeffs

        # normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs / np.linalg.norm(coeffs)

        # initialize the second target star
        self.second_target = None
        self.ko = 1

        self.coeff = coeffs

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
        allModes = OS.observingModes

        # create DRM
        DRM = {}
        self.DRM = [DRM]
        # populate DRM with 2 fake star_sInd values to star the mission
        DRM["star_ind"] = [0, 0]

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

        # calculate the angular separation and slew times for both starshades
        if OS.haveOcculter:
            sd = Obs.star_angularSep(
                TL, self.DRM[-1]["star_ind"][-1], sInds, tmpCurrentTimeAbs
            )
            sd_2 = Obs.star_angularSep(
                TL, self.DRM[-1]["star_ind"][-2], sInds, tmpCurrentTimeAbs
            )

            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, self.DRM[-1]["star_ind"][-1], sInds, sd, 0, None
            )
            slewTimes_2 = Obs.calculate_slewTimes(
                TL, self.DRM[-1]["star_ind"][-2], sInds, sd_2, 1, None
            )
       
        self.slewTimes_2 = slewTimes_2
        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # start times, including slew times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        """# 2.5 Filter stars not observable at startTimes (not required in this implementation as we're
        checking for 2 target at once in choose_next_Target)
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
        except:  # noqa: E722 # If there are no target stars to observe
            sInds = np.asarray([], dtype=int)"""

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

        if len(sInds.tolist()) > 0:
            if OS.haveOcculter and (old_sInd is not None):
                (
                    sInds,
                    slewTimes[sInds],
                    intTimes[sInds],
                    dV[sInds],
                ) = self.refineOcculterSlews(
                    self.DRM[-1]["star_ind"][-1], sInds, slewTimes, obsTimes, sd, mode
                )
                (
                    sInds,
                    slewTimes_2[sInds],
                    intTimes[sInds],
                    dV_2[sInds],
                ) = self.refineOcculterSlews(
                    self.DRM[-1]["star_ind"][-2],
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
                    sInds = np.asarray([], dtype=int)

        """# 5.1 TODO Add filter to filter out stars entering and exiting keepout
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
                sInds = np.asarray([], dtype=int)"""

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0 and old_sInd is None:

            # calculating the first target star based on maximum completeness value

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

            return (
                sInd,
                slewTimes[sInd],
            )

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

        # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        # populate DRM with occulter related values
        if OS.haveOcculter:
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

        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimes_2 = tmpCurrentTimeAbs.copy() + self.slewTimes_2
        
        intTimes_2 = np.zeros(sInds)
        #integration time for all the target that will be observed first
        intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)

        #integration time for all the targets that will be observed second
        intTimes_2[sInds] = self.calc_targ_intTime(sInds,startTimes_2[sInds],mode)
        # appropriate koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # cast sInds to array (pre-filtered target list)
        sInds = np.array(sInds, ndmin=1, copy=False)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm.copy() + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values (use it for later purposes)
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)

        # defining a dummy cost matrix for random walk scheduler
        cost_matrix = np.array([sInds, sInds])

        cost_matrix = cost_matrix * np.random.randint(100, size=(sInds, sInds))
        # kill diagonal
        cost_matrix = np.fill_diagonal(cost_matrix, 1e9)

        if self.second_target is None:

            while self.ko == 1:
                # figure out the next two steps, edit method to select a random index instead of an element from array.
                h = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
                first_target_sInd = h[0]
                second_target_sInd = h[1]
                np.prod(
                    koMap[
                        first_target_sInd,
                        TK.currentTimeNorm.copy() : TK.currentTimeNorm.copy(),
                        +intTimes[first_target_sInd] + slewTimes[first_target_sInd],
                    ]
                ) + np.prod(
                    koMap[
                        second_target_sInd,
                        TK.currentTimeNorm.copy() + intTimes[first_target_sInd],
                        +slewTimes[first_target_sInd] : TK.currentTimeNorm.copy()
                        + intTimes[first_target_sInd]
                        + slewTimes[first_target_sInd]
                        + self.slewTimes_2[second_target_sInd]
                        + intTimes_2[second_target_sInd],
                    ]
                )
                if self.ko == 0:
                    pass
                else:
                    cost_matrix[h] = 1e9
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
            Obs.scMass[0] = Obs.scMass[0] - mass_used
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
            Obs.scMass[1] = Obs.scMass[1] - mass_used
            DRM["scMass_second"] = Obs.scMass[1].to("kg")
            if Obs.twotanks:
                Obs.skMass = Obs.skMass - mass_used
                DRM["skMass"] = Obs.skMass.to("kg")

        return DRM
