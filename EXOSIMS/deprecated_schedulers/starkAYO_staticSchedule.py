from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import scipy
import os.path
import pickle


class starkAYO_staticSchedule(SurveySimulation):
    """starkAYO _static Scheduler

    This class implements a Scheduler that creates a list of stars to observe and
    integration times to observe them. It also selects the best star to observe at any
    moment in time

    2nd execution time 2 min 30 sec
    """

    def __init__(self, cacheOptTimes=False, staticOptTimes=False, **specs):
        SurveySimulation.__init__(self, **specs)

        assert isinstance(staticOptTimes, bool), "staticOptTimes must be boolean."
        self.staticOptTimes = staticOptTimes
        self._outspec["staticOptTimes"] = self.staticOptTimes

        assert isinstance(cacheOptTimes, bool), "cacheOptTimes must be boolean."
        self._outspec["cacheOptTimes"] = cacheOptTimes

        # bring inherited class objects to top level of Survey Simulation
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        # Create and start Schedule
        self.schedule = np.arange(TL.nStars)  # self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(
            TL.nStars
        )  # preserves initial list of targets

        dMagLim = max(TL.intCutoff_dMag)
        self.dmag_startSaved = np.linspace(1, dMagLim, num=1500, endpoint=True)

        sInds = self.schedule_startSaved.copy()
        dmag = self.dmag_startSaved
        WA = TL.int_WA
        startTime = np.zeros(sInds.shape[0]) * u.d + TK.currentTimeAbs  # noqa: F841
        self.detmode = list(
            filter(
                lambda mode: mode["detectionMode"],
                self.OpticalSystem.observingModes,
            )
        )[0]

        # Estimate Yearly fZmax
        self.fZmax, self.abdTimefZmax = ZL.calcfZmax(
            sInds, Obs, TL, TK, self.detmode, self.cachefname
        )

        # CACHE Cb Cp Csp
        # Sept 20, 2017 execution time 10.108 sec
        # WE DO NOT NEED TO CALCULATE OVER EVERY DMAG
        Cp = np.zeros([sInds.shape[0], dmag.shape[0]])
        Cb = np.zeros(sInds.shape[0])
        Csp = np.zeros(sInds.shape[0])
        for i in range(dmag.shape[0]):
            Cp[:, i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(
                TL, sInds, self.valfZmin, ZL.fEZ0, dmag[i], WA, self.detmode
            )
        self.Cb = Cb[:] / u.s
        self.Csp = Csp[:] / u.s

        # Load cached Observation Times
        cachefname = self.cachefname + "starkcache"  # Generate cache Name
        if cacheOptTimes and os.path.isfile(
            cachefname
        ):  # Checks if flag to load cached optimal times exists
            self.vprint("Loading starkcache from %s" % cachefname)
            try:
                with open(cachefname, "rb") as ff:
                    tmpDat = pickle.load(ff)
            except UnicodeDecodeError:
                with open(cachefname, "rb") as ff:
                    tmpDat = pickle.load(ff, encoding="latin1")
            self.schedule = tmpDat[0, :].astype(int)
            self.t_dets = tmpDat[1, :]
            self.CbyT = tmpDat[2, :]
            self.Comp00 = tmpDat[3, :]
        else:  # create cachedOptTimes
            self.altruisticYieldOptimization(sInds)
        self.t0 = np.zeros(TL.nStars)
        self.t0[self.schedule] = self.t_dets
        # END INIT

    def altruisticYieldOptimization(self, sInds):
        """
        Updates attributes:
        self.schedule
        self.t_dets
        self.CbyT
        self.Comp00
        """
        TL = self.TargetList
        ZL = self.ZodiacalLight
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        Obs = self.Observatory
        WA = TL.int_WA

        # Calculate Initial Integration Times
        maxCbyTtime = self.calcTinit(
            sInds, TL, self.valfZmin, ZL.fEZ0, WA, self.detmode
        )
        t_dets = maxCbyTtime  # [sInds] #t_dets has length TL.nStars

        # Sacrifice Stars and then Distribute Excess Mission Time
        # Sept 28, 2017 execution time 19.0 sec
        startingsInds = len(sInds)
        overheadTime = (
            Obs.settlingTime.value + OS.observingModes[0]["syst"]["ohTime"].value
        )  # OH time in days
        print(str(TK.currentTimeNorm.value) + " " + str(TK.missionLife.to("day").value))
        while (sum(t_dets) + sInds.shape[0] * overheadTime) > (
            (TK.missionLife.to("day").value - TK.currentTimeNorm.to("day").value)
            * TK.missionPortion
        ):  # the sum of star observation times is still larger than the mission length
            sInds, t_dets, sacrificedStarTime = self.sacrificeStarCbyT(
                sInds, t_dets, self.valfZmin[sInds], ZL.fEZ0, WA, overheadTime
            )
        self.vprint(
            "Started with "
            + str(startingsInds)
            + " sacrificed down to "
            + str(sInds.shape[0])
        )

        if sum(t_dets + sInds.shape[0] * overheadTime) > (
            (TK.missionLife.to("day").value - TK.currentTimeNorm.to("day").value)
            * TK.missionPortion
        ):  # There is some excess time
            sacrificedStarTime = (
                (TK.missionLife.to("day").value - TK.currentTimeNorm.to("day").value)
                * TK.missionPortion
            ) - (
                sum(t_dets) + sInds.shape[0] * overheadTime
            )  # The amount of time the list is under the total mission Time
            t_dets = self.distributedt(
                sInds, t_dets, sacrificedStarTime, self.valfZmin[sInds], ZL.fEZ0, WA
            )

        # STARK AYO LOOP
        numits = 0  # ensure an infinite loop does not occur. Should be depricated
        # this is some ludacrisly negative number to ensure sumcomp runs.
        # All sumcomps should be positive
        lastIterationSumComp = -10000000
        while numits < 100000 and sInds is not None:
            numits = numits + 1  # we increment numits each loop iteration

            # Sacrifice Lowest Performing Star
            # Sept 28, 2017 execution time 0.0744 0.032 at smallest list size
            sInds, t_dets, sacrificedStarTime = self.sacrificeStarCbyT(
                sInds, t_dets, self.valfZmin[sInds], ZL.fEZ0, WA, overheadTime
            )

            # Distribute Sacrificed Time to new star observations
            # Sept 28, 2017 execution time 0.715, 0.64 at smallest (depends on # stars)
            t_dets = self.distributedt(
                sInds, t_dets, sacrificedStarTime, self.valfZmin[sInds], ZL.fEZ0, WA
            )

            # AYO Termination Conditions
            # Sept 28, 2017 execution time 0.033 sec
            Comp00 = self.Completeness.comp_per_intTime(
                t_dets * u.d,
                TL,
                sInds,
                self.valfZmin[sInds],
                ZL.fEZ0,
                WA[sInds],
                self.detmode,
                self.Cb[sInds],
                self.Csp[sInds],
            )

            # change this to an assert
            if 1 >= len(sInds):  # if this is the last ement in the list
                break
            # If the total sum of completeness at this moment is less than the last
            # sum, then exit
            # If sacrificing the additional target reduced performance,
            # then Define Output of AYO Process
            if sum(Comp00) < lastIterationSumComp:
                CbyT = (
                    self.Completeness.comp_per_intTime(
                        t_dets * u.d,
                        self.TargetList,
                        sInds,
                        self.valfZmin[sInds],
                        ZL.fEZ0,
                        WA[sInds],
                        self.detmode,
                        self.Cb[sInds],
                        self.Csp[sInds],
                    )
                    / t_dets
                )  # takes 5 seconds to do 1 time for all stars
                sortIndex = np.argsort(CbyT, axis=-1)[::-1]

                # This is the static optimal schedule
                self.schedule = sInds[sortIndex]
                self.t_dets = t_dets[sortIndex]
                self.CbyT = CbyT[sortIndex]
                # self.fZ = fZ[sortIndex]
                self.Comp00 = Comp00[sortIndex]

                cachefname = self.cachefname + "starkcache"  # Generate cache Name
                tmpDat = np.zeros([4, self.schedule.shape[0]])
                tmpDat[0, :] = self.schedule
                tmpDat[1, :] = self.t_dets
                tmpDat[2, :] = self.CbyT
                tmpDat[3, :] = self.Comp00
                if self.staticOptTimes:
                    self.vprint("Saving starkcache to %s" % cachefname)
                    with open(cachefname, "wb") as f:  # save to cache
                        pickle.dump(tmpDat, f)
                break
            else:  # else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                self.vprint(
                    "%d SumComp %.2f Sum(t_dets) %.2f sInds %d Time Conservation %.2f"
                    % (
                        numits,
                        sum(Comp00),
                        sum(t_dets),
                        sInds.shape[0],
                        sum(t_dets) + sInds.shape[0] * overheadTime,
                    )
                )
        # End While Loop

    def choose_next_target(self, old_sInd, sInds, slewTime, intTimes):
        """Generate Next Target to Select based off of AYO at this instant in time

        Args:
            sInds (int array):
                indicies of stars under consideration
            old_sInd (int):
                unused
            slewTime (quantity):
                unused
            intTimes (quantity array):
                unused

        Returns:
            tuple:
                sInd (int):
                    the single index of self.schedule_startSaved to observe
                waitTime (astropy.units.Quantity):
                    a strategic amount of time to wait (this module always returns None)
        """
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        # float from 0 to 1000 of where at in 1 year
        # indexFrac = np.interp((TK.currentTimeNorm).value%365.25,[0,365.25],[0,1000])
        # tmp = np.asarray(ZL.fZMap)[self.schedule,:]
        # fZ_matrixSched = (indexFrac%1)*tmp[:,int(indexFrac)]
        #               + (1-indexFrac%1)*tmp[:,int(indexFrac%1+1)]#A simple interpolant

        # fZ_matrixSched = ZL.fZ(Obs, TL, sInds, TK.currentTimeAbs, self.mode)
        fZ_matrixSched = np.zeros(TL.nStars)
        fZ_matrixSched[sInds] = ZL.fZ(Obs, TL, sInds, TK.currentTimeAbs, self.detmode)
        # fZ_matrixSched = np.asarray(ZL.fZMap)[self.schedule,indexFrac]
        # has shape [self.schedule.shape[0], 1000]
        # The above line might not work because it must be filtered down
        # one index at a time...
        fZminSched = np.zeros(TL.nStars)
        fZminSched[sInds] = self.valfZmin[
            sInds
        ].value  # has shape [self.schedule.shape[0]]

        commonsInds = np.intersect1d(self.schedule, sInds)  # noqa: F841

        # commonsInds = [x for x in self.schedule if x in sInds]
        # finds indicies in common between sInds and self.schedule
        # (we need the inherited filtering from sInds)
        # indmap = [self.schedule.tolist().index(x) for x in commonsInds]
        # get index of schedule for the Inds in self.schedule and sInds
        # indmap = [x for x in np.arange(self.schedule.shape[0])
        #                           if self.schedule[x] in sInds]

        # maps self.schedule defined to Inds ##find indicies of occurence of
        # commonsInds in self.schedule
        indmap1 = [self.schedule.tolist().index(x) for x in self.schedule if x in sInds]
        # maps sInds defined to Inds
        indmap2 = [sInds.tolist().index(x) for x in sInds if x in self.schedule]
        # indmap = [x for x in self.schedule if x in sInds]

        # CbyT = np.zeros(TL.nStars)
        # CbyT[self.schedule] = self.CbyT
        CbyT = self.CbyT[indmap1]  # noqa: F841
        # t_dets = np.zeros(TL.nStars)
        # t_dets[self.schedule] = self.t_dets
        t_dets = self.t_dets[indmap1]  # noqa: F841
        # Comp00 = np.zeros(TL.nStars)
        # Comp00[self.schedule] = self.Comp00
        Comp00 = self.Comp00[indmap1]

        fZ = fZ_matrixSched[indmap2]  # noqa: F841 # DONE
        fZmin = fZminSched[indmap2]  # noqa: F841 # DONE

        tmp = TL.coords.dec[self.schedule].value
        dec = tmp[indmap1]  # noqa: F841

        if len(indmap1) > 0:
            # store selected star integration time
            # selectInd = np.argmin(Comp00*abs(fZ-fZmin)/abs(dec))
            selectInd = np.argmin(1 / Comp00)
            sInd = self.schedule[indmap1[selectInd]]

            return sInd, None
        else:  # return a strategic amount of time to wair
            return None, 1 * u.d

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Finds and Returns Precomputed Observation Time

        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.
                must be of the same size as sInds
            mode (dict):
                Selected observing mode for detection

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Integration times for detection. same dimension as sInds
        """
        # commonsInds = [val for val in self.schedule if val in sInds]
        # finds indicies in common between sInds and self.schedule
        if self.staticOptTimes:
            imat = [
                self.schedule.tolist().index(x) for x in self.schedule if x in sInds
            ]  # find indicies of occurence of commonsInds in self.schedule
            intTimes = np.zeros(
                self.TargetList.nStars
            )  # default observation time is 0 days
            intTimes[self.schedule[imat]] = self.t_dets[imat]  #
            intTimes = intTimes * u.d  # add units of day to intTimes
            return intTimes[sInds]
        else:
            self.altruisticYieldOptimization(sInds)
            imat = [
                self.schedule.tolist().index(x) for x in self.schedule if x in sInds
            ]  # find indicies of occurence of commonsInds in self.schedule
            intTimes = np.zeros(
                self.TargetList.nStars
            )  # default observation time is 0 days
            intTimes[self.schedule[imat]] = self.t_dets[imat]  #
            intTimes = intTimes * u.d  # add units of day to intTimes
            return intTimes[sInds]

    def distributedt(
        self, sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA
    ):  # distributing the sacrificed time
        """Distributes sacrificedStarTime amoung sInds
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
            fZ[nStars] - zodiacal light for each target
            fEZ - 0
        Returns:
            t_dets[len(sInds)] - time to observe each star (in days)
        """
        # Calculate dCbydT for each star at this point in time
        dCbydt = self.Completeness.dcomp_dt(
            t_dets * u.d,
            self.TargetList,
            sInds,
            fZ,
            fEZ,
            WA[sInds],
            self.detmode,
            C_b=self.Cb[sInds],
            C_sp=self.Csp[sInds],
            TK=self.TimeKeeping,
        )  # dCbydT[len(sInds)]#Sept 28, 2017 0.12sec

        if len(t_dets) <= 1:
            return t_dets

        timeToDistribute = sacrificedStarTime
        dt_static = 0.1
        dt = dt_static

        # Now decide where to put dt
        numItsDist = 0
        while timeToDistribute > 0:
            if numItsDist > 1000000:  # this is an infinite loop check
                break
            else:
                numItsDist = numItsDist + 1

            # if the timeToDistribute is smaller than dt
            if timeToDistribute < dt:
                dt = timeToDistribute  # dt is now the timeToDistribute
            # timeToDistribute >= dt under nominal conditions, this is dt to use
            else:
                # this is the maximum quantity of time to distribute at a time.
                dt = dt_static

            maxdCbydtIndex = np.argmax(dCbydt)  # Find most worthy target

            t_dets[maxdCbydtIndex] = (
                t_dets[maxdCbydtIndex] + dt
            )  # Add dt to the most worthy target
            timeToDistribute = (
                timeToDistribute - dt
            )  # subtract distributed time dt from the timeToDistribute
            dCbydt[maxdCbydtIndex] = self.Completeness.dcomp_dt(
                t_dets[maxdCbydtIndex] * u.d,
                self.TargetList,
                sInds[maxdCbydtIndex],
                fZ[maxdCbydtIndex],
                fEZ,
                WA[maxdCbydtIndex],
                self.detmode,
                C_b=self.Cb[maxdCbydtIndex],
                C_sp=self.Csp[maxdCbydtIndex],
                TK=self.TimeKeeping,
            )  # dCbydT[nStars]#Sept 28, 2017 0.011sec
        # End While Loop
        return t_dets

    def sacrificeStarCbyT(self, sInds, t_dets, fZ, fEZ, WA, overheadTime):
        """Sacrifice the worst performing CbyT star
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            fZ[nStars] - zodiacal light for each target
            fEZ - 0
            WA - inner working angle of the instrument
            overheadTime - overheadTime added to each observation
        Return:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
        """
        CbyT = (
            self.Completeness.comp_per_intTime(
                t_dets * u.d,
                self.TargetList,
                sInds,
                self.valfZmin[sInds],
                fEZ,
                WA[sInds],
                self.detmode,
                self.Cb[sInds],
                self.Csp[sInds],
            )
            / t_dets
        )  # takes 5 seconds to do 1 time for all stars

        sacrificeIndex = np.argmin(CbyT)  # finds index of star to sacrifice

        # Need index of sacrificed star by this point
        sacrificedStarTime = (
            t_dets[sacrificeIndex] + overheadTime
        )  # saves time being sacrificed
        sInds = np.delete(sInds, sacrificeIndex)
        t_dets = np.delete(t_dets, sacrificeIndex)
        return sInds, t_dets, sacrificedStarTime

    def calcTinit(self, sInds, TL, fZ, fEZ, WA, mode, TK=None):
        """Calculate Initial Values for starkAYO (use max C/T)
        <- this is a poor IC selection"""
        cachefname = self.cachefname + "maxCbyTt0"  # Generate cache Name

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached maxCbyTt0 from %s" % cachefname)
            try:
                with open(cachefname, "rb") as ff:
                    maxCbyTtime = pickle.load(ff)
            except UnicodeDecodeError:
                with open(cachefname, "rb") as ff:
                    maxCbyTtime = pickle.load(ff, encoding="latin1")
            return maxCbyTtime[sInds]

        self.vprint("Calculating maxCbyTt0")
        # This contains the time maxCbyT occurs at
        maxCbyTtime = np.zeros(sInds.shape[0])
        # this contains the value of maxCbyT
        maxCbyT = np.zeros(sInds.shape[0])  # noqa: F841

        # Solve Initial Integration Times
        def CbyTfunc(t_dets, self, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp, TK=None):
            CbyT = (
                -self.Completeness.comp_per_intTime(
                    t_dets * u.d, TL, sInds, fZ, fEZ, WA, mode, C_b=Cb, C_sp=Csp, TK=TK
                )
                / t_dets
                * u.d
            )
            return CbyT.value

        # Calculate Maximum C/T
        for i in range(sInds.shape[0]):
            x0 = 0.5
            retVals = scipy.optimize.fmin(
                CbyTfunc,
                x0,
                args=(
                    self,
                    TL,
                    sInds[i],
                    fZ[i],
                    fEZ,
                    WA[i],
                    mode,
                    self.Cb[i],
                    self.Csp[i],
                ),
                xtol=1e-8,
                ftol=1e-8,
                disp=True,
            )
            maxCbyTtime[i] = retVals[0]
            self.vprint(
                "Max C/T calc completion: "
                + str(float(i) / sInds.shape[0])
                + " "
                + str(maxCbyTtime[i])
            )
        # Sept 27, Execution time 101 seconds for 651 stars

        with open(cachefname, "wb") as fo:
            pickle.dump(maxCbyTtime, fo)
            self.vprint("Saved cached 1st year Tinit to %s" % cachefname)
        return maxCbyTtime
