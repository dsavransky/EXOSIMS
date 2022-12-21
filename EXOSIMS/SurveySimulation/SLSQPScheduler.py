from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize, minimize_scalar
import os
import pickle
from astropy.time import Time


class SLSQPScheduler(SurveySimulation):
    """SLSQPScheduler

    This class implements a continuous optimization of integration times
    using the scipy minimize function with method SLSQP.  ortools with the CBC
    linear solver is used to find an initial solution consistent with the constraints.
    For details see Keithly et al. 2019. Alternatively: Savransky et al. 2017 (SPIE).

    Args:
        **specs:
            user specified values

    Notes:
        Due to the time costs of the current comp_per_inttime calculation in
        GarrettCompleteness this should be used with BrownCompleteness.

        Requires ortools

    """

    def __init__(
        self,
        cacheOptTimes=False,
        staticOptTimes=False,
        selectionMetric="maxC",
        Izod="current",
        maxiter=60,
        ftol=1e-3,
        **specs
    ):  # fZminObs=False,

        # initialize the prototype survey
        SurveySimulation.__init__(self, **specs)

        # Calculate fZmax
        self.valfZmax, self.absTimefZmax = self.ZodiacalLight.calcfZmax(
            np.arange(self.TargetList.nStars),
            self.Observatory,
            self.TargetList,
            self.TimeKeeping,
            list(
                filter(
                    lambda mode: mode["detectionMode"],
                    self.OpticalSystem.observingModes,
                )
            )[0],
            self.cachefname,
        )

        assert isinstance(staticOptTimes, bool), "staticOptTimes must be boolean."
        self.staticOptTimes = staticOptTimes
        self._outspec["staticOptTimes"] = self.staticOptTimes

        assert isinstance(cacheOptTimes, bool), "cacheOptTimes must be boolean."
        self._outspec["cacheOptTimes"] = cacheOptTimes

        # Informs what selection metric to use
        assert selectionMetric in [
            "maxC",
            "Izod-Izodmin",
            "Izod-Izodmax",
            "(Izod-Izodmin)/(Izodmax-Izodmin)",
            "(Izod-Izodmin)/(Izodmax-Izodmin)/CIzod",
            "TauIzod/CIzod",
            "random",
            "priorityObs",
        ], "selectionMetric not valid input"
        self.selectionMetric = selectionMetric
        self._outspec["selectionMetric"] = self.selectionMetric

        # Informs what Izod to optimize integration times for
        # [fZmin, fZmin+45d, fZ0, fZmax, current]
        assert Izod in [
            "fZmin",
            "fZ0",
            "fZmax",
            "current",
        ], "Izod not valid input"
        self.Izod = Izod
        self._outspec["Izod"] = self.Izod

        # maximum number of iterations to optimize integration times for
        assert isinstance(maxiter, int), "maxiter is not an int"
        assert maxiter >= 1, "maxiter must be positive real"
        self.maxiter = maxiter
        self._outspec["maxiter"] = self.maxiter

        # tolerance to place on optimization
        assert isinstance(ftol, float), "ftol must be boolean"
        assert ftol > 0, "ftol must be positive real"
        self.ftol = ftol
        self._outspec["ftol"] = self.ftol

        # some global defs
        self.detmode = list(
            filter(
                lambda mode: mode["detectionMode"],
                self.OpticalSystem.observingModes,
            )
        )[0]
        self.ohTimeTot = (
            self.Observatory.settlingTime + self.detmode["syst"]["ohTime"]
        )  # total overhead time per observation
        self.maxTime = (
            self.TimeKeeping.missionLife * self.TimeKeeping.missionPortion
        )  # total mission time

        self.constraints = {
            "type": "ineq",
            "fun": lambda x: self.maxTime.to(u.d).value
            - np.sum(x[x * u.d > 0.1 * u.s])
            - np.sum(x * u.d > 0.1 * u.s).astype(float)  # maxTime less sum of intTimes
            * self.ohTimeTot.to(u.d).value,  # sum of True -> goes to 1 x OHTime
            "jac": lambda x: np.ones(len(x)) * -1.0,
        }

        self.t0 = None
        if cacheOptTimes:
            # Generate cache Name
            cachefname = self.cachefname + "t0"

            if os.path.isfile(cachefname):
                self.vprint("Loading cached t0 from %s" % cachefname)
                with open(cachefname, "rb") as f:
                    try:
                        self.t0 = pickle.load(f)
                    except UnicodeDecodeError:
                        self.t0 = pickle.load(f, encoding="latin1")
                sInds = np.arange(self.TargetList.nStars)
                fZ = (
                    np.array([self.ZodiacalLight.fZ0.value] * len(sInds))
                    * self.ZodiacalLight.fZ0.unit
                )
                self.sint_comp = -self.objfun(self.t0.to("day").value, sInds, fZ)

        if self.t0 is None:
            # 1. find nominal background counts for all targets in list
            int_dMag = 25.0  # this works fine for WFIRST
            _, Cbs, Csps = self.OpticalSystem.Cp_Cb_Csp(
                self.TargetList,
                np.arange(self.TargetList.nStars),
                self.ZodiacalLight.fZ0,
                self.ZodiacalLight.fEZ0,
                int_dMag,
                self.TargetList.int_WA,
                self.detmode,
                TK=self.TimeKeeping,
            )

            # find baseline solution with intCutoff_dMag-based integration times
            # 3.
            t0 = self.OpticalSystem.calc_intTime(
                self.TargetList,
                np.arange(self.TargetList.nStars),
                self.ZodiacalLight.fZ0,
                self.ZodiacalLight.fEZ0,
                self.TargetList.int_dMag,
                self.TargetList.int_WA,
                self.detmode,
                TK=self.TimeKeeping,
            )
            # 4.
            int_comp = self.Completeness.comp_per_intTime(
                t0,
                self.TargetList,
                np.arange(self.TargetList.nStars),
                self.ZodiacalLight.fZ0,
                self.ZodiacalLight.fEZ0,
                self.TargetList.int_WA,
                self.detmode,
                C_b=Cbs,
                C_sp=Csps,
                TK=self.TimeKeeping,
            )

            # 5. Formulating MIP to filter out stars we can't or don't want to
            # reasonably observe
            solver = pywraplp.Solver(
                "SolveIntegerProblem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )  # create solver instance
            xs = [
                solver.IntVar(0.0, 1.0, "x" + str(j)) for j in np.arange(len(int_comp))
            ]  # define x_i variables for each star either 0 or 1
            self.vprint("Finding baseline fixed-time optimal target set.")

            # constraint is x_i*t_i < maxtime
            constraint = solver.Constraint(
                -solver.infinity(), self.maxTime.to(u.day).value
            )  # hmmm I wonder if we could set this to 0,maxTime
            for j, x in enumerate(xs):
                constraint.SetCoefficient(
                    x, t0[j].to("day").value + self.ohTimeTot.to(u.day).value
                )  # this forms x_i*(t_0i+OH) for all i

            # objective is max x_i*comp_i
            objective = solver.Objective()
            for j, x in enumerate(xs):
                objective.SetCoefficient(x, int_comp[j])
            objective.SetMaximization()

            # this line enables output of the CBC MIXED INTEGER PROGRAM
            # (Was hard to find don't delete)
            # solver.EnableOutput()
            solver.SetTimeLimit(5 * 60 * 1000)  # time limit for solver in milliseconds
            cpres = solver.Solve()  # noqa: F841 actually solve MIP
            x0 = np.array([x.solution_value() for x in xs])  # convert output solutions

            self.sint_comp = np.sum(int_comp * x0)  # calculate sum Comp from MIP
            self.t0 = t0  # assign calculated t0

            # Observation num x0=0 @ int_dMag=25 is 1501
            # Observation num x0=0 @ int_dMag=30 is 1501...

            # now find the optimal eps baseline and use whichever gives you the highest
            # starting completeness
            self.vprint("Finding baseline fixed-eps optimal target set.")

            def totCompfeps(eps):
                compstars, tstars, x = self.inttimesfeps(
                    eps, Cbs.to("1/d").value, Csps.to("1/d").value
                )
                return -np.sum(compstars * x)

            # Note: There is no way to seed an initial solution to minimize scalar
            # 0 and 1 are supposed to be the bounds on epsres.
            # I could define upper bound to be 0.01, However defining the bounds to be
            # 5 lets the solver converge
            epsres = minimize_scalar(
                totCompfeps,
                method="bounded",
                bounds=[0, 7],
                options={"disp": 3, "xatol": self.ftol, "maxiter": self.maxiter},
            )
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-
            #                       bounded.html#optimize-minimize-scalar-bounded
            comp_epsmax, t_epsmax, x_epsmax = self.inttimesfeps(
                epsres["x"], Cbs.to("1/d").value, Csps.to("1/d").value
            )
            if np.sum(comp_epsmax * x_epsmax) > self.sint_comp:
                x0 = x_epsmax
                self.sint_comp = np.sum(comp_epsmax * x_epsmax)
                self.t0 = t_epsmax * x_epsmax * u.day

            # Optimize the baseline solution
            self.vprint("Optimizing baseline integration times.")
            sInds = np.arange(self.TargetList.nStars)
            if self.Izod == "fZ0":  # Use fZ0 to calculate integration times
                fZ = (
                    np.array([self.ZodiacalLight.fZ0.value] * len(sInds))
                    * self.ZodiacalLight.fZ0.unit
                )
            elif self.Izod == "fZmin":  # Use fZmin to calculate integration times
                fZ = self.valfZmin[sInds]
            elif self.Izod == "fZmax":  # Use fZmax to calculate integration times
                fZ = self.valfZmax[sInds]
            elif (
                self.Izod == "current"
            ):  # Use current fZ to calculate integration times
                fZ = self.ZodiacalLight.fZ(
                    self.Observatory,
                    self.TargetList,
                    sInds,
                    self.TimeKeeping.currentTimeAbs.copy()
                    + np.zeros(self.TargetList.nStars) * u.d,
                    self.detmode,
                )

            (
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
            ) = self.TimeKeeping.get_ObsDetectionMaxIntTime(
                self.Observatory, self.detmode, self.TimeKeeping.currentTimeNorm.copy()
            )
            maxIntTime = min(
                maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
            )  # Maximum intTime allowed
            bounds = [(0, maxIntTime.to(u.d).value) for i in np.arange(len(sInds))]
            initguess = x0 * self.t0.to(u.d).value
            self.save_initguess = initguess

            # While we use all sInds as input, theoretically, this can be solved faster
            # if we use the following lines:
            # sInds = np.asarray([sInd for sInd in sInds if np.bool(x0[sInd])])
            # bounds = [(0,maxIntTime.to(u.d).value) for i in np.arange(len(sInds))]
            # and use initguess[sInds], fZ[sInds], and self.t0[sInds].
            # There was no noticable performance improvement
            ires = minimize(
                self.objfun,
                initguess,
                jac=self.objfun_deriv,
                args=(sInds, fZ),
                constraints=self.constraints,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": self.maxiter, "ftol": self.ftol, "disp": True},
            )  # original method

            assert ires["success"], "Initial time optimization failed."

            self.t0 = ires["x"] * u.d
            self.sint_comp = -ires["fun"]

            if cacheOptTimes:
                with open(cachefname, "wb") as f:
                    pickle.dump(self.t0, f)
                self.vprint("Saved cached optimized t0 to %s" % cachefname)

        # Redefine filter inds
        self.intTimeFilterInds = np.where(
            (self.t0.value > 0.0)
            * (self.t0.value <= self.OpticalSystem.intCutoff.value)
            > 0.0
        )[
            0
        ]  # These indices are acceptable for use simulating

    def inttimesfeps(self, eps, Cb, Csp):
        """
        Compute the optimal subset of targets for a given epsilon value
        where epsilon is the maximum completeness gradient.

        Everything is in units of days
        """

        tstars = (
            -Cb * eps * np.sqrt(np.log(10.0))
            + np.sqrt((Cb * eps) ** 2.0 * np.log(10.0) + 5.0 * Cb * Csp**2.0 * eps)
        ) / (
            2.0 * Csp**2.0 * eps * np.log(10.0)
        )  # calculating Tau to achieve dC/dT #double check

        compstars = self.Completeness.comp_per_intTime(
            tstars * u.day,
            self.TargetList,
            np.arange(self.TargetList.nStars),
            self.ZodiacalLight.fZ0,
            self.ZodiacalLight.fEZ0,
            self.TargetList.int_WA,
            self.detmode,
            C_b=Cb / u.d,
            C_sp=Csp / u.d,
            TK=self.TimeKeeping,
        )

        solver = pywraplp.Solver(
            "SolveIntegerProblem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        xs = [solver.IntVar(0.0, 1.0, "x" + str(j)) for j in np.arange(len(compstars))]
        constraint = solver.Constraint(-solver.infinity(), self.maxTime.to(u.d).value)

        for j, x in enumerate(xs):
            constraint.SetCoefficient(x, tstars[j] + self.ohTimeTot.to(u.day).value)

        objective = solver.Objective()
        for j, x in enumerate(xs):
            objective.SetCoefficient(x, compstars[j])
        objective.SetMaximization()
        # this line enables output of the CBC MIXED INTEGER PROGRAM
        # solver.EnableOutput()
        solver.SetTimeLimit(5 * 60 * 1000)  # time limit for solver in milliseconds

        _ = solver.Solve()
        # self.vprint(solver.result_status())

        x = np.array([x.solution_value() for x in xs])
        # self.vprint('Solver is FEASIBLE: ' + str(solver.FEASIBLE))
        # self.vprint('Solver is OPTIMAL: ' + str(solver.OPTIMAL))
        # self.vprint('Solver is BASIC: ' + str(solver.BASIC))

        return compstars, tstars, x

    def objfun(self, t, sInds, fZ):
        """
        Objective Function for SLSQP minimization. Purpose is to maximize summed
        completeness

        Args:
            t (ndarray):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t * u.d >= 0.1 * u.s  # inds that were not downselected by initial MIP

        comp = self.Completeness.comp_per_intTime(
            t[good] * u.d,
            self.TargetList,
            sInds[good],
            fZ[good],
            self.ZodiacalLight.fEZ0,
            self.TargetList.int_WA[sInds][good],
            self.detmode,
        )
        # self.vprint(-comp.sum()) # for verifying SLSQP output
        return -comp.sum()

    def objfun_deriv(self, t, sInds, fZ):
        """
        Jacobian of objective Function for SLSQP minimization.

        Args:
            t (astropy Quantity):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t * u.d >= 0.1 * u.s  # inds that were not downselected by initial MIP

        tmp = (
            self.Completeness.dcomp_dt(
                t[good] * u.d,
                self.TargetList,
                sInds[good],
                fZ[good],
                self.ZodiacalLight.fEZ0,
                self.TargetList.int_WA[sInds][good],
                self.detmode,
                TK=self.TimeKeeping,
            )
            .to("1/d")
            .value
        )

        jac = np.zeros(len(t))
        jac[good] = tmp
        return -jac

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """
        Given a subset of targets, calculate their integration times given the
        start of observation time.

        This implementation updates the optimized times based on current conditions and
        mission time left.

        Note: next_target filter will discard targets with zero integration times.

        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.
                must be of the same size as sInds
            mode (dict):
                Selected observing mode for detection

        Returns:
            astropy Quantity array:
                Integration times for detection. Same dimension as sInds
        """

        if self.staticOptTimes:
            intTimes = self.t0[sInds]
        else:
            # assumed values for detection
            if self.Izod == "fZ0":  # Use fZ0 to calculate integration times
                fZ = (
                    np.array([self.ZodiacalLight.fZ0.value] * len(sInds))
                    * self.ZodiacalLight.fZ0.unit
                )
            elif self.Izod == "fZmin":  # Use fZmin to calculate integration times
                fZ = self.valfZmin[sInds]
            elif self.Izod == "fZmax":  # Use fZmax to calculate integration times
                fZ = self.valfZmax[sInds]
            elif (
                self.Izod == "current"
            ):  # Use current fZ to calculate integration times
                fZ = self.ZodiacalLight.fZ(
                    self.Observatory, self.TargetList, sInds, startTimes, mode
                )

            # instead of actual time left, try bounding by maxTime - detection time used
            # need to update time used in choose_next_target

            timeLeft = (
                self.TimeKeeping.missionLife - self.TimeKeeping.currentTimeNorm.copy()
            ) * self.TimeKeeping.missionPortion
            bounds = [(0, timeLeft.to(u.d).value) for i in np.arange(len(sInds))]

            initguess = self.t0[sInds].to(u.d).value
            ires = minimize(
                self.objfun,
                initguess,
                jac=self.objfun_deriv,
                args=(sInds, fZ),
                constraints=self.constraints,
                method="SLSQP",
                bounds=bounds,
                options={"disp": True, "maxiter": self.maxiter, "ftol": self.ftol},
            )

            # update default times for these targets
            self.t0[sInds] = ires["x"] * u.d

            intTimes = ires["x"] * u.d

        intTimes[intTimes < 0.1 * u.s] = 0.0 * u.d

        return intTimes

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """

        Given a subset of targets (pre-filtered by method next_target or some
        other means), select the best next one.

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (integer):
                    Index of next target star
                waitTime (astropy Quantity):
                    the amount of time to wait (this method returns None)

        """
        # Do Checking to Ensure There are Targetswith Positive Nonzero Integration Time
        # tmpsInds = sInds
        sInds = sInds[
            np.where(intTimes.value > 1e-10)[0]
        ]  # filter out any intTimes that are essentially 0
        intTimes = intTimes[intTimes.value > 1e-10]

        # calcualte completeness values for current intTimes
        if self.Izod == "fZ0":  # Use fZ0 to calculate integration times
            fZ = (
                np.array([self.ZodiacalLight.fZ0.value] * len(sInds))
                * self.ZodiacalLight.fZ0.unit
            )
        elif self.Izod == "fZmin":  # Use fZmin to calculate integration times
            fZ = self.valfZmin[sInds]
        elif self.Izod == "fZmax":  # Use fZmax to calculate integration times
            fZ = self.valfZmax[sInds]
        elif self.Izod == "current":  # Use current fZ to calculate integration times
            fZ = self.ZodiacalLight.fZ(
                self.Observatory,
                self.TargetList,
                sInds,
                self.TimeKeeping.currentTimeAbs.copy() + slewTimes[sInds],
                self.detmode,
            )
        comps = self.Completeness.comp_per_intTime(
            intTimes,
            self.TargetList,
            sInds,
            fZ,
            self.ZodiacalLight.fEZ0,
            self.TargetList.int_WA[sInds],
            self.detmode,
            TK=self.TimeKeeping,
        )

        # Selection Metric Type
        valfZmax = self.valfZmax[sInds]
        valfZmin = self.valfZmin[sInds]
        if self.selectionMetric == "maxC":  # A choose target with maximum completeness
            sInd = np.random.choice(sInds[comps == max(comps)])
        elif (
            self.selectionMetric == "Izod-Izodmin"
        ):  # B choose target closest to its fZmin
            selectInd = np.argmin(fZ - valfZmin)
            sInd = sInds[selectInd]
        elif (
            self.selectionMetric == "Izod-Izodmax"
        ):  # C choose target furthest from fZmax
            selectInd = np.argmin(
                fZ - valfZmax
            )  # this is most negative when fZ is smallest
            sInd = sInds[selectInd]
        elif (
            self.selectionMetric == "(Izod-Izodmin)/(Izodmax-Izodmin)"
        ):  # D choose target closest to fZmin with largest fZmin-fZmax variation
            selectInd = np.argmin(
                (fZ - valfZmin) / (valfZmin - valfZmax)
            )  # this is most negative when fZ is smallest
            sInd = sInds[selectInd]
        elif (
            self.selectionMetric == "(Izod-Izodmin)/(Izodmax-Izodmin)/CIzod"
        ):  # E = D + current completeness at intTime optimized at
            selectInd = np.argmin(
                (fZ - valfZmin) / (valfZmin - valfZmax) * (1.0 / comps)
            )
            sInd = sInds[selectInd]
        # F is simply E but where comp is calculated sing fZmin
        # elif self.selectionMetric == '(Izod-Izodmin)/(Izodmax-Izodmin)/CIzodmin':
        # # F = D + current completeness at Izodmin and intTime
        #     selectInd = np.argmin((fZ - valfZmin)/(valfZmin - valfZmax)*(1./comps))
        #     sInd = sInds[selectInd]
        elif self.selectionMetric == "TauIzod/CIzod":  # G maximum C/T
            selectInd = np.argmin(intTimes / comps)
            sInd = sInds[selectInd]
        elif self.selectionMetric == "random":  # I random selection of available
            sInd = np.random.choice(sInds)
        elif self.selectionMetric == "priorityObs":  # Advances time to
            # Apply same filters as in next_target (the issue here is that we might
            # want to make a target observation that is currently in keepout so we need
            # to "add back in those targets")
            sInds = np.arange(self.TargetList.nStars)
            sInds = sInds[np.where(self.t0.value > 1e-10)[0]]
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)
            sInds = self.revisitFilter(sInds, self.TimeKeeping.currentTimeNorm.copy())

            TK = self.TimeKeeping

            # Pick which absTime
            # We will readjust self.absTimefZmin later
            # we have to do this because "self.absTimefZmin does not support
            # item assignment"
            tmpabsTimefZmin = list()
            for i in np.arange(len(self.fZQuads)):
                fZarr = np.asarray(
                    [
                        self.fZQuads[i][j][1].value
                        for j in np.arange(len(self.fZQuads[i]))
                    ]
                )  # create array of fZ for the Target Star
                fZarrInds = np.where(
                    np.abs(fZarr - self.valfZmin[i].value) < 0.000001 * np.min(fZarr)
                )[0]

                dt = self.t0[
                    i
                ]  # amount to subtract from points just about to enter keepout
                # Extract fZ Type
                assert not len(fZarrInds) == 0, "This should always be greater than 0"
                if len(fZarrInds) == 2:
                    fZminType0 = self.fZQuads[i][fZarrInds[0]][0]
                    fZminType1 = self.fZQuads[i][fZarrInds[1]][0]
                    if (
                        fZminType0 == 2 and fZminType1 == 2
                    ):  # Ah! we have a local minimum fZ!
                        # which one occurs next?
                        tmpabsTimefZmin.append(
                            self.whichTimeComesNext(
                                [
                                    self.fZQuads[i][fZarrInds[0]][3],
                                    self.fZQuads[i][fZarrInds[1]][3],
                                ]
                            )
                        )
                    elif (fZminType0 == 0 and fZminType1 == 1) or (
                        fZminType0 == 1 and fZminType1 == 0
                    ):  # we have an entering and exiting or exiting and entering
                        if fZminType0 == 0:  # and fZminType1 == 1
                            tmpabsTimefZmin.append(
                                self.whichTimeComesNext(
                                    [
                                        self.fZQuads[i][fZarrInds[0]][3] - dt,
                                        self.fZQuads[i][fZarrInds[1]][3],
                                    ]
                                )
                            )
                        else:  # fZminType0 == 1 and fZminType1 == 0
                            tmpabsTimefZmin.append(
                                self.whichTimeComesNext(
                                    [
                                        self.fZQuads[i][fZarrInds[0]][3],
                                        self.fZQuads[i][fZarrInds[1]][3] - dt,
                                    ]
                                )
                            )
                    elif (
                        fZminType1 == 2 or fZminType0 == 2
                    ):  # At least one is local minimum
                        if fZminType0 == 2:
                            tmpabsTimefZmin.append(
                                self.whichTimeComesNext(
                                    [
                                        self.fZQuads[i][fZarrInds[0]][3] - dt,
                                        self.fZQuads[i][fZarrInds[1]][3],
                                    ]
                                )
                            )
                        else:  # fZminType1 == 2
                            tmpabsTimefZmin.append(
                                self.whichTimeComesNext(
                                    [
                                        self.fZQuads[i][fZarrInds[0]][3],
                                        self.fZQuads[i][fZarrInds[1]][3] - dt,
                                    ]
                                )
                            )
                    else:  # Throw error
                        raise Exception(
                            "A fZminType was not assigned or handled correctly 1"
                        )
                elif len(fZarrInds) == 1:
                    fZminType0 = self.fZQuads[i][fZarrInds[0]][0]
                    if fZminType0 == 2:  # only 1 local fZmin
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3])
                    elif fZminType0 == 0:  # entering
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3] - dt)
                    elif fZminType0 == 1:  # exiting
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3])
                    else:  # Throw error
                        raise Exception(
                            "A fZminType was not assigned or handled correctly 2"
                        )
                elif len(fZarrInds) == 3:
                    # Not entirely sure why 3 is occuring. Looks like entering,
                    # exiting, and local minima exist.... strange
                    tmpdt = list()
                    for k in np.arange(3):
                        if self.fZQuads[i][fZarrInds[k]][0] == 0:
                            tmpdt.append(dt)
                        else:
                            tmpdt.append(0.0 * u.d)
                    tmpabsTimefZmin.append(
                        self.whichTimeComesNext(
                            [
                                self.fZQuads[i][fZarrInds[0]][3] - tmpdt[0],
                                self.fZQuads[i][fZarrInds[1]][3] - tmpdt[1],
                                self.fZQuads[i][fZarrInds[2]][3] - tmpdt[2],
                            ]
                        )
                    )
                elif len(fZarrInds) >= 4:
                    raise Exception("Unexpected Error: Number of fZarrInds was 4")
                    # might check to see if local minimum and koentering/exiting
                    # happened
                elif len(fZarrInds) == 0:
                    raise Exception("Unexpected Error: Number of fZarrInds was 0")

            # reassign
            tmpabsTimefZmin = Time(
                np.asarray([tttt.value for tttt in tmpabsTimefZmin]),
                format="mjd",
                scale="tai",
            )
            self.absTimefZmin = tmpabsTimefZmin

            # Time relative to now where fZmin occurs
            timeWherefZminOccursRelativeToNow = (
                self.absTimefZmin.value - TK.currentTimeAbs.copy().value
            )  # of all targets
            indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < 0))[
                0
            ]  # find all inds that are less than 0
            cnt = 0.0
            # iterate until we know the next time in the future where fZmin occurs
            # for all targets
            while len(indsLessThan0) > 0:
                cnt += 1.0
                # take original and add 365.25 until we get the right number of
                # years to add
                timeWherefZminOccursRelativeToNow[indsLessThan0] = (
                    self.absTimefZmin.copy().value[indsLessThan0]
                    - TK.currentTimeAbs.copy().value
                    + cnt * 365.25
                )
                indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < 0))[0]
            # contains all "next occuring" fZmins in absolute time
            timeToStartfZmins = timeWherefZminOccursRelativeToNow

            timefZminAfterNow = [
                timeToStartfZmins[i] for i in sInds
            ]  # filter by times in future and times not filtered
            timeToAdvance = np.min(
                np.asarray(timefZminAfterNow)
            )  # find the minimum time

            tsInds = np.where((timeToStartfZmins == timeToAdvance))[
                0
            ]  # find the index of the minimum time and return that sInd
            tsInds = [i for i in tsInds if i in sInds]
            if len(tsInds) > 1:
                sInd = tsInds[0]
            else:
                sInd = tsInds[0]
            del timefZminAfterNow

            # Advance To fZmin of Target
            _ = self.TimeKeeping.advanceToAbsTime(
                Time(
                    timeToAdvance + TK.currentTimeAbs.copy().value,
                    format="mjd",
                    scale="tai",
                ),
                False,
            )

            # Check if exoplanetObsTime would be exceeded
            OS = self.OpticalSystem
            Obs = self.Observatory
            TK = self.TimeKeeping
            allModes = OS.observingModes
            mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
            (
                maxIntTimeOBendTime,
                maxIntTimeExoplanetObsTime,
                maxIntTimeMissionLife,
            ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
            maxIntTime = min(
                maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
            )  # Maximum intTime allowed
            intTimes2 = self.calc_targ_intTime(sInd, TK.currentTimeAbs.copy(), mode)
            if (
                intTimes2 > maxIntTime
            ):  # check if max allowed integration time would be exceeded
                self.vprint("max allowed integration time would be exceeded")
                sInd = None
                # waitTime = 1.0 * u.d
        # H is simply G but where comp and intTime are calculated using fZmin
        # elif self.selectionMetric == 'TauIzodmin/CIzodmin':
        # #H maximum C at fZmin / T at fZmin

        if sInd is not None:
            # We assume any observation with integration time of less than 1 second
            # is not a valid integration time
            if self.t0[sInd] < 1.0 * u.s:
                self.vprint("sInd to None is: " + str(sInd))
                sInd = None

        return sInd, None

    def arbitrary_time_advancement(self, dt):
        """Handles fully dynamically scheduled case where OBduration is infinite and
        missionPortion is less than 1.
        Input dt is the total amount of time, including all overheads and extras
        used for the previous observation.
        """
        if self.selectionMetric == "priorityObs":
            pass
        else:
            self.TimeKeeping.allocate_time(
                dt
                * (1.0 - self.TimeKeeping.missionPortion)
                / self.TimeKeeping.missionPortion,
                addExoplanetObsTime=False,
            )

    def whichTimeComesNext(self, absTs):
        """Determine which absolute time comes next from current time
        Specifically designed to determine when the next local zodiacal light event
        occurs form fZQuads

        Args:
            absTs (list):
                the absolute times of different events (list of absolute times)

        Returns:
            astropy time quantity:
                the absolute time which occurs next
        """
        TK = self.TimeKeeping
        # Convert Abs Times to norm Time
        tabsTs = list()
        for i in np.arange(len(absTs)):
            tabsTs.append(
                (absTs[i] - TK.missionStart).value
            )  # all should be in first year
        tSinceStartOfThisYear = TK.currentTimeNorm.copy().value % 365.25
        if len(tabsTs) == len(
            np.where(tSinceStartOfThisYear < np.asarray(tabsTs))[0]
        ):  # time strictly less than all absTs
            absT = absTs[np.argmin(tabsTs)]
        elif len(tabsTs) == len(
            np.where(tSinceStartOfThisYear > np.asarray(tabsTs))[0]
        ):
            absT = absTs[np.argmin(tabsTs)]
        else:  # Some are above and some are below
            tmptabsTsInds = np.where(tSinceStartOfThisYear < np.asarray(tabsTs))[0]
            absT = absTs[
                np.argmin(np.asarray(tabsTs)[tmptabsTsInds])
            ]  # of times greater than current time, returns smallest

        return absT
