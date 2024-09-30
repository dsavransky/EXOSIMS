from EXOSIMS.Observatory.SotoStarshade_SKi import SotoStarshade_SKi
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import scipy.optimize as optimize
from scipy.integrate import solve_bvp
from copy import deepcopy
import time
import os
import pickle

EPS = np.finfo(float).eps


class SotoStarshade_ContThrust(SotoStarshade_SKi):
    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """

    def __init__(self, orbit_datapath=None, **specs):

        SotoStarshade_SKi.__init__(self, **specs)

        # to convert from dimensionalized to normalized, (Dimension) / self.(Dimension)U
        self.mass = 10930 * u.kg

        Tmax = 1040 * u.mN
        Isp = 3000 * u.s
        ve = const.g0 * Isp

        self.Tmax = Tmax
        self.ve = self.convertVel_to_canonical(ve)

        # smoothing factor (eps = 1 means min energy, eps = 0 means min fuel)
        self.epsilon = 1

        self.lagrangeMults = self.lagrangeMult()

    # =============================================================================
    # Miscellaneous
    # =============================================================================
    def DCM_r2i(self, t):
        """Direction cosine matrix to rotate from Rotating Frame to Inertial Frame

        Finds rotation matrix for positions and velocities (6x6)

        Args:
            t (float):
                Rotation angle

        Returns:
            float 6x6 array:
                rotation of full 6 dimensional state from R to I frame
        """

        Ts = np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

        dTs = np.array(
            [[-np.sin(t), -np.cos(t), 0], [np.cos(t), -np.sin(t), 0], [0, 0, 0]]
        )

        Qu = np.hstack([Ts, np.zeros([3, 3])])
        Ql = np.hstack([dTs, Ts])

        rotMatrix = np.vstack([Qu, Ql])

        return rotMatrix

    def DCM_i2r(self, t):
        """Direction cosine matrix to rotate from Inertial Frame to Rotating Frame

        Finds rotation matrix for positions and velocities (6x6)

        Args:
            t (float):
                Rotation angle

        Returns:
            float 6x6 array:
                rotation of full 6 dimensional state from I to R frame
        """

        Ts = np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

        dTs = np.array(
            [[-np.sin(t), -np.cos(t), 0], [np.cos(t), -np.sin(t), 0], [0, 0, 0]]
        )

        argD = np.matmul(np.matmul(-Ts.T, dTs), Ts.T)
        Qu = np.hstack([Ts.T, np.zeros([3, 3])])
        Ql = np.hstack([argD, Ts.T])

        return np.vstack([Qu, Ql])

    def DCM_r2i_9(self, t):
        """Direction cosine matrix to rotate from Rotating Frame to Inertial Frame

        Finds rotation matrix for positions, velocities, and accelerations (9x9)

        Args:
            t (float):
                Rotation angle

        Returns:
            float 9x9 array:
                rotation of full 9 dimensional state from R to I frame
        """

        Ts = np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

        dTs = np.array(
            [[-np.sin(t), -np.cos(t), 0], [np.cos(t), -np.sin(t), 0], [0, 0, 0]]
        )

        ddTs = -np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 0]]
        )

        Qu = np.hstack([Ts, np.zeros([3, 6])])
        Qm = np.hstack([dTs, Ts, np.zeros([3, 3])])
        Ql = np.hstack([ddTs, 2 * dTs, Ts])

        return np.vstack([Qu, Qm, Ql])

    def lagrangeMult(self):
        """Generate a random lagrange multiplier for initial guess (6x1)

        Generates an array of 6 numbers with two pairs of 3 coordinates describing
        position on a sphere. The first two represent the x and y positions on a sphere
        with random radius between 1 and 5, and the last coordinate represents the z
        coordinate scaled by the radius.

        Returns:
            ~numpy.ndarray(float):
               6x1 Random lagrange multipliers for an initial guess
        """

        Lr = np.random.uniform(1, 5)
        Lv = np.random.uniform(1, 5)

        alpha_r = np.random.uniform(0, 2 * np.pi)
        alpha_v = np.random.uniform(0, 2 * np.pi)

        delta_r = np.random.uniform(-np.pi / 2, np.pi / 2)
        delta_v = np.random.uniform(-np.pi / 2, np.pi / 2)

        # TODO: seems odd that 3rd and 6th coordinate aren't part of a spherical
        # coordinate

        L = np.array(
            [
                Lr * np.cos(alpha_r) * np.cos(delta_r),
                Lr * np.sin(alpha_r) * np.cos(delta_r),
                np.sin(delta_r),
                Lv * np.cos(alpha_v) * np.cos(delta_v),
                Lv * np.sin(alpha_v) * np.cos(delta_v),
                np.sin(delta_v),
            ]
        )

        return L

    def newStar_angularSep(self, TL, iStars, jStars, currentTime, dt):
        """Finds angular separation from old star to given list of stars

        This method returns the angular separation from the last observed
        star to all others on the given list at the currentTime. This is distinct
        from the prototype version of the method in its signing of the
        angular separation based on halo velocity direction

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            iStars (~numpy.ndarray(int)):
                Integer indices of originating stars
            jStars (~numpy.ndarray(int)):
                Integer indices of destination stars
                Observation times for targets.
            currentTime (~astropy.time.Time(~numpy.ndarray)):
                Current absolute mission time in MJD
            dt (float):
                Timestep in units of days for determining halo velocity frame

        Returns:
            float:
                Angular separation between two target stars
        """

        nStars = len(iStars)

        # time in canonical units
        absTimes = currentTime + np.array([0, dt]) * u.d
        t = (
            self.convertTime_to_canonical(
                np.mod(absTimes.value, self.equinox.value) * u.d
            )
            * u.rad
        )

        # halo positions and velocities
        haloPos = self.haloPosition(absTimes) + np.array([1, 0, 0]) * self.L2_dist.to(
            "au"
        )
        haloVel = self.haloVelocity(absTimes)

        # halo positions and velocities in canonical units
        r_T0_R = np.array(
            [self.convertPos_to_canonical(haloPos[:, n]) for n in range(3)]
        )
        Rdr_T0_R = np.array(
            [self.convertVel_to_canonical(haloVel[:, n]) for n in range(3)]
        )

        # V-frame
        v1_R = Rdr_T0_R / np.linalg.norm(Rdr_T0_R, axis=0)

        # Position of stars
        coords = TL.coords
        beta = coords.lat
        lamb = coords.lon
        dist = self.convertPos_to_canonical(coords.distance[0])

        # unit vector to star locations
        u_i0_R = np.array(
            [
                [np.cos(beta[iStars]) * np.cos(lamb[iStars] - t[0])],
                [np.cos(beta[iStars]) * np.sin(lamb[iStars] - t[0])],
                [np.sin(beta[iStars])],
            ]
        ).reshape(3, nStars)

        u_j0_R = np.array(
            [
                [np.cos(beta[jStars]) * np.cos(lamb[jStars] - t[-1])],
                [np.cos(beta[jStars]) * np.sin(lamb[jStars] - t[-1])],
                [np.sin(beta[jStars])],
            ]
        ).reshape(3, nStars)

        # star locations to relative to telescope
        r_iT_R = dist * u_i0_R - r_T0_R[:, 0].reshape(3, 1)
        r_jT_R = dist * u_j0_R - r_T0_R[:, -1].reshape(3, 1)

        # unit vector from Telescope to star i
        u_iT = r_iT_R / np.linalg.norm(r_iT_R, axis=0)

        # unit vector from Telescope to star j
        u_jT = r_jT_R / np.linalg.norm(r_jT_R, axis=0)

        # direction of travel from star i to j
        u_ji = u_jT - u_iT

        # sign of the angular separation, + in direction of halo velocity
        # (NOTE: only using halo velocity at t0, not t0+dt)
        dot_sgn = np.matmul(v1_R[:, 0].T, u_ji)
        sgn = np.array([np.sign(x) if x != 0 else 1 for x in dot_sgn])

        # calculating angular separation and assigning signs
        dot_product = np.array([np.dot(a.T, b) for a, b in zip(u_iT.T, u_jT.T)])
        dot_product = np.clip(dot_product, -1, 1)
        psi = sgn * (np.arccos(dot_product) * u.rad).to("deg").value

        return psi

    def star_angularSepDesiredDist(self, psiPop, nSamples=1e6, angBinSize=3):
        """Rejection sample from psiPop to achieve nSamples fitting logistic
        distribution

        Args:
            psiPop (array float):
                List of floats between -180 and 180
            nSamples (int):
                Number of samples to return
            angBinSize (float):
                Size of a bin for the histogram used in rejection sampling

        Returns:
            tuple:
                array float:
                    Angles (nSamples) fitting the logistic distribution
                array int:
                    An array of indices of original psiPop that were accepted

        """

        # distribution of the full psi population
        N = len(psiPop)
        psiBins = np.arange(-180, 181, 3)
        envelopeDist, EnvDbins = np.histogram(psiPop, psiBins, density=True)

        # Desired Distribution pdf
        s = 32
        desiredDist = 1 / (4 * s) * 1 / (np.cosh(psiBins[:-1] / (2 * s)) ** 2)

        # Scaling factor
        M = np.max(desiredDist / envelopeDist)

        # how many samples from the full data set are we taking?
        samplingSize = int(nSamples) if (nSamples > 0) & (nSamples <= N) else N

        # randomly selecting from the population (no replacing, so unique samples)
        sampling_inds = np.random.choice(N, samplingSize, replace=False)

        # an empty array to which we'll add accepted indices
        final_inds = np.array([], dtype=int)

        # running rejection sampling
        for n in sampling_inds:

            # figuring out which bin to put this particular ang sep in
            corresponding_Bin = np.where(psiBins <= psiPop[n])[0][-1]

            # calculating pdf ratio
            ratio = desiredDist[corresponding_Bin] / (
                M * envelopeDist[corresponding_Bin]
            )

            # random number from 0 to 1
            rand = np.random.uniform(0, 1)

            # sample has been accepted!
            if rand < ratio:
                final_inds = np.hstack([final_inds, n])

        psiFiltered = psiPop[final_inds]

        return psiFiltered, final_inds

    def selectPairsOfStars(self, TL, nPairs, currentTime, dt, nSamples):
        """Rejection sampling of star pairings using desired distribution and sampling
        from that final distribution

        Args:
            TL (TargetList module):
                TargetList class object
            nPairs (int):
                Number of pairs to produce
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            dt (float):
                Timestep in units of days for determining halo velocity frame
            nSamples (int):
                Number of samples to be used when generating random stars for pairing
        Returns:
            tuple:
                int list:
                    The list of indices corresponding to starting stars in TargetList
                int list:
                    The list of indices corresponding to the ending stars in TargetList
                float array:
                    Angular distance between pairs from iFinal and jFinal

        """

        psiBins = np.arange(-180, 181, 3)
        nPairs -= len(psiBins) - 1

        # randomly selected pairs of stars
        iLog = np.random.randint(0, TL.nStars, nSamples)
        jLog = np.random.randint(0, TL.nStars, nSamples)

        # angular separation between each of these pairs
        # star i -> at currentTime
        # star j -> at currentTime + dt
        psi = self.newStar_angularSep(TL, iLog, jLog, currentTime, dt)

        # filtering pairings list, creating desired distribution using rejection
        # sampling
        filtered_psi, filtered_inds = self.star_angularSepDesiredDist(psi, nSamples)

        # sampling nPairs of stars from the final distribution
        samples = np.random.choice(len(filtered_inds), nPairs, replace=False)
        sampling_inds = filtered_inds[samples]

        # non-sampled star pairings
        leftover_psi = np.delete(psi, sampling_inds)
        leftover_inds = np.delete(np.arange(0, nSamples), sampling_inds)

        # grab a star from each bin and add it to the list
        psiFinal = np.array([], dtype=int)

        for lb, ub in zip(psiBins[:-1], psiBins[1:]):
            find_one = np.where((leftover_psi >= lb) & (leftover_psi < ub))[0]
            if find_one.size > 0:
                sampling_inds = np.hstack([sampling_inds, leftover_inds[find_one[0]]])
                psiFinal = np.hstack([psiFinal, leftover_psi[find_one[0]]])

        # results
        iFinal = iLog[sampling_inds]
        jFinal = jLog[sampling_inds]

        # add the sampled stars from our desired distributions
        psiFinal = np.hstack([psiFinal, filtered_psi[samples]])

        return iFinal, jFinal, psiFinal

    # =============================================================================
    # Helper functions
    # =============================================================================

    def determineThrottle(self, state):
        """Determines throttle based on instantaneous switching function value.

        Typically being used during collocation algorithms. A zero-crossing of
        the switching function is highly unlikely between the large number of
        nodes.

        Args:
            state (float array):
                State vector and lagrange multipliers
        Returns:
            float array:
                Throttle between 0 and 1, and in the same shape as state
        """

        eps = self.epsilon
        n = 1 if state.size == 14 else state.shape[1]

        throttle = np.zeros(n)
        S = self.switchingFunction(state)
        S = S.reshape(n)

        for i, s in enumerate(S):
            if eps > 0:
                midthrottle = (eps - s) / (2 * eps)
                throttle[i] = 0 if s > eps else 1 if s < -eps else midthrottle
            else:
                throttle[i] = 0 if s > eps else 1

        return throttle

    def switchingFunction(self, state):
        """Evaluates the switching function at specific states.

        Args:
            state (float array):
                State vector and lagrange multipliers
        Returns:
            float:
                Value of the switching function

        """

        x, y, z, dx, dy, dz, m, L1, L2, L3, L4, L5, L6, L7 = state

        Lv_, lv = self.unitVector(np.array([L4, L5, L6]))

        S = -lv * self.ve / m - L7 + 1

        return S

    def switchingFunctionDer(self, state):
        """Evaluates the time derivative of the switching function.

        Switching function derivative evaluated for specific states.

        Args:
            state (float array):
                State vector and lagrange multipliers
        Returns:
            float:
                Value of the switching function time derivative

        """
        ve = self.ve
        n = 1 if state.size == 14 else state.shape[1]
        x, y, z, dx, dy, dz, m, L1, L2, L3, L4, L5, L6, L7 = state

        Lr = np.array([L1, L2, L3]).reshape(3, n)
        Lv = np.array([L4, L5, L6]).reshape(3, n)
        Lv_, lv = self.unitVector(Lv)

        Pv_arr = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
        Pv = np.dstack([Pv_arr] * n)

        PLdot = np.vstack([np.dot(a.T, b) for a, b in zip(Pv.T, Lv.T)]).T

        dS = (
            -(ve / m)
            * np.vstack([np.dot(a.T, b) for a, b in zip((-Lr - PLdot).T, Lv_.T)]).T
        )

        return dS

    def selectEventFunctions(self, s0):
        """Selects the proper event function for integration.

        This method calculates the switching function and its derivative at a
        single specific state. It then determines which thrust case it will be
        in: full, medium, or no thrust. If the value of the switching function
        is within a certain tolerance of the boundaries, it uses the derivative
        to determine the direction it is heading in. Then the proper event
        functions are created for the integrator to determine the next crossing
        (i.e. the next case change).

        Args:
            s0 (float array):
                Iniitial state vector and lagrange multipliers
        Returns:
            tuple:
                function list:
                    A list of functions which are the switching function adjusted by
                    1e-10.
                    These take in an unused first parameter, and then the state.
                    Depending on the case (2), multiple functions are returned.
                    These functions will return 0 when the switching function crosses
                    1e-10 or -1e-10.
                int:
                    An integer 0,1,2 representing whether the switching
                    function of the initial state is within a 1e-10 tolerance of the
                    of zero (2), less than -1e-10 (1), or greater than 1e-10 (0).
        """
        eps = self.epsilon

        S = self.switchingFunction(s0)
        dS = self.switchingFunctionDer(s0)[0]

        # finding which case we are in:
        #   - case 2 if       -eps < S < eps       (medium thrust)
        #   - case 1 if   S < -eps                 (full thrust)
        #   - case 0 if                  eps < S   (no thrust)

        case = 0 if S > eps else 1 if S < -eps else 2

        # checking to see if S is within a certain tolerance from epsilon
        withinTol = np.abs((np.abs(S) - eps)) < 1e-10
        # determine if there is a case error if within tolerance
        if withinTol:
            # not the minimum fuel case
            if eps != 0:
                # at the upper bound, case determined by derivative
                if S > 0:
                    case = 2 if dS < 0 else 0
                # at the lower bound, case determined by derivative
                else:
                    case = 2 if dS > 0 else 1
            # minimum fuel case, only two cases
            else:
                case = 0 if dS > 0 else 1

        eventFunctions = []
        CrossingUpperBound = lambda t, s: self.switchingFunction(s) - eps
        CrossingLowerBound = lambda t, s: self.switchingFunction(s) + eps

        CrossingUpperBound.terminal = True
        CrossingLowerBound.terminal = True

        if case == 0:
            # crossing upper epsilon from above
            CrossingUpperBound.direction = -1
            # appending event function
            eventFunctions.append(CrossingUpperBound)
        elif case == 1:
            # crossing lower epsilon from below
            CrossingLowerBound.direction = 1
            # appending event function
            eventFunctions.append(CrossingLowerBound)
        else:
            # can either cross lower epsilon from above or upper from below
            CrossingLowerBound.direction = -1
            CrossingUpperBound.direction = 1
            # appending event function
            eventFunctions.append(CrossingUpperBound)
            eventFunctions.append(CrossingLowerBound)

        return eventFunctions, case

    # =============================================================================
    # Equations of Motion and Boundary Conditions
    # =============================================================================
    def boundary_conditions_thruster(self, sA, sB, constrained=False):
        """Creates boundary conditions for solving a boundary value problem

        Function used in scipy solve_bvp function call. Returns residuals

        Args:
            sA (6 or 7 array):
                To be compared with the initial state for boundary condition.
            sB (6 or 7 array):
                To be compared with the final state for boundary condition.
            constrained (boolean):
                Whether there are 6 (false) or 7 (true) boundary conditions.
        Returns:
            array:
                Returns the residuals of the boundary conditions/constraint equation.
        """

        BCo1 = sA[0] - self.sA[0]
        BCo2 = sA[1] - self.sA[1]
        BCo3 = sA[2] - self.sA[2]
        BCo4 = sA[3] - self.sA[3]
        BCo5 = sA[4] - self.sA[4]
        BCo6 = sA[5] - self.sA[5]

        BCf1 = sB[0] - self.sB[0]
        BCf2 = sB[1] - self.sB[1]
        BCf3 = sB[2] - self.sB[2]
        BCf4 = sB[3] - self.sB[3]
        BCf5 = sB[4] - self.sB[4]
        BCf6 = sB[5] - self.sB[5]

        if constrained:
            BCo7 = sA[6] - self.sA[6]
            BCf7 = sB[-1]
            BC = np.array(
                [
                    BCo1,
                    BCo2,
                    BCo3,
                    BCo4,
                    BCo5,
                    BCo6,
                    BCo7,
                    BCf1,
                    BCf2,
                    BCf3,
                    BCf4,
                    BCf5,
                    BCf6,
                    BCf7,
                ]
            )
        else:
            BC = np.array(
                [BCo1, BCo2, BCo3, BCo4, BCo5, BCo6, BCf1, BCf2, BCf3, BCf4, BCf5, BCf6]
            )

        return BC

    def EoM_Adjoint(self, t, state, constrained=False, amax=False, integrate=False):
        """Equations of Motion with costate vectors

        Equations of motion for CR3BP with costates for control.

        Args:
            t (float):
                Currently unused
            state (array):
                The state and lagrange multipliers.
            constrained (boolean):
                Currently unused
            amax (float or boolean):
                Maximum acceleration attainable or False otherwise
            integrate (boolean):
                If true, the array is flattened for integration.

        Returns:
            array:
                The time derivatives of the states and costates.
        """

        mu = self.mu
        ve = self.ve

        n = 1 if state.size == 14 else state.shape[1]

        if amax:
            x, y, z, dx, dy, dz, m, L1, L2, L3, L4, L5, L6, L7 = state
        else:
            x, y, z, dx, dy, dz, L1, L2, L3, L4, L5, L6 = state

        if integrate:
            state = state.T

        # vector distances from primaries
        r1 = np.array([x - (-mu), y, z])
        r2 = np.array([x - (1 - mu), y, z])
        # norms of distances from primaries
        R1 = np.linalg.norm(r1, axis=0)
        R2 = np.linalg.norm(r2, axis=0)

        # Position-dependent acceleration terms
        qx = np.array(
            [x - (1 - mu) * (x + mu) / R1**3 - mu * (x + mu - 1) / R2**3]
        ).reshape(1, n)
        qy = np.array([y - (1 - mu) * y / R1**3 - mu * y / R2**3]).reshape(1, n)
        qz = np.array([-(1 - mu) * z / R1**3 - mu * z / R2**3]).reshape(1, n)
        q = np.vstack([qx, qy, qz])  # shape of 3xn

        # Position partial derivatives
        Q11 = (
            1
            - (1 - mu) / R1**3
            + 3 * (1 - mu) * (x + mu) ** 2 / R1**5
            - mu / R2**3
            + 3 * mu * (x + mu - 1) ** 2 / R2**5
        )
        Q22 = (
            1
            - (1 - mu) / R1**3
            + 3 * (1 - mu) * y**2 / R1**5
            - mu / R2**3
            + 3 * mu * y**2 / R2**5
        )
        Q33 = (
            -(1 - mu) / R1**3
            + 3 * (1 - mu) * z**2 / R1**5
            - mu / R2**3
            + 3 * mu * z**2 / R2**5
        )
        Q12 = 3 * (1 - mu) * (x + mu) * y / R1**5 + 3 * mu * (x + mu - 1) * y / R2**5
        Q13 = 3 * (1 - mu) * (x + mu) * z / R1**5 + 3 * mu * (x + mu - 1) * z / R2**5
        Q23 = 3 * (1 - mu) * y * z / R1**5 + 3 * mu * y * z / R2**5
        Qr = np.array([[Q11, Q12, Q13], [Q12, Q22, Q23], [Q13, Q23, Q33]]).reshape(
            3, 3, n
        )  # shape of 3x3xn

        # Velocity-dependent acceleration terms
        px = 2 * dy
        py = -2 * dx
        pz = np.zeros([1, n])
        p = np.vstack([px, py, pz])  # shape of 3xn

        # Velocity partial derivatives
        Pv_arr = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
        Pv = np.dstack([Pv_arr] * n)

        # Costate vectors
        Lr = np.vstack([L1, L2, L3])
        Lv = np.vstack([L4, L5, L6])
        Lr_, lr = self.unitVector(Lr)
        Lv_, lv = self.unitVector(Lv)

        # ================================================
        # Equations of Motion
        # ================================================
        dX = np.vstack([dx, dy, dz])
        dV = q + p
        dLx = -np.vstack([np.dot(a.T, b) for a, b in zip(Qr.T, Lv.T)]).T
        dLv = -Lr - np.vstack([np.dot(a.T, b) for a, b in zip(Pv.T, Lv.T)]).T

        if amax:
            # throttle factor
            throttle = self.determineThrottle(state)

            dV -= Lv_ * amax * throttle / m
            dm = -throttle * amax / ve
            dLm = -lv * throttle * amax / m**2
            # putting them all together, a 14xn array
            f = np.vstack([dX, dV, dm, dLx, dLv, dLm])
        else:
            dV -= Lv
            # putting them all together, a 12xn array
            f = np.vstack([dX, dV, dLx, dLv])

        if integrate:
            f = f.flatten()

        return f

    def starshadeBoundaryVelocity(self, TL, sInd, currentTime, SRP=True):
        """Calculates starshade and telescope velocities in R- or I-frames during
        stationkeeping

        Calculates the rotating system position and velocity of the
        starshade at insertion into the optimal initial state for
        observation of the given star.

        Args:
            TL (TargetList module):
                Target List
            sInd (int):
                Index of star for observation in the target list
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            SRP (boolean):
                Whether or not solar radiation pressure should be used
                in calculating the insertion state for optimal
                starshade stationkeeping observation position.
        Returns:
            tuple:
                array:
                    Position in the rotating frame
                array:
                    Velocity in the rotating frame
        """

        # absolute times (Note: equinox is start time of Halo AND when inertial frame
        # and rotating frame match)
        modTimes = (
            np.mod(currentTime.value, self.equinox.value) * u.d
        )  # mission times relative to equinox )
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonical units

        r_S0_I, Iv_S0_I, qq, qq, qq, qq = self.starshadeKinematics(
            TL, sInd, currentTime
        )

        # star a at t=ta
        r_AS_C, Iv_AS_C = self.starshadeInjectionVelocity(
            TL, sInd, currentTime, SRP=SRP
        )
        s_AS_C = np.hstack([r_AS_C, Iv_AS_C])
        r_AS_I, Iv_AS_I = self.rotateComponents2NewFrame(
            TL, sInd, currentTime, s_AS_C, np.array([0]), SRP=SRP, final_frame="I"
        )

        rQi__tA = self.rot(t[0].value, 3)
        r_A0_I = r_S0_I + r_AS_I
        Iv_A0_I = Iv_S0_I + Iv_AS_I

        r_A0_R = np.dot(rQi__tA, r_A0_I)
        Rv_A0_I = deepcopy(Iv_A0_I)
        Rv_A0_I[0] += r_A0_I[1]
        Rv_A0_I[1] -= r_A0_I[0]
        Rv_A0_R = np.dot(rQi__tA, Rv_A0_I)

        return r_A0_R, Rv_A0_R

    # =============================================================================
    # Initial conditions
    # =============================================================================

    def findInitialTmax(self, TL, nA, nB, tA, dt, m0=1, s_init=np.array([])):
        """Finding initial guess for starting Thrust

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            nA (int):
                The index for the starting star in the target list
            nB (int):
                The index for the final star in the target list
            tA (~astropy.time.Time(~numpy.ndarray)):
                Initial absolute mission time in MJD
            dt (~astropy.time.Time(~numpy.ndarray)):
                A time step between the two voundary conditions
            m0 (float):
                Initial mass
            s_init (~numpy.ndarray):
                An initial guess for the state

        Returns:
            tuple:
                ~astropy.units.quantity.Quantity:
                    The maximum initial thrust (force units).
                ~numpy.ndarray:
                    States from the solved bvp.
                ~numpy.ndarray:
                    Times at corrsponding to each state.
        """

        tB = tA + dt
        # angle,uA,uB,r_tscp = self.lookVectors(TL,nA,nB,tA,tB)

        # position vector of occulter in heliocentric frame
        # self_rA = uA*self.occulterSep.to('au').value + r_tscp[ 0]
        # self_rB = uB*self.occulterSep.to('au').value + r_tscp[-1]

        r_A0_R, Rv_A0_R = self.starshadeBoundaryVelocity(TL, nA, tA)
        r_B0_R, Rv_B0_R = self.starshadeBoundaryVelocity(TL, nB, tB)

        Rs_A0_R = np.hstack([r_A0_R[:, 0], Rv_A0_R[:, 0]])
        Rs_B0_R = np.hstack([r_B0_R[:, 0], Rv_B0_R[:, 0]])

        Rsf_A0_R = np.hstack([Rs_A0_R, self.lagrangeMult()])
        Rsf_B0_R = np.hstack([Rs_B0_R, self.lagrangeMult()])

        a = ((np.mod(tA.value, self.equinox.value) * u.d)).to("yr") / u.yr * (2 * np.pi)
        b = ((np.mod(tB.value, self.equinox.value) * u.d)).to("yr") / u.yr * (2 * np.pi)

        # running collocation
        tGuess = np.hstack([a, b]).value

        if s_init.size:
            sGuess = np.vstack([s_init[:, 0], s_init[:, -1]])
        else:
            sGuess = np.vstack([Rsf_A0_R, Rsf_B0_R])

        s, t_s, status = self.send_it_thruster(sGuess.T, tGuess, m0=m0, verbose=False)

        lv = s[9:, :]

        aNorms0 = np.linalg.norm(lv, axis=0)
        aMax0 = self.convertAcc_to_dim(np.max(aNorms0)).to("m/s^2")
        Tmax0 = (aMax0 * self.mass).to("N")

        return Tmax0, s, t_s

    def findTmaxGrid(self, TL, tA, dtRange):
        """Create grid of Tmax values using unconstrained thruster

        This method is used purely for creating figures.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            tA (~astropy.time.Time(~numpy.ndarray)):
                Initial absolute mission time in MJD
            dtRange (~astropy.time.Time(~numpy.ndarray)):
                An array of delta times to consider

        Returns:
            ~astropy.units.quantity.Quantity:
                Max thrust in force units (dtRange dimensions by TL.nStars)
        """

        midInt = int(np.floor((TL.nStars - 1) / 2))

        sInds = np.arange(0, TL.nStars)
        ang = self.star_angularSep(TL, midInt, sInds, tA)
        sInd_sorted = np.argsort(ang)
        angles = ang[sInd_sorted].to("deg").value

        TmaxMap = np.zeros([len(dtRange), len(angles)]) * u.N

        for i, t in enumerate(dtRange):
            for j, n in enumerate(sInd_sorted):
                print(i, j)
                Tmax, s, t_s = self.findInitialTmax(TL, midInt, n, tA, t)
                TmaxMap[i, j] = Tmax

        return TmaxMap

    # =============================================================================
    # BVP solvers
    # =============================================================================

    def send_it_thruster(
        self,
        sGuess,
        tGuess,
        aMax=False,
        constrained=False,
        m0=1,
        maxNodes=1e5,
        verbose=False,
    ):
        """Solving generic bvp from t0 to tF using states and costates

        Uses the Scipy solve_bvp method.

        Args:
            sGuess (array):
                Initial state and costate guess
            tGuess (astropy Time array):
                Times corresponding to the guess of the state at each time
            aMax (astropy Newton or boolean):
                Maximum attainable acceleration
            constrained (boolean):
                Flag for whether this is constrained or unconstrained problem
                This determines dimensions of the state inputs.
            m0 (float):
                Initial mass
            maxNodes (int):
                Maximum number of nodes to use in the BVP problem solution
            verbose (boolean):
                Flag passed to solve_bvp
        Returns:
            tuple:
                array:
                    State and costate at the various mesh times
                array:
                    Corresponding times to the sampled states.
                boolean:
                    Status returned by the bvp_solve method
        """

        sG = sGuess
        # unconstrained problem begins with 12 states, rather than 14. checking for that
        if len(sGuess) == 12:
            x, y, z, dx, dy, dz, L1, L2, L3, L4, L5, L6 = sGuess
            # if unconstrained is initial guess for constrained problem, make 14 state
            # array
            if aMax:
                mRange = np.linspace(m0, 0.8, len(x))
                lmRange = np.linspace(1, 0, len(x))
                sG = np.vstack(
                    [x, y, z, dx, dy, dz, mRange, L1, L2, L3, L4, L5, L6, lmRange]
                )

        # only saves initial and final desired states if first solving unconstrained
        # problem
        if not constrained:
            self.sA = np.hstack([sG[0:6, 0], m0, sG[6:, 0], 1])
            self.sB = np.hstack([sG[0:6, -1], 0.8, sG[6:, -1], 0])
            self.sG = sG

        # creating equations of motion and boundary conditions functions
        EoM = lambda t, s: self.EoM_Adjoint(t, s, constrained, aMax)
        BC = lambda t, s: self.boundary_conditions_thruster(t, s, constrained)
        # solving BVP
        sol = solve_bvp(
            EoM, BC, tGuess, sG, tol=1e-8, max_nodes=int(maxNodes), verbose=0
        )

        if verbose:
            self.vprint(sol.message)
        # saving results
        s = sol.y
        t_s = sol.x

        return s, t_s, sol.status

    def collocate_Trajectory(self, TL, nA, nB, tA, dt):
        """Solves minimum energy and minimum fuel cases for continuous thrust

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            nA (int):
                The index for the starting star in the target list
            nB (int):
                The index for the final star in the target list
            tA (~astropy.time.Time(~numpy.ndarray)):
                Initial absolute mission time in MJD
            dt (~astropy.time.Time(~numpy.ndarray)):
                A time step between the two voundary conditions

        Returns:
            tuple:
                ~numpy.ndarray:
                    Trajectory
                ~numpy.ndarray:
                    Times corresponding to trajectory
                float:
                    last epsilon that fully converged (2 if minimum energy didn't work)
                    Parameterizes minimum energy to minimum fuel solution.
                ~astropy.units.quantity.Quantity:
                    Range of thrusts (Newtons) considered.
        """

        # initializing arrays
        stateLog = []
        timeLog = []

        # solving using unconstrained thruster as initial guess
        Tmax, sTmax, tTmax = self.findInitialTmax(TL, nA, nB, tA, dt)
        aMax = self.convertAcc_to_canonical((Tmax / self.mass).to("m/s^2"))
        # saving results
        stateLog.append(sTmax)
        timeLog.append(tTmax)

        # all thrusts were successful
        e_best = 2
        s_best = deepcopy(stateLog)
        t_best = deepcopy(timeLog)

        # thrust values
        desiredT = self.Tmax.to("N").value
        currentT = Tmax.value
        # range of thrusts to try
        TmaxRange = np.linspace(currentT, desiredT, 30)

        # range of epsilon values to try (e=1 is minimum energy, e=0 is minimum fuel)
        epsilonRange = np.round(np.arange(1, -0.1, -0.1), decimals=1)

        # Huge loop over all thrust and epsilon values:
        # we start at the minimum energy case, e=1, using the thrust value from the
        # unconstrained solution
        #     In/De-crement the thrust until we reach the desired thrust level
        #   then we decrease e and repeat process until we get to e=0 (minimum fuel)
        #   saves the last successful result in case collocation fails

        # loop over epsilon starting with e=1
        for j, e in enumerate(epsilonRange):
            print("Collocate Epsilon = ", e)
            # initialize epsilon
            self.epsilon = e

            # loop over thrust values from current to desired thrusts
            for i, thrust in enumerate(TmaxRange):
                # print("Thrust #",i," / ",len(TmaxRange))
                # convert thrust to canonical acceleration
                aMax = self.convertAcc_to_canonical(
                    (thrust * u.N / self.mass).to("m/s^2")
                )
                # retrieve state and time initial guesses
                sGuess = stateLog[i]
                tGuess = timeLog[i]
                # perform collocation
                s, t_s, status = self.send_it_thruster(
                    sGuess, tGuess, aMax, constrained=True, maxNodes=1e5, verbose=False
                )

                # collocation failed, exits out of everything
                if status != 0:
                    self.epsilon = e_best
                    if e_best == 2:
                        # if only the unconstrained problem worked, still returns a
                        # 14 length array
                        s_out = []
                        length = s_best[0].shape[1]
                        m = np.linspace(1, 0.9, length)
                        lm = np.linspace(0.3, 0, length)
                        s_out.append(np.vstack([s_best[0][:6], m, s_best[0][6:], lm]))
                        s_best = deepcopy(s_out)
                    return s_best, t_best, e_best, TmaxRange

                # collocation was successful!
                if j == 0:
                    # creates log of state and time results for next thrust iteration
                    # (at the beginning of the loop)
                    stateLog.append(s)
                    timeLog.append(t_s)
                else:
                    # updates log of state and time results for next thrust iteration
                    stateLog[i] = s
                    timeLog[i] = t_s

            # all thrusts were successful, save results
            e_best = self.epsilon
            s_best = deepcopy(stateLog)
            t_best = deepcopy(timeLog)

        return s_best, t_best, e_best, TmaxRange

    def collocate_Trajectory_minEnergy(self, TL, nA, nB, tA, dt, m0=1):
        """Solves minimum energy and minimum fuel cases for continuous thrust

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            nA (int):
                The index for the starting star in the target list
            nB (int):
                The index for the final star in the target list
            tA (~astropy.time.Time(~numpy.ndarray)):
                Initial absolute mission time in MJD
            dt (~astropy.time.Time(~numpy.ndarray)):
                A time step between the two voundary conditions
            m0 (float):
                Initial mass


        Returns:
            tuple:
                ~numpy.ndarray:
                    Trajectory
                ~numpy.ndarray:
                    Times corresponding to trajectory
                float:
                    last epsilon that fully converged (2 if minimum energy didn't work)
                    Parameterizes minimum energy to minimum fuel solution.
                ~astropy.units.quantity.Quantity:
                    Range of thrusts (Newtons) considered.

        """

        # initializing arrays
        stateLog = []
        timeLog = []

        # solving using unconstrained thruster as initial guess
        Tmax, sTmax, tTmax = self.findInitialTmax(TL, nA, nB, tA, dt, m0)
        aMax = self.convertAcc_to_canonical((Tmax / self.mass).to("m/s^2"))
        # saving results
        stateLog.append(sTmax)
        timeLog.append(tTmax)

        # all thrusts were successful
        e_best = 2
        s_best = deepcopy(sTmax)
        t_best = deepcopy(tTmax)

        # thrust values
        desiredT = self.Tmax.to("N").value
        currentT = Tmax.value
        # range of thrusts to try
        TmaxRange = np.linspace(currentT, desiredT, 30)

        # Huge loop over all thrust and epsilon values:
        #   we start at the minimum energy case, e=1, using the thrust value from
        #   the unconstrained solution
        #     In/De-crement the thrust until we reach the desired thrust level
        #   then we decrease e and repeat process until we get to e=0 (minimum fuel)
        #   saves the last successful result in case collocation fails

        # loop over epsilon starting with e=1
        self.epsilon = 1

        # loop over thrust values from current to desired thrusts
        for i, thrust in enumerate(TmaxRange):
            # print("Thrust #",i," / ",len(TmaxRange))
            # convert thrust to canonical acceleration
            aMax = self.convertAcc_to_canonical((thrust * u.N / self.mass).to("m/s^2"))
            # retrieve state and time initial guesses
            sGuess = stateLog[i]
            tGuess = timeLog[i]
            # perform collocation
            s, t_s, status = self.send_it_thruster(
                sGuess,
                tGuess,
                aMax,
                constrained=True,
                m0=m0,
                maxNodes=1e5,
                verbose=False,
            )

            # collocation failed, exits out of everything
            if status != 0:
                self.epsilon = e_best
                return s_best, t_best, e_best, TmaxRange

            # creates log of state and time results for next thrust iteration
            # (at the beginning of the loop)
            stateLog.append(s)
            timeLog.append(t_s)

        # all thrusts were successful, save results
        e_best = self.epsilon
        s_best = deepcopy(s)
        t_best = deepcopy(t_s)

        return s_best, t_best, e_best, TmaxRange

    # =============================================================================
    # Shooting Algorithms
    # =============================================================================

    def integrate_thruster(self, sGuess, tGuess, Tmax, verbose=False):
        """Integrates thruster trajectory with thrust case switches

        This methods integrates an initial guess for the spacecraft state
        forwards in time. It uses event functions to find the next zero of the
        switching function which means a new thrust case is needed (full,
        medium or no thrust).

        Args:
            sGuess (~numpy.ndarray):
                Initial state and costate guess
            tGuess (~astropy.time.Time(~numpy.ndarray)):
                Times corresponding to the guess of the state at each time
            Tmax (~astropy.units.quantity.Quantity):
                Maximum thrust attainable
            verbose (bool):
                Toggle verbosity (defaults False)

        Returns:
            tuple:
                ~numpy.ndarray:
                    Trajectory states
                ~astropy.time.Time(~numpy.ndarray):
                    Times corresponding to states
        """
        s0 = sGuess[:, 0]
        t0 = tGuess[0]
        tF = tGuess[-1]

        # initializing starting time
        tC = deepcopy(t0)

        # converting Tmax to a canonical acceleration
        Tmax = Tmax.to("N").value
        aMax = self.convertAcc_to_canonical((Tmax * u.N / self.mass).to("m/s^2"))

        # establishing equations of motion
        EoM = lambda t, s: self.EoM_Adjoint(
            t, s, constrained=True, amax=aMax, integrate=True
        )

        # starting integration
        count = 0
        while tC < tF:
            # selecting the switch functions with correct boundaries
            switchFunctions, case = self.selectEventFunctions(s0)
            if verbose:
                print("[%.3f / %.3f] with case %d" % (tC, tF, case))
            # running integration with event functions
            res = solve_ivp(EoM, [tC, tF], s0, events=switchFunctions)

            # saving final integration time, if greater than tF,
            # we completed the trajectory
            tC = deepcopy(res.t[-1])
            s0 = deepcopy(res.y[:, -1])

            # saving results in a new array
            if count == 0:
                sLog = deepcopy(res.y)
                tLog = deepcopy(res.t)
            # adding to the results log if there was a previous thrust case switch
            else:
                sLog = np.hstack([sLog, res.y])
                tLog = np.hstack([tLog, res.t])
            count += 1

        return sLog, tLog

    def conFun_singleShoot(self, w, t0, tF, Tmax, returnLog=False):
        """Objective Function for single shooting thruster

        Args:
            w (~numpy.ndarray):
                Costate vector
            t0 (~astropy.time.Time):
                Initial time
            tF (~astropy.time.Time):
                Final time
            Tmax (~astropy.units.quantity.Quantity):
                Maximum thrust attainable
            returnLog (boolean):
                Return the states and times of the solution (default false

        Returns:
            tuple:
                float:
                    Norm of difference between current state and boundary value
                ~numpy.ndarray:
                    Trajectory states
                ~astropy.time.Time(~numpy.ndarray):
                    Times corresponding to states
        """

        sInit = np.hstack([self.sA[:7], w]).reshape(14, 1)
        tGuess = np.array([t0, tF])

        sLog, tLog = self.integrate_thruster(sInit, tGuess, Tmax)

        f = self.boundary_conditions_thruster(sLog[:, 0], sLog[:, -1], constrained=True)
        fnorm = np.linalg.norm(f)

        if returnLog:
            return fnorm, sLog, tLog
        else:
            return fnorm

    def minimize_TerminalState(self, s_best, t_best, Tmax, method):
        """Minimizes boundary conditions for thruster

        Args:
            s_best (array):
                Initial state and costate
            t_best (astropy Time array):
                Times corresponding to the guess of the state at each time
            Tmax (astropy force):
                Maximum thrust attainable
            method (string):
                Optimization method for Scipy minimize call

        Returns:
            tuple:
                float:
                    Norm of difference between current state and boundary value
                array:
                    Trajectory states
                astropy Time array:
                    Times corresponding to states
        """

        w0 = s_best[7:, 0]
        t0 = t_best[0]
        tF = t_best[-1]

        res = optimize.minimize(
            self.conFun_singleShoot,
            w0,
            method=method,
            tol=1e-12,
            args=(
                t0,
                tF,
                Tmax,
            ),
        )
        # minimizer_kwargs = {"method":method,"args":(t0,tF,Tmax,)}
        # res = optimize.basinhopping(self.conFun_singleShoot,w0,
        #                             minimizer_kwargs=minimizer_kwargs)
        #
        fnorm, sLog, tLog = self.conFun_singleShoot(res.x, t0, tF, Tmax, returnLog=True)

        return fnorm, sLog, tLog

    def singleShoot_Trajectory(
        self, stateLog, timeLog, e_best, TmaxRange, method="SLSQP"
    ):
        """Perform single shooting to solve the boundary value problem.

        Args:
            stateLog (array):
                Approximate trajectory typically determined by collocation
            timeLog (astropy Time array):
                Corresponging time values for the trajectory
            e_best (float):
                Epsilon value corresponding to previous approxmiate trajectory
            TmaxRange (astropy Newton array):
                Range of thrusts (Newtons) considered.
            method (string):
                Optimization method for Scipy minimize call

        Returns:
            tuple:
                array:
                    Trajectory states
                astropy Time array:
                    Times corresponding to states
                float:
                    Epsilon value determining how fuel vs energy optimal the trajectory
                    is.
        """

        # initializing arrays
        s_best = deepcopy(stateLog)
        t_best = deepcopy(timeLog)

        incLogFlag = True if len(stateLog) != len(TmaxRange) else False

        # range of epsilon values to try (e=1 is minimum energy, e=0 is minimum fuel)
        e_best = 1 if e_best == 2 else e_best
        epsilonRange = np.round(np.arange(e_best, -0.1, -0.1), decimals=1)

        # Huge loop over all thrust and epsilon values, just like collocation method

        # loop over epsilon starting with e=1
        for j, e in enumerate(epsilonRange):
            print("SS Epsilon = ", e)
            # initialize epsilon
            self.epsilon = e

            # loop over thrust values from current to desired thrusts
            for i, thrust in enumerate(TmaxRange):
                # print("Thrust #",i," / ",len(TmaxRange))
                # retrieve state and time initial guesses
                sGuess = stateLog[i]
                tGuess = timeLog[i]
                # perform single shooting
                fnorm, sLog, tLog = self.minimize_TerminalState(
                    sGuess, tGuess, thrust, method
                )
                print(fnorm)
                # single shooting failed, exits out of everything
                if fnorm > 1e-7:
                    self.epsilon = e_best
                    return s_best, t_best, e_best

                # single shooting was successful!
                if incLogFlag:
                    # appends stateLog if the input was incomplete
                    stateLog.append(sLog)
                    timeLog.append(tLog)
                else:
                    # updates log of state and time results for next thrust iteration
                    stateLog[i] = sLog
                    timeLog[i] = tLog

            # all thrusts were successful, save results
            e_best = self.epsilon
            s_best = deepcopy(sLog)
            t_best = deepcopy(tLog)

        return s_best, t_best, e_best

    # =============================================================================
    #  Putting it al together
    # =============================================================================
    def calculate_dMmap(self, TL, tA, dtRange, filename):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file.

        Args:
            TL (TargetList module):
                Target list of stars
            tA (astropy Time array):
                Initial absolute mission time in MJD
            dtRange (astropy Time array):
                Transfer times to try
            filename (string):
                File name to store the cached data.
        """

        sInds = np.arange(0, TL.nStars)
        ang = self.star_angularSep(TL, 0, sInds, tA)
        sInd_sorted = np.argsort(ang)
        angles = ang[sInd_sorted].to("deg").value

        dtFlipped = np.flipud(dtRange)

        self.dMmap = np.zeros([len(dtRange), len(angles)])
        self.eMap = np.zeros([len(dtRange), len(angles)])

        tic = time.perf_counter()
        for j, n in enumerate(sInd_sorted):
            for i, t in enumerate(dtFlipped):
                print(i, j)
                s_coll, t_coll, e_coll, TmaxRange = self.collocate_Trajectory(
                    TL, 0, n, tA, t
                )

                if e_coll != 0:
                    s_ssm, t_ssm, e_ssm = self.singleShoot_Trajectory(
                        s_coll, t_coll, e_coll, TmaxRange * u.N
                    )

                if e_ssm == 2 and t.value < 30:
                    break

                m = s_ssm[-1][6, :]
                dm = m[-1] - m[0]
                self.dMmap[i, j] = dm
                self.eMap[i, j] = e_ssm
                toc = time.perf_counter()

                dmPath = os.path.join(self.cachedir, filename + ".dmmap")
                A = {
                    "dMmap": self.dMmap,
                    "eMap": self.eMap,
                    "angles": angles,
                    "dtRange": dtRange,
                    "time": toc - tic,
                    "tA": tA,
                    "m0": 1,
                    "ra": TL.coords.ra,
                    "dec": TL.coords.dec,
                    "mass": self.mass,
                }
                with open(dmPath, "wb") as f:
                    pickle.dump(A, f)
                print("Mass - ", dm * self.mass)
                print("Best Epsilon - ", e_ssm)

    def calculate_dMmap_collocate(self, TL, tA, dtRange, filename):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file. Only collocation without the single
        shooting.

        Args:
            TL (TargetList module):
                Target list of stars
            tA (astropy Time array):
                Initial absolute mission time in MJD
            dtRange (astropy Time array):
                Transfer times to try
            filename (string):
                File name to store the cached data.
        """

        sInds = np.arange(0, TL.nStars)
        ang = self.star_angularSep(TL, 0, sInds, tA)
        sInd_sorted = np.argsort(ang)
        angles = ang[sInd_sorted].to("deg").value

        dtFlipped = np.flipud(dtRange)

        self.dMmap = np.zeros([len(dtRange), len(angles)])
        self.eMap = np.zeros([len(dtRange), len(angles)])

        tic = time.perf_counter()
        for j, n in enumerate(sInd_sorted):
            for i, t in enumerate(dtFlipped):
                print(i, j)
                s_coll, t_coll, e_coll, TmaxRange = self.collocate_Trajectory(
                    TL, 0, n, tA, t
                )

                if e_coll == 2 and t.value < 30:
                    break

                m = s_coll[-1][6, :]
                dm = m[-1] - m[0]
                self.dMmap[i, j] = dm
                self.eMap[i, j] = e_coll
                toc = time.perf_counter()

                dmPath = os.path.join(self.cachedir, filename + ".dmmap")
                A = {
                    "dMmap": self.dMmap,
                    "eMap": self.eMap,
                    "angles": angles,
                    "dtRange": dtRange,
                    "time": toc - tic,
                    "tA": tA,
                    "ra": TL.coords.ra,
                    "dec": TL.coords.dec,
                    "mass": self.mass,
                }
                with open(dmPath, "wb") as f:
                    pickle.dump(A, f)
                print("Mass - ", dm * self.mass)
                print("Best Epsilon - ", e_coll)

    def calculate_dMmap_collocateEnergy(
        self, TL, tA, dtRange, filename, m0=1, seed=000000000
    ):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file. Only minimum energy collocation is
        used.

        Args:
            TL (TargetList module):
                Target list of stars
            tA (astropy Time array):
                Initial absolute mission time in MJD
            dtRange (astropy Time array):
                Transfer times to try
            filename (string):
                File name to store the cached data.
            m0 (float):
                Initial mass
            seed (int):
                Seed random number for repeatability of experiments
        """

        sInds = np.arange(0, TL.nStars)
        ang = self.star_angularSep(TL, 0, sInds, tA)
        sInd_sorted = np.argsort(ang)
        angles = ang[sInd_sorted].to("deg").value

        dtFlipped = np.flipud(dtRange)

        self.dMmap = np.zeros([len(dtRange), len(angles)])
        self.eMap = 2 * np.ones([len(dtRange), len(angles)])

        tic = time.perf_counter()
        for j, n in enumerate(sInd_sorted):
            for i, t in enumerate(dtFlipped):
                print(i, j)
                s_coll, t_coll, e_coll, TmaxRange = self.collocate_Trajectory_minEnergy(
                    TL, 0, n, tA, t, m0
                )

                # if unsuccessful, reached min time -> move on to next star
                if e_coll == 2 and t.value < 30:
                    break

                m = s_coll[6, :]
                dm = m[-1] - m[0]
                self.dMmap[i, j] = dm
                self.eMap[i, j] = e_coll
                toc = time.perf_counter()

                dmPath = os.path.join(self.cachedir, filename + ".dmmap")
                A = {
                    "dMmap": self.dMmap,
                    "eMap": self.eMap,
                    "angles": angles,
                    "dtRange": dtRange,
                    "time": toc - tic,
                    "tA": tA,
                    "m0": m0,
                    "ra": TL.coords.ra,
                    "dec": TL.coords.dec,
                    "seed": seed,
                    "mass": self.mass,
                }
                with open(dmPath, "wb") as f:
                    pickle.dump(A, f)
                print("Mass - ", dm * self.mass)
                print("Best Epsilon - ", e_coll)

    def calculate_dMsols_collocateEnergy(
        self, TL, tStart, tArange, dtRange, N, filename, m0=1, seed=000000000
    ):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file. Only minimum energy collocation is
        used.
        N random combinations of starting times, transfer times, and initial, and
        final stars are used.

        Args:
            TL (TargetList module):
                Target list of stars
            tStart (astropy Time array):
                Initial reference absolute mission time in MJD
            tArange (astropy Time array):
                Potential times to add to tStart
            dtRange (astropy Time array):
                Transfer times to try
            N (int):
                Number of trials/combinations to perform
            filename (string):
                File name to store the cached data.
            m0 (float):
                Initial mass
            seed (int):
                Seed random number for repeatability of experiments
        """

        self.dMmap = np.zeros(N)
        self.eMap = 2 * np.ones(N)
        iLog = np.zeros(N)
        jLog = np.zeros(N)
        dtLog = np.zeros(N)
        tALog = np.zeros(N)
        angLog = np.zeros(N) * u.deg

        tic = time.perf_counter()
        for n in range(N):
            print("---------\nIteration", n)

            i = np.random.randint(0, TL.nStars)
            j = np.random.randint(0, TL.nStars)
            dt = np.random.randint(0, len(dtRange))
            tA = np.random.randint(0, len(tArange))
            ang = self.star_angularSep(TL, i, j, tStart + tArange[tA])

            print("star pair  :", i, j)
            print("ang  :", ang.to("deg").value)
            print("dt   :", dtRange[dt].to("d").value)
            print("tau  :", tArange[tA].to("d").value, "\n")

            # pair = np.array([i, j])
            iLog[n] = i
            jLog[n] = j
            dtLog[n] = dt
            tALog[n] = tA
            angLog[n] = ang

            s_coll, t_coll, e_coll, TmaxRange = self.collocate_Trajectory_minEnergy(
                TL, i, j, tStart + tArange[tA], dtRange[dt], m0
            )

            # if unsuccessful, reached min time -> move on to next star
            if e_coll == 2 and dtRange[dt].value < 30:
                break

            m = s_coll[6, :]
            dm = m[-1] - m[0]
            self.dMmap[n] = dm
            self.eMap[n] = e_coll
            toc = time.perf_counter()

            dmPath = os.path.join(self.cachedir, filename + ".dmsols")
            A = {
                "dMmap": self.dMmap,
                "eMap": self.eMap,
                "angLog": angLog,
                "dtLog": dtLog,
                "time": toc - tic,
                "tArange": tArange,
                "dtRange": dtRange,
                "N": N,
                "tStart": tStart,
                "tALog": tALog,
                "m0": m0,
                "ra": TL.coords.ra,
                "dec": TL.coords.dec,
                "seed": seed,
                "mass": self.mass,
            }
            with open(dmPath, "wb") as f:
                pickle.dump(A, f)
            print("Mass - ", dm * self.mass)
            print("Best Epsilon - ", e_coll)

    def calculate_dMmap_collocateEnergy_angSepDist(
        self, TL, tA, dtRange, nPairs, filename, m0=1, seed=000000000
    ):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file. Only minimum energy collocation is
        used.
        nPair random combinations of initial and final star are used while all transfer
        times are used.

        Args:
            TL (TargetList module):
                Target list of stars
            tA (astropy Time array):
                Initial reference absolute mission time in MJD
            dtRange (astropy Time array):
                Transfer times to try
            nPairs (int):
                Number of trials/combinations to perform
            filename (string):
                File name to store the cached data.
            m0 (float):
                Initial mass
            seed (int):
                Seed random number for repeatability of experiments
        """

        iLog = np.zeros([len(dtRange), nPairs])
        jLog = np.zeros([len(dtRange), nPairs])
        psiLog = np.zeros([len(dtRange), nPairs])

        dtFlipped = np.flipud(dtRange)

        self.dMmap = np.zeros([len(dtRange), nPairs])
        self.eMap = 2 * np.ones([len(dtRange), nPairs])

        tic = time.perf_counter()
        toc = time.perf_counter()
        for t, dt in enumerate(dtFlipped):

            iSelect, jSelect, psiSelect = self.selectPairsOfStars(
                TL, nPairs, tA, dt.value, int(1e6)
            )
            sort = np.argsort(psiSelect)
            iSelect = iSelect[sort]
            jSelect = jSelect[sort]
            psiSelect = psiSelect[sort]

            for n, (i, j) in enumerate(zip(iSelect, jSelect)):
                elapsedTime = (toc - tic) / 3600
                totalTime = elapsedTime * len(dtRange) * nPairs / (1 + t * nPairs + n)
                print("dt :", dt, " star #:", n, " /", nPairs)
                print("Time Elapsed: ", elapsedTime, " hrs")
                print("Time Left: ", totalTime - elapsedTime, " hrs")
                s_coll, t_coll, e_coll, TmaxRange = self.collocate_Trajectory_minEnergy(
                    TL, i, j, tA, dt, m0
                )

                m = s_coll[6, :]
                dm = m[-1] - m[0]
                self.dMmap[t, n] = dm
                self.eMap[t, n] = e_coll

                iLog[t, n] = int(i)
                jLog[t, n] = int(j)
                psiLog[t, n] = psiSelect[n]

                toc = time.perf_counter()

                dmPath = os.path.join(self.cachedir, filename + ".dmmap")
                A = {
                    "dMmap": self.dMmap,
                    "eMap": self.eMap,
                    "angles": psiLog,
                    "dtRange": dtRange,
                    "time": toc - tic,
                    "tA": tA,
                    "m0": m0,
                    "lon": TL.coords.lon,
                    "lat": TL.coords.lat,
                    "seed": seed,
                    "mass": self.mass,
                    "iLog": iLog,
                    "jLog": jLog,
                }

                with open(dmPath, "wb") as f:
                    pickle.dump(A, f)
                print("---Mass - ", dm * self.mass)
                print("---Best Epsilon - ", e_coll)

    def calculate_dMmap_collocateEnergy_LatLon(
        self, TL, tA, dtRange, nStars, filename, m0=1, seed=000000000
    ):
        """Calculates change in mass for transfers of various times and angular
        distances

        These are stored in a cache .dmmap file. Only minimum energy collocation is
        used.
        All pairs between nStars random stars from the targetlist. All transfer times
        are used.

        Args:
            TL (TargetList module):
                Target list of stars
            tA (astropy Time array):
                Initial reference absolute mission time in MJD
            dtRange (astropy Time array):
                Transfer times to try
            nStars (int):
                Number of trials/combinations to perform
            filename (string):
                File name to store the cached data.
            m0 (float):
                Initial mass
            seed (int):
                Seed random number for repeatability of experiments
        """

        coords = TL.coords
        lon = coords.lon
        sInds = np.random.choice(TL.nStars, int(nStars), replace=False)
        sInds = sInds[np.argsort(lon[sInds])]
        dtFlipped = np.flipud(dtRange)

        self.dMmap = np.zeros([len(dtRange), len(sInds), len(sInds)])
        self.eMap = 2 * np.ones([len(dtRange), len(sInds), len(sInds)])

        tic = time.perf_counter()
        toc = time.perf_counter()
        for i, ni in enumerate(sInds):
            for j, nj in enumerate(sInds):
                for n, t in enumerate(dtFlipped):
                    elapsedTime = (toc - tic) / 3600
                    totalTime = (
                        elapsedTime
                        * len(dtRange)
                        * nStars**2
                        / (1 + t + j * len(dtRange) + i * nStars * len(dtRange))
                    )
                    print("dt :", t, " star #:", i, "-->", j)
                    print("Time Elapsed: ", elapsedTime, " hrs")
                    print("Time Left: ", totalTime - elapsedTime, " hrs")
                    print(i, j, t.value)
                    (
                        s_coll,
                        t_coll,
                        e_coll,
                        TmaxRange,
                    ) = self.collocate_Trajectory_minEnergy(TL, ni, nj, tA, t, m0)

                    # if unsuccessful, reached min time -> move on to next star
                    if e_coll == 2 and t.value < 30:
                        break

                    m = s_coll[6, :]
                    dm = m[-1] - m[0]
                    self.dMmap[n, i, j] = dm
                    self.eMap[n, i, j] = e_coll
                    toc = time.perf_counter()

                    dmPath = os.path.join(self.cachedir, filename + ".dmmap")
                    A = {
                        "dMmap": self.dMmap,
                        "eMap": self.eMap,
                        "dtRange": dtRange,
                        "time": toc - tic,
                        "tA": tA,
                        "m0": m0,
                        "lon": TL.coords.lon,
                        "lat": TL.coords.lat,
                        "sInds": sInds,
                        "seed": seed,
                        "mass": self.mass,
                    }
                    with open(dmPath, "wb") as f:
                        pickle.dump(A, f)
                    print("Mass - ", dm * self.mass)
                    print("Best Epsilon - ", e_coll)
