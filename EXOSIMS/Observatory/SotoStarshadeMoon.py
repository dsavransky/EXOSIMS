from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
from EXOSIMS.Observatory.ObservatoryMoonHalo import ObservatoryMoonHalo
from EXOSIMS.TargetList.EclipticTargetList import EclipticTargetList
import numpy as np
import astropy.units as u
from scipy.integrate import solve_bvp
import scipy.optimize as optimize
import scipy.interpolate as interp

EPS = np.finfo(float).eps


class SotoStarshadeMoon(SotoStarshade, ObservatoryMoonHalo):
    """StarShade Observatory class
    This class is implemented at EM L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """

    def __init__(self, orbit_datapath=None, f_nStars=10, **specs):
        ObservatoryMoonHalo.__init__(self, **specs)

        self.f_nStars = int(f_nStars)

        # instantiating fake star catalog, used to generate good dVmap
        lat_sep = 20
        lon_sep = 20
        star_dist = 1

        fTL = EclipticTargetList(
            **{
                "lat_sep": lat_sep,
                "lon_sep": lon_sep,
                "star_dist": star_dist,
                "modules": {
                    "StarCatalog": "FakeCatalog_UniformSpacing_wInput",
                    "TargetList": "EclipticTargetList ",
                    "OpticalSystem": "Nemati",
                    "ZodiacalLight": "Stark",
                    "PostProcessing": " ",
                    "Completeness": " ",
                    "BackgroundSources": "GalaxiesFaintStars",
                    "PlanetPhysicalModel": " ",
                    "PlanetPopulation": "KeplerLike1",
                },
                "scienceInstruments": [{"name": "imager"}],
                "starlightSuppressionSystems": [{"name": "HLC-565"}],
            }
        )

        f_sInds = np.arange(0, fTL.nStars)

        dV, ang, dt = self.generate_dVMap(fTL, 0, f_sInds, self.equinox[0])

        # pick out unique angle values
        ang, unq = np.unique(ang, return_index=True)
        dV = dV[:, unq]

        # create dV 2D interpolant
        self.dV_interp = interp.interp2d(dt, ang, dV.T, kind="linear")

    def impulsiveSlew_dV(self, dt, TL, nA, N, tA):
        """Finds the change in velocity needed to transfer to a new star line of sight

        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        This method can handle multiple indeces for the next target stars and calculates
        the dVs of each trajectory from the same starting star.

        Args:
            dt (float 1x1 ndarray):
                Number of days corresponding to starshade slew time
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            N  (integer):
                Integer index of the next star(s) of interest
            tA (astropy Time array):
                Current absolute mission time in MJD

        Returns:
            float nx6 ndarray:
                State vectors in rotating frame in normalized units
        """

        if dt.shape:
            dt = dt[0]

        if nA is None:
            dV = np.zeros(len(N))
        else:
            # if only calculating one trajectory, this allows loop to run
            if N.size == 1:
                N = np.array([N])

            # time to reach star B's line of sight
            tB = tA + dt * u.d

            # initializing arrays for BVP state solutions
            sol_slew = np.zeros([2, len(N), 6])
            t_sol = np.zeros([2, len(N)])
            for x in range(len(N)):
                # simulating slew trajectory from star A at tA to star B at tB

                sol, t, status, uA, uB = self.send_it(
                    TL, nA, N[x], tA, tB
                )  # fix so status, uA, uB isn't returned
                sol_slew[:, x, :] = np.array([sol[0], sol[-1]])
                t_sol[:, x] = np.array([t[0], t[-1]])

                # starshade velocities at both endpoints of the slew trajectory
                r_slewA = sol_slew[0, :, 0:3]
                r_slewB = sol_slew[-1, :, 0:3]
                v_slewA = sol_slew[0, :, 3:6]
                v_slewB = sol_slew[-1, :, 3:6]

                if len(N) == 1:  # change this back to len(N)
                    t_slewA = t_sol[0]
                    t_slewB = t_sol[1]
                else:
                    t_slewA = t_sol[0, :]
                    t_slewB = t_sol[1, :]

                r_haloA = (self.haloPosition(tA) + self.L2_dist * np.array([1, 0, 0]))[
                    0
                ]
                r_haloA = self.convertPos_to_canonical(r_haloA)
                r_haloB = (self.haloPosition(tB) + self.L2_dist * np.array([1, 0, 0]))[
                    0
                ]
                r_haloB = self.convertPos_to_canonical(r_haloB)

                v_haloA = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
                v_haloB = self.convertVel_to_canonical(self.haloVelocity(tB)[0])

                dvAs = self.rot2inertV(r_slewA, v_slewA, t_slewA)
                dvAh = self.rot2inertV(r_haloA, v_haloA, t_slewA)
                dvA = dvAs - dvAh

                dvBs = self.rot2inertV(r_slewB, v_slewB, t_slewB)
                dvBh = self.rot2inertV(r_haloB, v_haloB, t_slewB)
                dvB = dvBs - dvBh

            if len(dvA) == 1:
                dV = self.convertVel_to_dim(
                    np.linalg.norm(dvA)
                ) + self.convertVel_to_dim(np.linalg.norm(dvB))
            else:
                dV = self.convertVel_to_dim(
                    np.linalg.norm(dvA, axis=1)
                ) + self.convertVel_to_dim(np.linalg.norm(dvB, axis=1))
        return dV.to("m/s")

    def minimize_slewTimes(self, TL, nA, nB, tA):
        """Minimizes the slew time for a starshade transferring to a new star
        line of sight

        This method uses scipy's optimization module to minimize the slew time for
        a starshade transferring between one star's line of sight to another's under
        the constraint that the total change in velocity cannot exceed more than a
        certain percentage of the total fuel on board the starshade.

        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD

        Returns:
            tuple:
                opt_slewTime (float):
                    Optimal slew time in days for starshade transfer to a new
                    line of sight
                opt_dV (float):
                    Optimal total change in velocity in m/s for starshade
                    line of sight transfer

        """

        def slewTime_objFun(dt):
            if dt.shape:
                dt = dt[0]

            return dt

        def slewTime_constraints(dt, TL, nA, nB, tA):
            dV = self.calculate_dV(dt, TL, nA, nB, tA)
            dV_max = self.dVmax

            return (dV_max - dV).value, dt - 1

        dt_guess = 20  # TODO: revisit these after initial sweep
        Tol = 1e-3

        t0 = [dt_guess]

        res = optimize.minimize(
            slewTime_objFun,
            t0,
            method="COBYLA",
            constraints={
                "type": "ineq",
                "fun": slewTime_constraints,
                "args": ([TL, nA, nB, tA]),
            },
            tol=Tol,
            options={"disp": False},
        )

        opt_slewTime = res.x
        opt_dV = self.calculate_dV(opt_slewTime, TL, nA, nB, tA)

        return opt_slewTime, opt_dV.value

    def minimize_fuelUsage(self, TL, nA, nB, tA):
        """Minimizes the fuel usage of a starshade transferring to a new star
        line of sight

        This method uses scipy's optimization module to minimize the fuel usage for
        a starshade transferring between one star's line of sight to another's. The
        total slew time for the transfer is bounded with some dt_min and dt_max.

        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD

        Returns:
            tuple:
                opt_slewTime (float):
                    Optimal slew time in days for starshade transfer to a
                    new line of sight
                opt_dV (float):
                    Optimal total change in velocity in m/s for starshade
                    line of sight transfer

        """

        def fuelUsage_objFun(dt, TL, nA, N, tA):
            dV = self.calculate_dV(dt, TL, nA, N, tA)
            return dV.value

        def fuelUsage_constraints(dt, dt_min, dt_max):
            return dt_max - dt, dt - dt_min

        dt_guess = 20  # TODO: revisit after initial sweep
        dt_min = 1
        dt_max = 45
        Tol = 1e-5

        t0 = [dt_guess]

        res = optimize.minimize(
            fuelUsage_objFun,
            t0,
            method="COBYLA",
            args=(TL, nA, nB, tA),
            constraints={
                "type": "ineq",
                "fun": fuelUsage_constraints,
                "args": ([dt_min, dt_max]),
            },
            tol=Tol,
            options={"disp": False},
        )
        opt_slewTime = res.x
        opt_dV = res.fun

        return opt_slewTime, opt_dV

    def send_it(self, TL, nA, nB, tA, tB):
        """Solves boundary value problem between starshade star alignments

        This method solves the boundary value problem for starshade star alignments
        with two given stars at times tA and tB. It uses scipy's solve_bvp method.

        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
            tB (astropy Time array):
                Absolute mission time for next star alignment in MJD

        Returns:
            float nx6 ndarray:
                State vectors in rotating frame in normalized units
        """

        angle, uA, uB, r_tscp = self.lookVectors(TL, nA, nB, tA, tB)

        vA = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
        vB = self.convertVel_to_canonical(self.haloVelocity(tB)[0])

        tmp_rA = uA * self.occulterSep.to("au") + r_tscp[0] * u.AU
        tmp_rB = uB * self.occulterSep.to("au") + r_tscp[-1] * u.AU

        self.rA = self.convertPos_to_canonical(tmp_rA)
        self.rB = self.convertPos_to_canonical(tmp_rB)

        a = self.convertTime_to_canonical(
            ((np.mod(tA.value, self.equinox[0].value) * u.d)).to("yr")
        )
        b = self.convertTime_to_canonical(
            ((np.mod(tB.value, self.equinox[0].value) * u.d)).to("yr")
        )

        # running shooting algorithm
        t = np.linspace(a, b, 2)

        sG = np.array(
            [
                [self.rA[0], self.rB[0]],
                [self.rA[1], self.rB[1]],
                [self.rA[2], self.rB[2]],
                [vA[0], vB[0]],
                [vA[1], vB[1]],
                [vA[2], vB[2]],
            ]
        )

        def jacobian_CRTBP2(t, state):
            """Equations of motion of the CRTBP

            Equations of motion for the Circular Restricted Three Body
            Problem (CRTBP). First order form of the equations for integration,
            returns 3 velocities and 3 accelerations in (x,y,z) rotating frame.
            All parameters are normalized so that time = 2*pi sidereal year.
            Distances are normalized to 1AU. Coordinates are taken in a rotating
            frame centered at the center of mass of the two primary bodies

            Args:
                t (float):
                    Times in normalized units
                state (float nx6 array):
                    State vector consisting of stacked position and velocity vectors
                    in normalized units

            Returns:
                float nx6x6 array:
                    Jacobian matrix of the state vector in normalized units
            """

            mu = self.mu
            m1 = self.m1
            m2 = self.m2

            # unpack components from state vector
            x, y, z, dx, dy, dz = state

            # determine shape of state vector (n = 6, m = size of t)
            n, m = state.shape

            # breaking up some of the calculations for the jacobian
            a8 = (mu + x - 1.0) ** 2.0 + y**2.0 + z**2.0
            a9 = (mu - x) ** 2.0 + y**2.0 + z**2.0
            a1 = 2.0 * mu + 2.0 * x - 2.0
            a2 = 2.0 * mu - 2.0 * x
            a3 = m2 / a8 ** (1.5)
            a4 = m1 / a9 ** (1.5)
            a5 = 3.0 * m1 * y * z / a9 ** (2.5) + 3.0 * m2 * y * z / a8 ** (2.5)
            a6 = 2.0 * a8
            a7 = 2.0 * a9

            # Calculating the different elements jacobian matrix

            # ddx,ddy,ddz wrt to x,y,z
            # this part of the jacobian has size 3 x 3 x m
            J1x = (
                3.0 * m2 * a1 * (mu + x - 1.0) / a6
                - a3
                - a4
                - 3.0 * m1 * a2 * (mu + x) / a7
                + 1.0
            )
            J1y = 3.0 * m1 * y * (mu + x) / a9 ** (2.5) + 3.0 * m2 * y * (
                mu + x - 1.0
            ) / a8 ** (2.5)
            J1z = 3.0 * m1 * z * (mu + x) / a9 ** (2.5) + 3.0 * m2 * z * (
                mu + x - 1.0
            ) / a8 ** (2.5)
            J2x = 3.0 * m2 * y * a1 / a6 - 3.0 * m1 * y * a2 / a7
            J2y = (
                3.0 * m1 * y**2.0 / a9 ** (2.5)
                - a3
                - a4
                + 3.0 * m2 * y**2.0 / a8 ** (2.5)
                + 1.0
            )
            J2z = a5
            J3x = 3.0 * m2 * z * a1 / a6 - 2.0 * m1 * z * a2 / a7
            J3y = a5
            J3z = (
                3.0 * m1 * z**2.0 / a9 ** (2.5)
                - a3
                - a4
                + 3.0 * m2 * z**2.0 / a8 ** (2.5)
            )

            J = np.array([[J1x, J1y, J1z], [J2x, J2y, J2z], [J3x, J3y, J3z]])

            # dx,dy,dz wrt to x,y,z
            # this part of the jacobian has size 3 x 3 x m
            Z = np.zeros([3, 3, m])

            # dx,dy,dz wrt to dx,dy,dz
            # this part of the jacobian has size 3 x 3 x m
            E = np.full_like(Z, np.eye(3).reshape(3, 3, 1))

            # ddx,ddy,ddz wrt to dx,dy,dz
            # this part of the jacobian has size 3 x 3 x m
            w = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            W = np.full_like(Z, w.reshape(3, 3, 1))

            # stacking the different matrix blocks into a matrix 6 x 6 x m
            row1 = np.hstack([Z, E])
            row2 = np.hstack([J, W])

            jacobian = np.vstack([row1, row2])

            return jacobian

        sol = solve_bvp(
            self.equationsOfMotion_CRTBP2,
            self.boundary_conditions,
            t,
            sG,
            tol=1e-10,
            fun_jac=jacobian_CRTBP2,
        )

        ss = sol.y.T
        t_s = sol.x
        status_s = sol.status

        return ss, t_s, status_s, uA, uB

    def boundary_conditions2(self, rA, rB):
        """Creates boundary conditions for solving a boundary value problem

        This method returns the boundary conditions for the starshade transfer
        trajectory between the lines of sight of two different stars. Point A
        corresponds to the starshade alignment with star A; Point B, with star B.

        Args:
            rA (float 1x3 ndarray):
                Starshade position vector aligned with current star of interest
            rB (float 1x3 ndarray):
                Starshade position vector aligned with next star of interest

        Returns:
            float 1x6 ndarray:
                Star position vector in rotating frame in units of AU
        """

        BC1 = rA[0] - self.rA2[0]
        BC2 = rA[1] - self.rA2[1]
        BC3 = rA[2] - self.rA2[2]

        BC4 = rB[0] - self.rB2[0]
        BC5 = rB[1] - self.rB2[1]
        BC6 = rB[2] - self.rB2[2]

        BC = np.array([BC1, BC2, BC3, BC4, BC5, BC6])

        return BC

    def equationsOfMotion_CRTBP2(self, t, state):
        """Equations of motion of the CRTBP with Solar Radiation Pressure

        Equations of motion for the Circular Restricted Three Body
        Problem (CRTBP). First order form of the equations for integration,
        returns 3 velocities and 3 accelerations in (x,y,z) rotating frame.
        All parameters are normalized so that time = 2*pi sidereal year.
        Distances are normalized to 1AU. Coordinates are taken in a rotating
        frame centered at the center of mass of the two primary bodies. Pitch
        angle of the starshade with respect to the Sun is assumed to be 60
        degrees, meaning the 1/2 of the starshade cross sectional area is
        always facing the Sun on average

        Args:
            t (float):
                Times in normalized units
            state (float 6xn array):
                State vector consisting of stacked position and velocity vectors
                in normalized units

        Returns:
            float 6xn array:
                First derivative of the state vector consisting of stacked
                velocity and acceleration vectors in normalized units
        """

        mu = self.mu
        m1 = self.m1
        m2 = self.m2

        x, y, z, dx, dy, dz = state

        rM1 = np.array([[-m2, 0, 0]])  # position of M1 rel 0
        rS_M1 = np.array([x, y, z]) - rM1.T  # position of starshade rel M1
        u1 = rS_M1 / np.linalg.norm(rS_M1, axis=0)  # radial unit vector along sun-line
        u2 = np.array([u1[1, :], -u1[0, :], np.zeros(len(u1.T))])
        u2 = u2 / np.linalg.norm(u2, axis=0)  # tangential unit vector to starshade

        # occulter distance from each of the two other bodies
        r1 = np.sqrt((x + mu) ** 2.0 + y**2.0 + z**2.0)
        r2 = np.sqrt((1.0 - mu - x) ** 2.0 + y**2.0 + z**2.0)

        # equations of motion
        ds1 = (
            x + 2.0 * dy + m1 * (-mu - x) / r1**3.0 + m2 * (1.0 - mu - x) / r2**3.0
        )
        ds2 = y - 2.0 * dx - m1 * y / r1**3.0 - m2 * y / r2**3.0
        ds3 = -m1 * z / r1**3.0 - m2 * z / r2**3.0

        dr = [dx, dy, dz]
        ddr = [ds1, ds2, ds3]
        ds = np.vstack([dr, ddr])

        return ds
