from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
from EXOSIMS.Observatory.ObservatoryMoonHalo import ObservatoryMoonHalo
from EXOSIMS.TargetList.EclipticTargetList import EclipticTargetList
import numpy as np
import astropy.units as u
from scipy.integrate import solve_bvp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
import scipy.interpolate as interp
import time
import os
import pickle

EPS = np.finfo(float).eps


class SotoStarshadeMoon(SotoStarshade,ObservatoryMoonHalo):
    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """

    def __init__(self, orbit_datapath=None, f_nStars=10, **specs):

#        SotoStarshade.__init__(self, **specs)
        ObservatoryMoonHalo.__init__(self, **specs)
        
        self.f_nStars = int(f_nStars)

        # instantiating fake star catalog, used to generate good dVmap
        lat_sep = 20
        lon_sep = 20
        star_dist = 1
        
        fTL = EclipticTargetList(**{"lat_sep":lat_sep,"lon_sep":lon_sep,"star_dist":star_dist,\
                    'modules':{"StarCatalog": "FakeCatalog_UniformSpacing_wInput", \
                    "TargetList":"EclipticTargetList ","OpticalSystem": "Nemati", "ZodiacalLight": "Stark", "PostProcessing": " ", \
                    "Completeness": " ","BackgroundSources": "GalaxiesFaintStars", "PlanetPhysicalModel": " ", \
                    "PlanetPopulation": "KeplerLike1"}, "scienceInstruments": [{ "name": "imager"}],  \
                    "starlightSuppressionSystems": [{ "name": "HLC-565"}]   })
        
        f_sInds = np.arange(0,fTL.nStars)
        dV,ang,dt = self.generate_dVMap(fTL,0,f_sInds,self.equinox[0])
        
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

                sol, t = self.send_it(TL, nA, N[x], tA, tB)

                sol_slew[:, x, :] = np.array([sol[0], sol[-1]])
                t_sol[:, x] = np.array([t[0], t[-1]])

            # starshade velocities at both endpoints of the slew trajectory
            r_slewA = sol_slew[0, :, 0:3]
            r_slewB = sol_slew[-1, :, 0:3]
            v_slewA = sol_slew[0, :, 3:6]
            v_slewB = sol_slew[-1, :, 3:6]

            if len(N) == 1:
                t_slewA = t_sol[0]
                t_slewB = t_sol[1]
            else:
                t_slewA = t_sol[0, 0]
                t_slewB = t_sol[1, 1]

            r_haloA = (self.haloPosition(tA) + self.L2_dist * np.array([1, 0, 0]))[0]
            r_haloA = self.convertPos_to_canonical(r_haloA)
            r_haloB = (self.haloPosition(tB) + self.L2_dist * np.array([1, 0, 0]))[0]
            r_haloB = self.convertPos_to_canonical(r_haloB)

            v_haloA = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
            v_haloB = self.convertVel_to_canonical(self.haloVelocity(tB)[0])

            dvA = self.rot2inertV(r_slewA, v_slewA, t_slewA) - self.rot2inertV(
                r_haloA, v_haloA, t_slewA
            )
            dvB = self.rot2inertV(r_slewB, v_slewB, t_slewB) - self.rot2inertV(
                r_haloB, v_haloB, t_slewB
            )

            if len(dvA) == 1:
                dV = self.convertVel_to_dim(np.linalg.norm(dvA)) + self.convertVel_to_dim(np.linalg.norm(dvB))
            else:
                dV = self.convertVel_to_dim(np.linalg.norm(dvA, axis=1)) + self.convertVel_to_dim(np.linalg.norm(dvB, axis=1))

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

        dt_guess = 20       # TODO: revisit these after initial sweep
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

        dt_guess = 20       # TODO: revisit after initial sweep
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

        # position vector of occulter in heliocentric frame
        self.rA = uA * self.occulterSep.to("au").value + r_tscp[0]
        self.rB = uB * self.occulterSep.to("au").value + r_tscp[-1]

        a = self.convertTime_to_canonical(((np.mod(tA.value, self.equinox[0].value) * u.d)).to("yr"))
        b = self.convertTime_to_canonical(((np.mod(tB.value, self.equinox[0].value) * u.d)).to("yr"))

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

        sol = solve_bvp(
            self.equationsOfMotion_CRTBP, self.boundary_conditions, t, sG, tol=1e-10
        )

        s = sol.y.T
        t_s = sol.x
        return s, t_s
