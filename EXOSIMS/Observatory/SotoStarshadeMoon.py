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
import scipy

import astropy.coordinates as coord
from astropy.coordinates import GCRS

EPS = np.finfo(float).eps


class SotoStarshadeMoon(SotoStarshade,ObservatoryMoonHalo):
    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """

    def __init__(self, orbit_datapath=None, f_nStars=10, **specs):
#        TK = self.TimeKeeping
#        SotoStarshade.__init__(self, **specs)

        ObservatoryMoonHalo.__init__(self, **specs)
        
#        from astropy.time import Time
#        import matplotlib.pyplot as plt
#        newTime = 60258.87431353273
#        times = np.linspace(60232,newTime,1001)
#        times = np.append(60232,times)
#        currentTimes = Time(times, scale='tai', format='mjd')
#        r_earth = self.spk_body(currentTimes, "Earth",False)
#        r_moon = self.spk_body(currentTimes, "Moon", False)
#        r_em = r_earth - r_moon
#        r_em = r_em.to('km')
#        norms = np.linalg.norm(r_em.value, axis=1)
#        tmp = norms - norms[0]
#
##        pos_val = np.argwhere(tmp == np.min(np.abs(tmp[1:-1])))
##        neg_val = np.argwhere(tmp == -np.min(np.abs(tmp[1:-1])))
##        ind = np.append(pos_val,neg_val)
##
##        goodTime = times[ind]-60232
#
##        ax = plt.figure(1).add_subplot(projection='3d')
##        ax.plot(r_em[:,0].value,r_em[:,1].value,r_em[:,2].value)
##
##        plt.show()
#        breakpoint()
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
#        from astropy.time import Time
#        import matplotlib.pyplot as plt
#        tmp2 = np.array([])
#        tA = Time(60575.25, scale='tai', format='mjd')
#        dt = .25
#        tC = tA + dt
#        tB = Time(60581.25, scale='tai', format='mjd')
#        times = np.arange(tC.value,tB.value,dt)
#        obsTimes = Time(times, scale='tai', format='mjd')
#        old_sInd = 0
#
#        sd = self.star_angularSep2(fTL, old_sInd, f_sInds, tA)
#        slewTimes = self.calculate_slewTimes2(fTL, old_sInd, f_sInds, sd, tA, tA)
#        dfp = np.arange(0.01,.1,0.01)
#        tmp_dV = np.array([])*u.m/u.s
#        tmp_time = np.array([])*u.s
#        for jj in dfp:
#            tmp1 = (slewTimes * self.ao * jj).to("m/s")
#            tmp_dV = np.append(tmp_dV,tmp1)
#            tmp_time = np.append(tmp_time,slewTimes*jj)
##        breakpoint()
#        tmp_dV = tmp_dV.reshape(144,9)
#        tmp_time = tmp_time.reshape(144,9)
#        tmp_dV = tmp_dV[1:11,:].T
#        tmp_time = tmp_time[1:11,:].T
#
#        plt.figure(1)
##        breakpoint()
#        for ii in np.arange(1,10):
#            lname = "next target ind = " + str(f_sInds[ii])
#            plt.plot((tmp_time[:,ii]).to('d').value, tmp_dV[:,ii].value,label=lname)
#
#        plt.xlabel('time burning [d]')
#        plt.ylabel('dv [m/s]')
#        plt.legend()
#        plt.show()
#        breakpoint()
#
#        print(min(tmp2))
#        print(max(tmp2))
#        print(np.mean(tmp2))
#        print(np.median(tmp2))
#        breakpoint()
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

        ctr_0 = 0
        ctr_1 = 0
        ctr_2 = 0
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

                sol, t, status, uA, uB = self.send_it(TL, nA, N[x], tA, tB)     # fix so status, uA, uB isn't returned
                if status == 0:
                    ctr_0 = ctr_0 + 1
                elif status == 1:
                    ctr_1 = ctr_1 + 1
                else:
                    ctr_2 = ctr_2 + 1

                sol_slew[:, x, :] = np.array([sol[0], sol[-1]])
                t_sol[:, x] = np.array([t[0], t[-1]])
                
#                def equationsOfMotion_CRTBP2(t, state):
#                    """Equations of motion of the CRTBP with Solar Radiation Pressure
#
#                    Equations of motion for the Circular Restricted Three Body
#                    Problem (CRTBP). First order form of the equations for integration,
#                    returns 3 velocities and 3 accelerations in (x,y,z) rotating frame.
#                    All parameters are normalized so that time = 2*pi sidereal year.
#                    Distances are normalized to 1AU. Coordinates are taken in a rotating
#                    frame centered at the center of mass of the two primary bodies. Pitch
#                    angle of the starshade with respect to the Sun is assumed to be 60
#                    degrees, meaning the 1/2 of the starshade cross sectional area is
#                    always facing the Sun on average
#
#                    Args:
#                        t (float):
#                            Times in normalized units
#                        state (float 6xn array):
#                            State vector consisting of stacked position and velocity vectors
#                            in normalized units
#
#                    Returns:
#                        float 6xn array:
#                            First derivative of the state vector consisting of stacked
#                            velocity and acceleration vectors in normalized units
#                    """
#
#                    mu = self.mu
#                    m1 = self.m1
#                    m2 = self.m2
#
#                    # conversions from SI to normalized units in CRTBP, numbers from spice kernels
#                    TU = (2.0 * np.pi) / (27.321582 * u.day).to("s")  # time unit
#                    DU = (3.844000E+5*u.km).to('m')  # distance unit
#                    MU = (7.349*10**22 + 5.97219*10**24)*u.kg/self.mu   # mass unit = m1+m2
#
#                    x, y, z, dx, dy, dz = state
#            #        breakpoint()
#            #        x = self.convertPos_to_canonical(x)
#            #        y = self.convertPos_to_canonical(y)
#            #        z = self.convertPos_to_canonical(z)
#            #        dx = self.convertVel_to_canonical(dx)
#            #        dy = self.convertVel_to_canonical(dx)
#            #        dz = self.convertVel_to_canonical(dx)
#
#
#                    rM1 = np.array([[-m2, 0, 0]])  # position of M1 rel 0
#                    rS_M1 = np.array([x, y, z]) - rM1.T  # position of starshade rel M1
#                    u1 = rS_M1 / np.linalg.norm(rS_M1, axis=0)  # radial unit vector along sun-line
#                    u2 = np.array([u1[1, :], -u1[0, :], np.zeros(len(u1.T))])
#                    u2 = u2 / np.linalg.norm(u2, axis=0)  # tangential unit vector to starshade
#
#                    # occulter distance from each of the two other bodies
#                    r1 = np.sqrt((x + mu) ** 2.0 + y**2.0 + z**2.0)
#                    r2 = np.sqrt((1.0 - mu - x) ** 2.0 + y**2.0 + z**2.0)
#
#                    # equations of motion
#                    ds1 = (
#                        x + 2.0 * dy + m1 * (-mu - x) / r1**3.0 + m2 * (1.0 - mu - x) / r2**3.0
#                    )
#                    ds2 = y - 2.0 * dx - m1 * y / r1**3.0 - m2 * y / r2**3.0
#                    ds3 = -m1 * z / r1**3.0 - m2 * z / r2**3.0
#
#                    dr = [dx, dy, dz]
#                    ddr = [ds1, ds2, ds3]
#                    ds = dr + ddr
#
#                    return ds
#
#                tspan = [t[0],t[-1]]
##                state = [sol[0,0],sol[0,1],sol[0,2],sol[0,3],sol[0,4],sol[0,5]]
#                r_tscp = self.haloPosition(tA) + (self.L2_dist)*np.array([1,0,0])
#                tmp_rA = uA * self.occulterSep.to("au") + r_tscp[0]
#                sp = self.convertPos_to_canonical(tmp_rA)
#
#                sv = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
#
#                state = [sp[0],sp[1],sp[2],sv[0],sv[1],sv[2]]
##                breakpoint()
#                sol_int = scipy.integrate.solve_ivp(equationsOfMotion_CRTBP2, tspan, state, t_eval=t)

#            from astropy.time import Time
#            import matplotlib.pyplot as plt
#            import seaborn as sns
#            tA = Time(60575.25, scale='tai', format='mjd')
#            dt = .25
#            tC = tA + dt
#            tB = Time(60581.25, scale='tai', format='mjd')
#            nA = 0
#            nB = np.linspace(1,11,11)
##            breakpoint()
#            times = np.arange(tC.value,tB.value,dt)
#            times = Time(times, scale='tai', format='mjd')
##            breakpoint()
#            # initializing arrays for BVP state solutions
#            sol_slew = np.zeros([2, len(times), 6])
#            t_sol = np.zeros([2, len(times)])
#            tmp3 = np.array([])
#            plt.figure(1)
#            for y in np.arange(len(nB)):
#                for x in np.arange(len(times)):
#                    # simulating slew trajectory from star A at tA to star B at tB
#        #                breakpoint()
#                    sol, t = self.send_it(TL, nA, int(nB[y]), tA, times[x])
#
#                    sol_slew[:, x, :] = np.array([sol[0], sol[-1]])
#                    t_sol[:, x] = np.array([t[0], t[-1]])
#
#        #            sol, t = self.send_it(TL, nA, nB, tA, tB)

                # starshade velocities at both endpoints of the slew trajectory
                r_slewA = sol_slew[0, :, 0:3]
                r_slewB = sol_slew[-1, :, 0:3]
                v_slewA = sol_slew[0, :, 3:6]
                v_slewB = sol_slew[-1, :, 3:6]

                if len(N) == 1:     # change this back to len(N)
                    t_slewA = t_sol[0]
                    t_slewB = t_sol[1]
                else:
                    t_slewA = t_sol[0, 0]
                    t_slewB = t_sol[1, 1]
        #            breakpoint()
                # starshade velocities at both endpoints of the slew trajectory
        #            r_slewA = sol[0,0:3]
        #            r_slewB = sol[-1,0:3]
        #            v_slewA = sol[0,3:6]
        #            v_slewB = sol[-1,3:6]
        #
        #            t_slewA = t[0]
        #            t_slewB = t[1]

                r_haloA = (self.haloPosition(tA) + self.L2_dist * np.array([1, 0, 0]))[0]
                r_haloA = self.convertPos_to_canonical(r_haloA)
                r_haloB = (self.haloPosition(tB) + self.L2_dist * np.array([1, 0, 0]))[0]
                r_haloB = self.convertPos_to_canonical(r_haloB)

                v_haloA = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
                v_haloB = self.convertVel_to_canonical(self.haloVelocity(tB)[0])

                dvAs = self.rot2inertV(r_slewA, v_slewA, t_slewA)
                dvAh = self.rot2inertV(r_haloA, v_haloA, t_slewA)
                dvA = dvAs - dvAh
                
                dvBs = self.rot2inertV(r_slewB, v_slewB, t_slewB)
                dvBh = self.rot2inertV(r_haloB, v_haloB, t_slewB)
                dvB = dvBs - dvBh
#                breakpoint()
                
        #            dV = self.convertVel_to_dim(np.linalg.norm(dvA)) + self.convertVel_to_dim(np.linalg.norm(dvB))

                if len(dvA) == 1:
                    dV = self.convertVel_to_dim(np.linalg.norm(dvA)) + self.convertVel_to_dim(np.linalg.norm(dvB))
                else:
                    dV = self.convertVel_to_dim(np.linalg.norm(dvA, axis=1)) + self.convertVel_to_dim(np.linalg.norm(dvB, axis=1))

#                if status == 1:
#                    import matplotlib.pyplot as plt
#                    pos_int = sol_int.y.T
#                    pos_int = pos_int[:,0:3]
#                    pos = sol[:,0:3]
#
#
#                    fig = plt.figure(1)
#                    ax = fig.add_subplot(221, projection='3d')
#                    ax.plot(pos[:,0],pos[:,1],pos[:,2],label='trajectory')
#                    ax.plot(pos_int[:,0],pos_int[:,1],pos_int[:,2],label='integration')
#                    ax.legend()
#                    plt.title("Slew Trajectory " + str(nA) + " to " + str(N[x]))
#                    ax.set_xlabel('X [DU]')
#                    ax.set_ylabel('Y [DU]')
#                    ax.set_zlabel('Z [DU]')
#
#                    ax = fig.add_subplot(222)
#                    ax.plot(pos[:,0],pos[:,1],label='trajectory')
#                    ax.plot(pos_int[:,0],pos_int[:,1],label='integration')
#                    ax.set_xlabel('X [DU]')
#                    ax.set_ylabel('Y [DU]')
#
#                    ax = fig.add_subplot(223)
#                    ax.plot(pos[:,2],pos[:,1],label='trajectory')
#                    ax.plot(pos_int[:,2],pos_int[:,1],label='integration')
#                    ax.set_xlabel('Z [DU]')
#                    ax.set_ylabel('Y [DU]')
#
#                    ax = fig.add_subplot(224)
#                    ax.plot(pos[:,0],pos[:,2],label='trajectory')
#                    ax.plot(pos_int[:,0],pos_int[:,2],label='integration')
#                    ax.set_xlabel('X [DU]')
#                    ax.set_ylabel('Z [DU]')
#
#
#                    r_e = (self.kernel[0, 3].compute(tA.jd) + self.kernel[3, 399].compute(tA.jd))*u.km
#                    r_m = (self.kernel[0, 3].compute(tA.jd) + self.kernel[3, 301].compute(tA.jd))*u.km
#
#                    r_e = self.icrs2gcrs(r_e,tA)
#                    r_m = self.icrs2gcrs(r_m,tA)
#
#                    C_G2B = self.body2geo(tA).T
#
#                    r_e = C_G2B @ r_e
#                    r_m = C_G2B @ r_m
#
#                    dt = tA.value - self.equinox.value[0]
#                    theta = self.convertTime_to_canonical(dt*u.d)
#
#                    C_B2R = self.rot(theta,3)
#
#                    r_e = C_B2R @ r_e
#                    r_m = C_B2R @ r_m
#
#                    vec1 = np.array([r_slewA[x,:],r_slewA[x,:] + uA])
#                    vec2 = np.array([r_slewB[x,:],r_slewB[x,:] + uB])
#
#                    r_e = self.convertPos_to_canonical(r_e)
#                    r_m = self.convertPos_to_canonical(r_m)
#                    l2 = self.convertPos_to_canonical(self.L2_dist)*np.array([1, 0, 0])
#
#                    halo_times = (np.arange(0,self.period_halo,.0001)*u.yr).to('d')
#                    r_halos = (self.haloPosition(halo_times) + self.L2_dist * np.array([1, 0, 0]))
#                    r_halos = self.convertPos_to_canonical(r_halos)
##                    breakpoint()
##                    fig = plt.figure(2)
##                    ax = fig.add_subplot(projection='3d')
##                    ax.scatter(r_e[0], r_e[1], r_e[2],label='Earth')
##                    ax.scatter(r_m[0], r_m[1], r_m[2],label='Moon')
##                    ax.scatter(l2[0],l2[1],l2[2],label='L2')
##                    ax.plot(pos[:,0],pos[:,1],pos[:,2],label='trajectory')
##                    ax.plot(vec1[:,0],vec1[:,1],vec1[:,2],label='lookVec 1')
##                    ax.plot(vec2[:,0],vec2[:,1],vec2[:,2],label='lookVec 2')
##                    ax.plot(r_halos[:,0],r_halos[:,1],r_halos[:,2],label='halo')
##                    ax.legend()
##                    ax.set_xlabel('X [DU]')
##                    ax.set_ylabel('Y [DU]')
##                    ax.set_zlabel('Z [DU]')
#                    fig = plt.figure(2)
#                    ax = fig.add_subplot(311)
#                    ax.plot(t,pos[:,0] - pos_int[:,0])
#                    ax.set_xlabel('time [TU]')
#                    ax.set_ylabel('X difference [DU]')
#
#                    ax = fig.add_subplot(312)
#                    ax.plot(t,pos[:,1]-pos_int[:,1])
#                    ax.set_xlabel('time [TU]')
#                    ax.set_ylabel('Y difference [DU]')
#
#                    ax = fig.add_subplot(313)
#                    ax.plot(t,pos[:,2]-pos_int[:,2])
#                    ax.set_xlabel('time [TU]')
#                    ax.set_ylabel('Z difference[DU]')
#
#                    plt.show()
#                    breakpoint()
#
                
                
#            tmpMin = min(tmp)
#            tmpMax = max(tmp)
#            tmpAvg = np.average(tmp)
#            tmpMed = np.median(tmp)
            
#            print(str(tmpMin))
#
#            breakpoint()
#            tmp3 = tmp.reshape(1,len(tmp))
#            tmp3 = tmp3.reshape(11,len(tmp))
#            plt.figure(1)
#            ax = sns.heatmap(tmp3, cbar=True)
#            ax.set_xlabel("Slew Time")
#            ax.set_ylabel("Next Target")
#            ax.set_title("dV Map m/s")
#            breakpoint()
#            ax.set_xticklabels((times.value).astype(str))
#            ax.set_yticklabels(np.array([nB]).astype(str))
#            plt.show()
#
#            plt.xlabel("Slew Time d")
#            plt.ylabel("dv m/s")
#            plt.title("log scale dV")
#            plt.legend()
#            plt.show()
#            breakpoint()
        
        print(str(ctr_0))
        print(str(ctr_1))
        print(str(ctr_2))
        tmp = dV.to("m/s")
        breakpoint()
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

        # check in lookVectors
#        breakpoint()
        angle, uA, uB, r_tscp = self.lookVectors(TL, nA, nB, tA, tB)
#        breakpoint()
        vA = self.convertVel_to_canonical(self.haloVelocity(tA)[0])
        vB = self.convertVel_to_canonical(self.haloVelocity(tB)[0])
        
##        tmp_rA = (uA*90000*u.km).to('AU') + r_tscp[0]*u.AU
##        tmp_rB = (uB*90000*u.km).to('AU') + r_tscp[-1]*u.AU
#        tmp_rA = r_tscp[0]*u.AU
#        tmp_rB = r_tscp[-1]*u.AU
        tmp_rA = uA * self.occulterSep.to("au") + r_tscp[0]*u.AU
        tmp_rB = uB * self.occulterSep.to("au") + r_tscp[-1]*u.AU
#        breakpoint()
        self.rA = self.convertPos_to_canonical(tmp_rA)
        self.rB = self.convertPos_to_canonical(tmp_rB)

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
                s (float nx6 array):
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
            self.equationsOfMotion_CRTBP, self.boundary_conditions, t, sG, tol=1e-10, fun_jac = jacobian_CRTBP2)

        s = sol.y.T
        t_s = sol.x
        status_s = sol.status
#        if status_s == 2:
#            breakpoint()
        return s, t_s, status_s, uA, uB

    def star_angularSep2(self, TL, old_sInd, sInds, currentTime):
        """Finds angular separation from old star to given list of stars

        This method returns the angular separation from the last observed
        star to all others on the given list at the currentTime.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the last star of interest
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            float:
                Angular separation between two target stars
        """
        if old_sInd is None:
            sd = np.zeros(len(sInds)) * u.rad
        else:
            # position vector of previous target star
            r_old = TL.starprop(old_sInd, currentTime)[0]
            u_old = r_old.to("AU").value / np.linalg.norm(r_old.to("AU").value)
            # position vector of new target stars
            r_new = TL.starprop(sInds, currentTime)
            u_new = (
                r_new.to("AU").value.T / np.linalg.norm(r_new.to("AU").value, axis=1)
            ).T
            # angle between old and new stars
            sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1)) * u.rad

            # A-frame
            a1 = u_old / np.linalg.norm(u_old)  # normalized old look vector
            a2 = np.array([a1[1], -a1[0], 0])  # normal to a1
            a3 = np.cross(a1, a2)  # last part of the A basis vectors

            # finding sign of angle
            # The star angular separation can be negative
            u2_Az = np.dot(a3, u_new.T)
            sgn = np.sign(u2_Az)
            sgn[np.where(sgn == 0)] = 1
            sd = sgn * sd

        return sd

    def calculate_slewTimes2(self, TL, old_sInd, sInds, sd, obsTimes, currentTime):
        """Finds slew times and separation angles between target stars

        This method determines the slew times of an occulter spacecraft needed
        to transfer from one star's line of sight to all others in a given
        target list.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the most recently observed star
            sInds (~numpy.ndarray(int)):
                Integer indices of the star of interest
            sd (~astropy.units.Quantity):
                Angular separation between stars in rad
            obsTimes (~astropy.time.Time(~numpy.ndarray)):
                Observation times for targets.
            currentTime (astropy Time):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity:
                Time to transfer to new star line of sight in units of days
        """

        self.ao = self.thrust / self.scMass
        slewTime_fac = (
            (
                2.0
                * self.occulterSep
                / np.abs(self.ao)
                / (self.defburnPortion / 2.0 - self.defburnPortion**2.0 / 4.0)
            )
            .decompose()
            .to("d2")
        )

        if old_sInd is None:
            slewTimes = np.zeros(TL.nStars) * u.d
        else:
            # calculate slew time
            slewTimes = np.sqrt(
                slewTime_fac * np.sin(abs(sd) / 2.0)
            )  # an issue exists if sd is negative

            # The following are debugging
            assert (
                np.where(np.isnan(slewTimes))[0].shape[0] == 0
            ), "At least one slewTime is nan"

        return slewTimes
