from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
import numpy as np
import astropy.units as u
import os
from STMint.STMint import STMint
import sympy
import scipy
import EXOSIMS
from EXOSIMS.util.OrbitVariationalFirstOrder import OrbitVariationalDataFirstOrder
from EXOSIMS.util.OrbitVariationalDataSecondOrder import OrbitVariationalDataSecondOrder
import math


EPS = np.finfo(float).eps


class KulikStarshade(ObservatoryL2Halo):
    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.

    Args:
        mode (str):
            One of "impulsive" or "energyOptimal".
            "impulsive" calculates approximate relative transfer delta-vs associated with a Hohmann transfer between two stationkeeping orbits in the vicinity of the L2 point.
            "energyOptimal" calculates approximate relative transfer delta-vs assocaited with an energy optimal lowthrust transfer between two stationkeeping orbits in the vicinity of the L2 point.
        dynamics (int):
            0, 1, 2, 3. 0 for default CRTBP dynamics. 1 for CRTBP dynamics including SRP. 2 for CRTBP dynamics accounting for the effects of the moon. 3 for CRTBP dynamics including perturbations from SRP and the moon.
        exponent (int):
            The exponent used for computing STMs at 2^exponent levels of time-discretization.
        precompfname (str):
            Contains the name of the file in which STMs are stored for orbit analysis.
        starShadeRadius (~astropy.units.Quantity):
            Radius of the starshade in meters.
    Attributes:
        mode (str):
            One of "impulsive" or "energyOptimal".
            "impulsive" calculates approximate relative transfer delta-vs associated with a Hohmann transfer between two stationkeeping orbits in the vicinity of the L2 point.
            "energyOptimal" calculates approximate relative transfer delta-vs assocaited with an energy optimal lowthrust transfer between two stationkeeping orbits in the vicinity of the L2 point.
        dynamics (int):
            0, 1, 2, 3. 0 for default CRTBP dynamics. 1 for CRTBP dynamics including SRP. 2 for CRTBP dynamics accounting for the effects of the moon. 3 for CRTBP dynamics including perturbations from SRP and the moon.
        exponent (int):
            The exponent used for computing STMs at 2^exponent levels of time-discretization.
        precompfname (str):
            Contains the name of the file in which STMs are stored for orbit analysis.
        canonical_time_unit (~astropy.units.Quantity):
            canonical time unit of the CRTBP in days.
        starShadeRad (~astropy.units.Quantity):
            Starshade radius in kilometers.
        orb (Union[OrbitVariationalDataFirstOrder, OrbitVariationalDataSecondOrder]):
            Object containing mathematical information necessary to compute relative transfer costs.
    """

    def __init__(
        self,
        mode="impulsive",
        dynamics=0,
        exponent=8,
        precompfname="Observatory/haloImpulsive",
        starShadeRadius=10 * u.m,
        **specs,
    ):
        """Initializes StarShade class. Checks if variational data has already been
        precomputed for given mode and dynamics.
        Args:
            orbit_datapath (float 1x3 ndarray):
                TargetList class object
            dynamics (integer):
                0, 1, 2, 3. 0 for default CRTBP. 1 for SRP. 2 for moon. 3 for SRP and moon.
            mode (str):
                One of "energyOptimal" or "impulsive"
            exponent (float):
                2^exponent subdivisions used in computing the STM and STTs
            precompfname (str) :
                filename where orbit variational data is stored
            starShadeRadius (~astropy.units.Quantity):
                radius of the starshade in meters
            **specs:
                additional specs to be passed to superclass constructor
        """

        self.mode = mode
        self.dynamics = dynamics
        self.exponent = exponent
        self.precompfname = precompfname

        self.canonical_time_unit = u.d * 365.2515 / (2 * math.pi)

        # should be given in m -- converting to km
        self.starShadeRad = starShadeRadius.to(u.km)

        if mode == "energyOptimal":
            if dynamics == 0:

                def optControlDynamics():
                    x, y, z, vx, vy, vz, lx, ly, lz, lvx, lvy, lvz, En = sympy.symbols(
                        "x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En"
                    )
                    mu = 3.00348e-6
                    mu1 = 1.0 - mu
                    mu2 = mu
                    r1 = sympy.sqrt((x + mu2) ** 2 + (y**2) + (z**2))
                    r2 = sympy.sqrt((x - mu1) ** 2 + (y**2) + (z**2))
                    U = (-1.0 / 2.0) * (x**2 + y**2) - (mu1 / r1) - (mu2 / r2)
                    dUdx = sympy.diff(U, x)
                    dUdy = sympy.diff(U, y)
                    dUdz = sympy.diff(U, z)

                    RHS = sympy.Matrix(
                        [
                            vx,
                            vy,
                            vz,
                            ((-1 * dUdx) + 2 * vy),
                            ((-1 * dUdy) - 2 * vx),
                            (-1 * dUdz),
                        ]
                    )

                    variables = sympy.Matrix(
                        [x, y, z, vx, vy, vz, lx, ly, lz, lvx, lvy, lvz, En]
                    )

                    dynamics = sympy.Matrix(
                        sympy.BlockMatrix(
                            [
                                [RHS - sympy.Matrix([0, 0, 0, lvx, lvy, lvz])],
                                [
                                    -1.0
                                    * RHS.jacobian(
                                        sympy.Matrix([x, y, z, vx, vy, vz]).transpose()
                                    )
                                    * sympy.Matrix([lx, ly, lz, lvx, lvy, lvz])
                                ],
                                [0.5 * sympy.Matrix([lvx**2 + lvy**2 + lvz**2])],
                            ]
                        )
                    )
                    # return Matrix([x,y,z,vx,vy,vz]), RHS
                    return variables, dynamics

                fileName = self.precompfname
                trvFileName = os.path.join(EXOSIMS.__path__[0], fileName + "_trvs.mat")
                STMFileName = os.path.join(EXOSIMS.__path__[0], fileName + "_STMs.mat")
                STTFileName = os.path.join(EXOSIMS.__path__[0], fileName + "_STTs.mat")
                # initial conditions for a sun-earth halo orbit
                # with zero initial conditions for costates and energy
                ics = [
                    1.00822114953991,
                    0.0,
                    -0.001200000000000000,
                    0.0,
                    0.010290010931740649,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                T = 3.1002569555488506
                # use 2^8 subdivisions when calculating the values of the STM
                exponent = self.exponent

                # precompute variational data if data file does not already exist
                print(trvFileName)
                if not os.path.isfile(trvFileName):
                    variables, dynamics = optControlDynamics()
                    print("Precomputing variational data")
                    threeBodyInt = STMint(variables, dynamics, variational_order=2)
                    t_step = T / 2.0**exponent
                    curState = ics
                    states = [ics]
                    # STMs = [np.identity(12)]
                    # STTs = [np.zeros((12,12))]
                    STMs = []
                    STTs = []
                    tVals = np.linspace(0, T, num=2**exponent + 1)
                    for i in range(2**exponent):
                        print(f"Integrating: {i}")
                        [state, STM, STT] = threeBodyInt.dynVar_int2(
                            [0, t_step], curState, output="final", max_step=0.001
                        )
                        states.append(state)
                        STMs.append(STM[:12, :12])
                        STTs.append(STT[12, :12, :12])
                        curState = state
                    scipy.io.savemat(
                        trvFileName,
                        {"trvs": np.hstack((np.transpose(np.array([tVals])), states))},
                    )
                    scipy.io.savemat(STMFileName, {"STMs": STMs})
                    scipy.io.savemat(STTFileName, {"STTs": STTs})

                    # load data from file
                trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
                # period
                T = trvmat[-1, 0]
                # Take off last element which is same as first element up to
                # integration error tolerances (periodicity)
                trvmat = trvmat[:-1]

                STMmat = list(scipy.io.loadmat(STMFileName).values())[-1]
                STTmat = list(scipy.io.loadmat(STTFileName).values())[-1]
                # initialize object used for computation
                self.orb = OrbitVariationalDataSecondOrder(
                    STTmat, STMmat, trvmat, T, exponent
                )
            elif dynamics == 1:
                raise Exception("Unimplemented")
            elif dynamics == 2:
                raise Exception("Unimplemented")
            elif dynamics == 3:
                raise Exception("Unimplemented")

        elif mode == "impulsive":
            if dynamics == 0:

                def optControlDynamics():
                    x, y, z, vx, vy, vz = sympy.symbols("x,y,z,vx,vy,vz")
                    mu = 3.00348e-6
                    mu1 = 1.0 - mu
                    mu2 = mu
                    r1 = sympy.sqrt((x + mu2) ** 2 + (y**2) + (z**2))
                    r2 = sympy.sqrt((x - mu1) ** 2 + (y**2) + (z**2))
                    U = (-1.0 / 2.0) * (x**2 + y**2) - (mu1 / r1) - (mu2 / r2)
                    dUdx = sympy.diff(U, x)
                    dUdy = sympy.diff(U, y)
                    dUdz = sympy.diff(U, z)

                    RHS = sympy.Matrix(
                        [
                            vx,
                            vy,
                            vz,
                            ((-1 * dUdx) + 2 * vy),
                            ((-1 * dUdy) - 2 * vx),
                            (-1 * dUdz),
                        ]
                    )
                    variables = sympy.Matrix([x, y, z, vx, vy, vz])
                    return variables, RHS

                # store precomputed data in the following files
                fileName = self.precompfname
                trvFileName = os.path.join(EXOSIMS.__path__[0], f"{fileName}_trvs.mat")
                STMFileName = os.path.join(EXOSIMS.__path__[0], f"{fileName}_STMs.mat")
                # initial conditions for a sun-earth halo orbit
                # with zero initial conditions for costates and energy
                ics = [
                    1.00822114953991,
                    0.0,
                    -0.001200000000000000,
                    0.0,
                    0.010290010931740649,
                    0.0,
                ]
                T = 3.1002569555488506

                exponent = self.exponent

                # precompute variational data if data file does not already exist
                if not os.path.isfile(trvFileName):
                    variables, dynamics = optControlDynamics()
                    print("Precomputing variational data")
                    threeBodyInt = STMint(variables, dynamics, variational_order=1)
                    t_step = T / 2.0**exponent
                    curState = ics
                    states = [ics]
                    # STMs = [np.identity(12)]
                    # STTs = [np.zeros((12,12))]
                    STMs = []
                    tVals = np.linspace(0, T, num=2**exponent + 1)
                    for i in range(2**exponent):
                        print(f"Integrating: {i}")
                        [state, STM] = threeBodyInt.dynVar_int(
                            [0, t_step], curState, output="final", max_step=0.001
                        )
                        states.append(state)
                        STMs.append(STM[:6, :6])
                        curState = state
                    scipy.io.savemat(
                        trvFileName,
                        {"trvs": np.hstack((np.transpose(np.array([tVals])), states))},
                    )
                    scipy.io.savemat(STMFileName, {"STMs": STMs})
                # load data from file
                trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
                # period
                T = trvmat[-1, 0]
                # Take off last element which is same as first element up to
                # integration error tolerances (periodicity)
                trvmat = trvmat[:-1]
                STMmat = list(scipy.io.loadmat(STMFileName).values())[-1]
                # initialize object used for computation
                self.orb = OrbitVariationalDataFirstOrder(STMmat, trvmat, T, exponent)
            elif dynamics == 1:
                raise Exception("Unimplemented")
            elif dynamics == 2:
                raise Exception("Unimplemented")
            elif dynamics == 3:
                raise Exception("Unimplemented")
        else:
            raise Exception('Mode must be one of "energyOptimal" or "impuslive"')

        ObservatoryL2Halo.__init__(self, use_alt=True, **specs)

    def calculate_dV(self, TL, old_sInd, sInds, slewTimes, tmpCurrentTimeAbs):
        """calculates delta v costs for slews from the current star to target stars w/
        indices given by sInds.

        Args:
            TL (:ref:`TargetList`):
                the target list being used in survey simulation that contains all possible targets
            old_sInd (int):
                the index of the star that was previously being observed
            sInds (~numpy.ndarray(int)):
                indices of the stars to be obverseved
            slewTimes(~astropy.time.Time(~numpy.ndarray)):
                set of desired slewTimes between current star and stars to be observed
            tmpCurrentTimeAbs (~astropy.time.Time):
                current absolute mission time in mjd
        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float))
                Array of delta v costs associated with slews from the current star to targets stars w/ indices given by sInds.
        """

        # IWA = TL.OpticalSystem.IWA
        # d = self.starShadeRad / math.tan(IWA.value * math.pi / (180 * 3600))
        d = self.occulterSep

        slewTimes += np.random.rand(slewTimes.shape[0], slewTimes.shape[1]) / 100000
        if old_sInd is None:
            dV = np.zeros(slewTimes.shape)
        else:
            dV = np.zeros(slewTimes.shape)
            if isinstance(slewTimes, u.Quantity):
                badSlews_i, badSlew_j = np.where(slewTimes.value < self.occ_dtmin.value)
            else:
                badSlews_i, badSlew_j = np.where(slewTimes < self.occ_dtmin.value)
            t0 = tmpCurrentTimeAbs

            starPost0 = TL.starprop(old_sInd, t0, eclip=True)

            obsPost0 = self.orbit(t0, eclip=True)

            # as long as starPost0 and obsPostt0 are in the same cooridinate system, can ignore their scaling since we are only calculating a direction

            # in km
            starShadePost0InertRel = (
                d * (starPost0 - obsPost0) / np.linalg.norm(starPost0 - obsPost0)
            )
            # canonical unit
            t0Can = self.abs_to_can(t0[0])

            # in km
            starShadePost0SynRel = self.inert_to_syn(starShadePost0InertRel, t0Can)

            tfs = tmpCurrentTimeAbs + slewTimes
            tfs_flattened = tfs.flatten()

            starPosttfs = np.zeros(shape=(slewTimes.shape[0], slewTimes.shape[1], 3))
            for idx, ind in enumerate(sInds):
                # in AU
                starPosttfs[idx] = TL.starprop(ind, tfs[idx], eclip=True)

            # in AU
            obsPosttfs = self.orbit(tfs_flattened, eclip=True)

            for t in range(slewTimes.shape[1]):
                for i in range(slewTimes.shape[0]):
                    # gets final target star positions in heliocentric ecliptic inertial frame
                    starPostf = starPosttfs[i, t]
                    obsPostf = obsPosttfs[i * slewTimes.shape[1] + t].value

                    # km
                    starShadePostfInertRel = (
                        d
                        * (starPostf - obsPostf)
                        / np.linalg.norm(starPostf - obsPostf)
                    )

                    # canonical unit
                    tfCan = self.abs_to_can(tfs[i, t])

                    # km
                    starShadePostfSynRel = self.inert_to_syn(
                        starShadePostfInertRel, tfCan
                    )

                    if self.mode == "impulsive":

                        precomputeData = self.orb.precompute_lu(t0Can, tfCan)
                        dV[i, t] = self.orb.solve_deltaV_convenience(
                            precomputeData,
                            starShadePost0SynRel.value,
                            starShadePostfSynRel.value,
                        )
                    else:
                        precomputeData = self.orb.precompute_lu(t0Can, tfCan)
                        dV[i, t] = self.orb.deltaV(
                            precomputeData[0],
                            precomputeData[4],
                            precomputeData[5],
                            starShadePost0SynRel.value,
                            starShadePostfSynRel.value,
                            t0Can,
                            tfCan,
                        )

            dV[badSlews_i, badSlew_j] = np.Inf

            # must convert from AU / canonical time unit to m / s
        return (
            (dV * 149597870.7 * 1000 / ((365.2515 / (2 * math.pi))) / 86400) * u.m / u.s
        )

    def inert_to_syn(self, intertial_relative_pos, tcan):
        """Converts to inertial relative position coordinates to synodic relative
        position coordinates

        Args:
            intertial_relative_pos (~astropy.units.Quantity(~numpy.ndarray(float))):
                inertial relative position
            tcan (float):
                current canonical time in CRTBP units
        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                synodic relative position coordinates associate with the given inertial relative position coordinates
        """
        transformmat = np.array(
            [
                [np.cos(tcan), np.sin(tcan), 0],
                [-np.sin(tcan), np.cos(tcan), 0],
                [0, 0, 1],
            ]
        )
        return transformmat @ intertial_relative_pos.squeeze()

    def abs_to_can(self, tabs):
        """Converts mission times in days to canonical times in CRTBP units

        Args:
            tabs (~astropy.time.Time):
                mission time to be converted
        Returns:
            float:
                tabs in canonical CRTBP time units
        """
        return tabs.value / self.canonical_time_unit.value
