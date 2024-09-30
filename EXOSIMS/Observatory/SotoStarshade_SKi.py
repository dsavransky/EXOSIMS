from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import scipy.interpolate as interp
import time
import os
import pickle

EPS = np.finfo(float).eps


class SotoStarshade_SKi(SotoStarshade):
    """StarShade Observatory class

    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """

    def __init__(
        self, latDist=0.9, latDistOuter=0.95, latDistFull=1, axlDist=250, **specs
    ):

        SotoStarshade.__init__(self, **specs)

        self.latDist = latDist * u.m
        self.latDistOuter = latDistOuter * u.m
        self.latDistFull = latDistFull * u.m
        self.axlDist = axlDist * u.km

        # optical coefficients for SRP
        Bf = self.non_lambertian_coefficient_front  # non-Lambertian coefficient (front)
        Bb = self.non_lambertian_coefficient_back  # non-Lambertian coefficient (back)
        s = self.specular_reflection_factor  # specular reflection factor
        p = self.nreflection_coefficient  # nreflection coefficient
        ef = self.emission_coefficient_front  # emission coefficient (front)
        eb = self.emission_coefficient_back  # emission coefficient (back)

        # optical coefficients
        self.a1 = 0.5 * (1.0 - s * p)
        self.a2 = s * p
        self.a3 = 0.5 * (
            Bf * (1.0 - s) * p + (1.0 - p) * (ef * Bf - eb * Bb) / (ef + eb)
        )

        # Moon
        mM_ = 7.342e22 * u.kg  # mass of the moon
        self.mu_moon = (mM_ / (const.M_earth + const.M_sun + mM_)).to(
            ""
        )  # mass of the moon in Mass Units
        aM = 384748 * u.km  # radius of lunar orbit (assume circular)
        self.a_moon = self.convertPos_to_canonical(aM)
        self.i_moon = 5.15 * u.deg  # inclination of lunar orbit to ecliptic
        TM = 29.53 * u.d  # period of lunar orbit
        self.w_moon = 2 * np.pi / self.convertTime_to_canonical(TM)
        OTM = 18.59 * u.yr  # period of lunar nodal precession (retrograde)
        self.dO_moon = 2 * np.pi / self.convertTime_to_canonical(OTM)

        # Earth
        self.mu_earth = const.M_earth / (mM_ + const.M_earth + const.M_sun)
        self.a_earth = self.convertPos_to_canonical(mM_ / const.M_earth * aM)

    # =============================================================================
    # Unit conversions
    # =============================================================================

    # converting times
    def convertTime_to_canonical(self, dimTime):
        """Convert array of times from dimensional units to canonical units

        Method converts the times inside the array from the given dimensional
        unit (doesn't matter which, it converts to units of years in an
        intermediate step) into canonical units of the CR3BP. 1 yr = 2 pi TU
        where TU are the canonical time units.

        Args:
            dimTime (float numpy.ndarray):
                Array of times in some time unit

        Returns:
            canonicalTime (float numpy.ndarray):
                Array of times in canonical units
        """

        dimTime = dimTime.to("yr")
        canonicalTime = dimTime.value * (2 * np.pi)

        return canonicalTime

    def convertTime_to_dim(self, canonicalTime):
        """Convert array of times from canonical units to unit of years

        Method converts the times inside the array from canonical units of the
        CR3BP into year units. 1 yr = 2 pi TU where TU are the canonical time
        units.

        Args:
            canonicalTime (float numpy.ndarray):
                Array of times in canonical units

        Returns:
            dimTime (float numpy.ndarray):
                Array of times in units of years
        """

        canonicalTime = canonicalTime / (2 * np.pi)
        dimTime = canonicalTime * u.yr

        return dimTime

    # converting distances
    def convertPos_to_canonical(self, dimPos):
        """Convert array of positions from dimensional units to canonical units

        Method converts the positions inside the array from the given dimensional
        unit (doesn't matter which, it converts to units of AU in an
        intermediate step) into canonical units of the CR3BP. 1 au = 1 DU
        where DU are the canonical position units.

        Args:
            dimPos (float numpy.ndarray):
                Array of positions in some distance unit

        Returns:
            canonicalPos (float numpy.ndarray):
                Array of distance in canonical units
        """

        dimPos = dimPos.to("au")
        canonicalPos = dimPos.value

        return canonicalPos

    def convertPos_to_dim(self, canonicalPos):
        """Convert array of positions from canonical units to dimensional units

        Method converts the positions inside the array from canonical units of
        the CR3BP into units of AU.

        Args:
            canonicalPos (float numpy.ndarray):
                Array of distance in canonical units

        Returns:
            dimPos (float numpy.ndarray):
                Array of positions in units of AU
        """

        dimPos = canonicalPos * u.au

        return dimPos

    # converting velocity
    def convertVel_to_canonical(self, dimVel):
        """Convert array of velocities from dimensional units to canonical units

        Method converts the velocities inside the array from the given dimensional
        unit (doesn't matter which, it converts to units of AU/yr in an
        intermediate step) into canonical units of the CR3BP.

        Args:
            dimVel (float numpy.ndarray)):
                Array of velocities in some speed unit

        Returns:
            canonicalVel (float numpy.ndarray)):
                Array of velocities in canonical units
        """

        dimVel = dimVel.to("au/yr")
        canonicalVel = dimVel.value / (2 * np.pi)

        return canonicalVel

    def convertVel_to_dim(self, canonicalVel):
        """Convert array of velocities from canonical units to dimensional units

        Method converts the velocities inside the array from canonical units of
        the CR3BP into units of AU/yr.

        Args:
            canonicalVel (float numpy.ndarray):
                Array of velocities in canonical units

        Returns:
            dimVel (float numpy.ndarray):
                Array of velocities in units of AU/yr
        """

        canonicalVel = canonicalVel * (2 * np.pi)
        dimVel = canonicalVel * u.au / u.yr

        return dimVel

    # converting angular velocity
    def convertAngVel_to_canonical(self, dimAngVel):
        """Convert array of angular velocities from dimensional units to canonical units

        Method converts the angular velocities inside the array from the given
        dimensional unit (doesn't matter which, it converts to units of rad/yr
        in an intermediate step) into canonical units of the CR3BP.

        Args:
            dimAngVel (float numpy.ndarray):
                Array of angular velocities in some angular velocity unit

        Returns:
            canonicalAngVel (float numpy.ndarray):
                Array of angular velocities in canonical units
        """

        dimAngVel = dimAngVel.to("rad/yr")
        canonicalAngVel = dimAngVel.value / (2 * np.pi)

        return canonicalAngVel

    def convertAngVel_to_dim(self, canonicalAngVel):
        """Convert array of angular velocities from canonical units to dimensional units

        Method converts the angular velocities inside the array from canonical
        units of the CR3BP into units of rad/yr.

        Args:
            canonicalAngVel (float numpy.ndarray):
                Array of angular velocities in canonical units

        Returns:
            dimAngVel (float numpy.ndarray):
                Array of angular velocities in units of rad/yr
        """

        canonicalAngVel = canonicalAngVel * (2 * np.pi)
        dimAngVel = canonicalAngVel * u.rad / u.yr

        return dimAngVel

    # converting acceleration
    def convertAcc_to_canonical(self, dimAcc):
        """Convert array of accelerations from dimensional units to canonical units

        Method converts the accelerationss inside the array from the given
        dimensional unit (doesn't matter which, it converts to units of au/yr^2
        in an intermediate step) into canonical units of the CR3BP.

        Args:
            dimAcc (float numpy.ndarray):
                Array of accelerations in some acceleration unit

        Returns:
            canonicalAcc (float numpy.ndarray):
                Array of accelerations in canonical units
        """

        dimAcc = dimAcc.to("au/yr**2")
        canonicalAcc = dimAcc.value / (2 * np.pi) ** 2

        return canonicalAcc

    def convertAcc_to_dim(self, canonicalAcc):
        """Convert array of accelerations from canonical units to dimensional units

        Method converts the accelerations inside the array from canonical
        units of the CR3BP into units of au/yr^2.

        Args:
            canonicalAcc (float numpy.ndarray):
                Array of accelerations in canonical units

        Returns:
            dimAcc (float numpy.ndarray):
                Array of accelerations in units of AU/yr^2
        """

        canonicalAcc = canonicalAcc * (2 * np.pi) ** 2
        dimAcc = canonicalAcc * u.au / u.yr**2

        return dimAcc

    # converting angular accelerations
    def convertAngAcc_to_canonical(self, dimAngAcc):
        """Convert array of angular accelerations from dimensional units
        to canonical units

        Method converts the angular accelerationss inside the array from the given
        dimensional unit (doesn't matter which, it converts to units of rad/yr^2
        in an intermediate step) into canonical units of the CR3BP.

        Args:
            dimAngAcc (float numpy.ndarray):
                Array of angular accelerations in some angular acceleration unit

        Returns:
            canonicalAngAcc (float numpy.ndarray):
                Array of angular accelerations in canonical units
        """

        dimAngAcc = dimAngAcc.to("rad/yr^2")
        canonicalAngAcc = dimAngAcc.value / (2 * np.pi) ** 2

        return canonicalAngAcc

    def convertAngAcc_to_dim(self, canonicalAngAcc):
        """Convert array of angular accelerations from canonical units
        to dimensional units

        Method converts the angular accelerations inside the array from canonical
        units of the CR3BP into units of rad/yr^2.

        Args:
            canonicalAngAcc (float numpy.ndarray):
                Array of accelerations in canonical units

        Returns:
            dimAngAcc (float numpy.ndarray):
                Array of accelerations in units of rad/yr^2
        """

        canonicalAngAcc = canonicalAngAcc * (2 * np.pi) ** 2
        dimAngAcc = canonicalAngAcc * u.rad / u.yr**2

        return dimAngAcc

    # no more units!!
    def unitVector(self, p):
        """Normalizes an array and returns associated unit vector

        Takes in some array p that represents a vector with dimensions 3xn. It
        then calculates the norm of that vector and also normalizes it to
        create a unit vector.

        Args:
            p (float 3xn numpy.ndarray):
                Array of values

        Returns:
            tuple:
                p (float 3xn numpy.ndarray):
                    Unit vector associated with p, same dimensions
                pnorm (float numpy.ndarray):
                    Norm of the given vector for each value n
        """

        pnorm = np.linalg.norm(p, axis=0)
        p_ = p / pnorm

        return p_, pnorm

    # =============================================================================
    # Kinematics
    # =============================================================================

    def EulerAngleAndDerivatives(self, TL, sInd, currentTime, tRange=np.array([0])):
        """Calculates Euler angles and rates for LOS from telescope to star sInd

        This method calculates Euler angles defining the line of sight (LOS)
        from the telescope to some star sInd in the target list TL. The Euler
        angles are defined relative to some B-frame placed at the inertial
        location of the telescope on its halo orbit. Derivatives of the Euler
        angles, representing slewing rates of the LOS, are also calculated.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m

        Returns:
            tuple:
                theta (float m numpy.ndarray):
                    Azimuthal angle to define star LOS in rad
                phi (float m numpy.ndarray):
                    Polar angle to define star LOS in rad
                dtheta (float m numpy.ndarray):
                    Azimuthal angle to define star LOS in canonical units
                dphi (float m numpy.ndarray):
                    Polar angle to define star LOS in canonical units
        """

        # ecliptic coordinates and parallax of stars
        coords = TL.coords.transform_to("barycentrictrueecliptic")
        lamb = coords.lon[sInd]
        beta = coords.lat[sInd]
        varpi = TL.parx[sInd].to("rad")
        varpiValue = varpi.value

        # absolute times (Note: equinox is start time of Halo AND when inertial
        # frame and rotating frame match)
        absTimes = currentTime + tRange  # mission times  in jd
        modTimes = (
            np.mod(absTimes.value, self.equinox.value) * u.d
        )  # mission times relative to equinox )
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonical units

        # halo kinematics in rotating frame relative to Origin of R-frame (in au)
        haloPos = self.haloPosition(absTimes) + np.array([1, 0, 0]) * self.L2_dist.to(
            "au"
        )
        haloVel = self.haloVelocity(absTimes)

        # halo positions and velocities in canonical units
        xTR, yTR, zTR = np.array(
            [self.convertPos_to_canonical(haloPos[:, n]) for n in range(3)]
        )
        dxTR, dyTR, dzTR = np.array(
            [self.convertVel_to_canonical(haloVel[:, n]) for n in range(3)]
        )

        # converting halo to inertial frame coordinates AND derivatives
        xTI = xTR * np.cos(t) - yTR * np.sin(t)
        yTI = xTR * np.sin(t) + yTR * np.cos(t)
        zTI = zTR
        IdxTI = dxTR * np.cos(t) - dyTR * np.sin(t) - yTI
        IdyTI = dxTR * np.sin(t) + dyTR * np.cos(t) + xTI
        IdzTI = dzTR

        # find cartesian components in I frame of star location relative to telescope
        x_comp = np.cos(beta) * np.cos(lamb) - varpiValue * xTI
        y_comp = np.cos(beta) * np.sin(lamb) - varpiValue * yTI
        z_comp = np.sin(beta) - varpiValue * zTI
        r_comp = np.sqrt(x_comp**2 + y_comp**2)

        # find Euler angles theta and phi (azimuth and polar angles, respectively)
        theta = np.arctan2(y_comp, x_comp)
        phi = np.arctan2(r_comp, z_comp)

        # find Euler angle derivatives---angular rates of changing LOS from
        # telescope to star
        dtheta = varpiValue * (-IdxTI * np.sin(theta) + IdyTI * np.cos(theta))
        dphi = (
            varpiValue
            * (np.cos(phi) * (IdxTI * np.cos(theta) + IdyTI * np.sin(theta)) + IdzTI)
            / np.sin(phi)
        )

        return theta.value, phi.value, dtheta, dphi

    def Bframe(self, TL, sInd, currentTime, tRange=np.array([0])):
        """Calculates unit vectors defining B-frame of telescope

        The B-frame is placed at the inertial location of the telescope on its
        halo orbit. The third axis points directly towards the target star
        sInd. The second axis, by our definition, points parallel to the
        ecliptic plane of the Sun-Earth.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m

        Returns:
            tuple:
                b1_I (float 3xm numpy.ndarray):
                    First axis B-frame unit vector for each time in dimension m
                b2_I (float 3xm numpy.ndarray):
                    Second axis B-frame unit vector for each time in dimension m
                b3_I (float 3xm numpy.ndarray):
                    Third axis B-frame unit vector for each time in dimension m.
                    This one points towards the star sInd
        """

        # find the Euler angles pointing towards sInd for each time in tRange
        theta, phi, dtheta, dphi = self.EulerAngleAndDerivatives(
            TL, sInd, currentTime, tRange
        )

        # first axis of B-frame
        b1_I = np.array(
            [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)]
        )

        # second axis of B-frame
        b2_I = np.array([-np.sin(theta), np.cos(theta), np.zeros(len(theta))])

        # third axis of B-frame. this is the important one, points towards star
        b3_I = np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )

        return b1_I, b2_I, b3_I

    def starshadeKinematics(self, TL, sInd, currentTime, tRange=np.array([0])):
        """Calculates full kinematics of nominal starshade positioning at LOS

        This method calculates the full kinematics (positions, velocities, and
        accelerations) of the nominal starshade trajectory during an observation.
        The nominal trajectory is one that follows the changing LOS from
        telescope to star at a constant separation distance. Kinematics are given
        in inertial frame components and derivates are taken as inertial vector
        derivatives. Also returns the inertial kinematics relative to the
        telescope.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m

        Returns:
            r_S0_I (float 3xm array):
                Nominal position (r) of starshade (S) relative to inertial frame
                origin (0) given in inertial frame components (_I)
            Iv_S0_I (float 3xm array):
                Nominal inertial velocity (Iv) of starshade (S) relative to
                inertial frame origin (0) given in inertial frame components (_I)
            Ia_S0_I (float 3xm array):
                Nominal inertial acceleration (Ia) of starshade (S) relative to
                inertial frame origin (0) given in inertial frame components (_I)
            r_ST_I (float 3xm array):
                Nominal position (r) of starshade (S) relative to telescope
                location (T) given in inertial frame components (_I)
            Iv_ST_I (float 3xm array):
                Nominal inertial velocity (Iv) of starshade (S) relative to
                telescope location (T) given in inertial frame components (_I)
            Ia_ST_I (float 3xm array):
                Nominal inertial acceleration (Ia) of starshade (S) relative to
                telescope location (T) given in inertial frame components (_I)
        """

        varpi = TL.parx[sInd].to("rad")
        varpiValue = varpi.value

        # absolute times (Note: equinox is start time of Halo AND when inertial
        # frame and rotating frame match)
        absTimes = currentTime + tRange  # mission times  in jd
        modTimes = (
            np.mod(absTimes.value, self.equinox.value) * u.d
        )  # mission times relative to equinox )
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonical units

        s = self.convertPos_to_canonical(self.occulterSep)

        # halo kinematics in rotating frame relative to Origin of R-frame (in au)
        haloPos = self.haloPosition(absTimes) + np.array([1, 0, 0]) * self.L2_dist.to(
            "au"
        )
        haloVel = self.haloVelocity(absTimes)

        # halo positions and velocities in canonical units
        xTR, yTR, zTR = np.array(
            [self.convertPos_to_canonical(haloPos[:, n]) for n in range(3)]
        )
        dxTR, dyTR, dzTR = np.array(
            [self.convertVel_to_canonical(haloVel[:, n]) for n in range(3)]
        )

        xTI = xTR * np.cos(t) - yTR * np.sin(t)
        yTI = xTR * np.sin(t) + yTR * np.cos(t)
        zTI = zTR
        dxTI = dxTR * np.cos(t) - dyTR * np.sin(t) - yTI
        dyTI = dxTR * np.sin(t) + dyTR * np.cos(t) + xTI
        dzTI = dzTR

        # halo accelerations
        rTI = np.vstack([xTI, yTI, zTI, dxTI, dyTI, dzTI])
        ddxTI, ddyTI, ddzTI = self.equationsOfMotion_CRTBPInertial(
            t.value, rTI, TL, sInd
        )[3:6, :]

        # Euler angles
        theta, phi, dtheta, dphi = self.EulerAngleAndDerivatives(
            TL, sInd, currentTime, tRange
        )

        ddtheta = varpiValue * (ddyTI * np.cos(theta) - ddxTI * np.sin(theta))
        ddphi = (varpiValue / np.sin(phi)) * (
            np.cos(phi) * (ddxTI * np.cos(theta) + ddyTI * np.sin(theta)) + ddzTI
        )

        # starshade positions
        r_ST_I = np.array(
            [
                [s * np.sin(phi) * np.cos(theta)],
                [s * np.sin(phi) * np.sin(theta)],
                [s * np.cos(phi)],
            ]
        )[:, 0, :]

        Iv_ST_I = np.array(
            [
                [
                    s
                    * (
                        dphi * np.cos(phi) * np.cos(theta)
                        - dtheta * np.sin(phi) * np.sin(theta)
                    )
                ],
                [
                    s
                    * (
                        dphi * np.cos(phi) * np.sin(theta)
                        - dtheta * np.sin(phi) * np.cos(theta)
                    )
                ],
                [-s * dphi * np.sin(phi)],
            ]
        )[:, 0, :]

        Ia_ST_I = np.array(
            [
                [
                    s
                    * (
                        ddphi * np.cos(phi) * np.cos(theta)
                        - ddtheta * np.sin(phi) * np.sin(theta)
                    )
                ],
                [
                    s
                    * (
                        ddphi * np.cos(phi) * np.sin(theta)
                        - ddtheta * np.sin(phi) * np.cos(theta)
                    )
                ],
                [-s * ddphi * np.sin(phi)],
            ]
        )[:, 0, :]

        r_S0_I = np.array(
            [[xTI + r_ST_I[0, :]], [yTI + r_ST_I[1, :]], [zTI + r_ST_I[2, :]]]
        )[:, 0, :]

        Iv_S0_I = np.array(
            [[dxTI + Iv_ST_I[0, :]], [dyTI + Iv_ST_I[1, :]], [dzTI + Iv_ST_I[2, :]]]
        )[:, 0, :]

        Ia_S0_I = np.array(
            [[ddxTI + Ia_ST_I[0, :]], [ddyTI + Ia_ST_I[1, :]], [ddzTI + Ia_ST_I[2, :]]]
        )[:, 0, :]

        return r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I, Iv_ST_I, Ia_ST_I

    # =============================================================================
    # Ideal Dynamics
    # =============================================================================

    def starshadeIdealDynamics(
        self, TL, sInd, currentTime, tRange=np.array([0]), SRP=False, Moon=False
    ):
        """Calculates ideal dynamics of nominal starshade positioning at LOS

        This method calculates things to define ideal dynamics of a starshade
        under an nominal trajectory. The starshade is assumed to be on the
        nominal trajectory (on the LOS at some separation distance) and
        experiences gravity from the Sun and Earth. SRP and Moon forces can be
        included but are optional inputs. Method returns differential forces,
        the difference between the forces on the starshade and the acceleration
        it must have to remain on the nominal path. This difference pushes the
        starshade away from the nominal trajectory onto some offset trajectory.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force

        Returns:
            psi (float m array):
                The third Euler angle that completes the set. Roll angle that
                rotates B frame to some new frame where lateral component of dF
                points in the negative 2nd axis direction
            dfL_I (float 3xm array):
                Lateral component of the differential force on starshade in
                canonical units (lateral to LOS)
            dfA (float m array):
                Axial component of the differential force on starshade in
                canonical units (along LOS)
            df_S0_I (float 3xm array):
                Full differential force on starshade in canonical units (net
                force - nominal accelerations of S)
            f_S0_I (float 3xm array):
                Full net force on starshade in canonical units
        """

        # absolute times (Note: equinox is start time of Halo AND when inertial
        # frame and rotating frame match)
        absTimes = currentTime + tRange  # mission times  in jd
        modTimes = (
            np.mod(absTimes.value, self.equinox.value) * u.d
        )  # mission times relative to equinox
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonical units

        # B-frame unit vectors
        b1, b2, b3 = self.Bframe(TL, sInd, currentTime, tRange)

        # full starshade kinematics
        r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I, Iv_ST_I, Ia_ST_I = self.starshadeKinematics(
            TL, sInd, currentTime, tRange
        )

        # full state of starshade (pos + vel)
        s0 = np.vstack([r_S0_I, Iv_S0_I])

        # gravitational forces on the starshade at nominal position
        f_S0_I = self.equationsOfMotion_CRTBPInertial(
            t.value, s0, TL, sInd, integrate=False, SRP=SRP, Moon=Moon
        )[3:6, :]

        # differential force and components (axial and lateral)
        df_S0_I = f_S0_I - Ia_S0_I
        dfA = np.array([np.matmul(a, b) for a, b in zip(df_S0_I.T, b3.T)])
        dfL_I = df_S0_I - dfA * b3

        # components of lateral differential force on the b1-b2 plane
        dfL_b1 = np.array([np.matmul(a, b) for a, b in zip(dfL_I.T, b1.T)])
        dfL_b2 = np.array([np.matmul(a, b) for a, b in zip(dfL_I.T, b2.T)])

        # roll angle so that df_L points in negative 2nd axis in new frame
        psi = np.arctan2(dfL_b1, -dfL_b2)

        return psi, dfL_I, dfA, df_S0_I, f_S0_I

    def rotateComponents2NewFrame(
        self,
        TL,
        sInd,
        trajStartTime,
        s_int,
        t_int,
        final_frame="C",
        SRP=False,
        Moon=False,
    ):
        """Rotates state vector at different times into an ideal dynamics frame

        We introduce a new frame (the C-frame) rotated from the B-frame by an
        angle psi. Psi is found through the self.starshadeIdealDynamics. The
        C-frame is defined so that the lateral component of the differential
        force on the starshade always points down (in the -c2 direction). This
        method rotates a state vector s_int at every given respective time t_int.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            s_int (float 6xn array):
                Array of n state vectors with inertial velocities. Components
                are given in canonical units and are in either I-frame or C-frame.
            t_int (float numpy.ndarray):
                Array of times for each of the n state vectors in s_int given
                in canonical units
            final_frame (string):
                String entry that rotates states to the C-frame if input is 'C'
                or I-frame otherwise
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force

        Returns:
            r_f (float 3xn array):
                Array of n position vectors rotated to frame specified by
                final_frame
            Iv_f (float 3xn array):
                Array of n velocity vectors rotated to frame specified by
                final_frame
        """
        # assume that t_int is relative to the given trajStartTime or currentTime
        tRange = self.convertTime_to_dim(t_int)
        # Euler angles
        theta, phi, dtheta, dphi = self.EulerAngleAndDerivatives(
            TL, sInd, trajStartTime, tRange
        )
        psi, dfL_I, df_A, df_S0_I, f_S0_I = self.starshadeIdealDynamics(
            TL, sInd, trajStartTime, tRange, SRP=SRP, Moon=Moon
        )

        # extracting initial states
        x, y, z, dx, dy, dz = s_int
        r_i = np.vstack([x, y, z])
        Iv_i = np.vstack([dx, dy, dz])

        # defining final states
        r_f = np.zeros(r_i.shape)
        Iv_f = np.zeros(Iv_i.shape)

        for n in range(len(t_int)):
            AcI = self.rot(theta[n], 3)
            BcA = self.rot(phi[n], 2)
            CcB = self.rot(psi[n], 3)

            CcI = np.matmul(CcB, np.matmul(BcA, AcI))

            FcI = CcI if final_frame == "C" else CcI.T

            r_f[:, n] = np.matmul(FcI, r_i[:, n])
            Iv_f[:, n] = np.matmul(FcI, Iv_i[:, n])

        return r_f, Iv_f

    # =============================================================================
    # Equations of Motion
    # =============================================================================

    def equationsOfMotion_CRTBPInertial(
        self, t, state, TL, sInd, integrate=False, SRP=False, Moon=False
    ):
        """Equations of motion in inertial frame with CRTBP framework

        Equations of motion for an object under Sun and Earth's gravity. Forces
        and accelerations are framed relative to an inertial I-frame with origin
        at the Sun-Earth barycenter. Assumptions of the Circular Restricted
        Three Body Problem (CRTBP) are applied here, namely that the Earth and
        Sun orbit their common center of mass in circular orbits. All components
        and vectors are given in canonical units of the CRTBP. Two boolean
        inputs specify whether to add solar radiation pressure or lunar gravity
        as perturbation forces.

        Args:
            t (float):
                Times in normalized units
            state (float 6xn array):
                State vector consisting of stacked position and velocity vectors
                in normalized units
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            integrate (bool):
                If true, output array is flattened to ensure it is proper input
                in solve_ivp. Typically have it set to False if using solve_bvp
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force


        Returns:
            f_P0_I (float 6xn array):
                First derivative of the state vector consisting of stacked
                velocity and acceleration vectors in normalized units
        """

        x, y, z, dx, dy, dz = state
        r_P0_I = np.vstack([x, y, z])
        Iv_P0_I = np.vstack([dx, dy, dz])

        # positions of the Earth and Sun
        try:
            len(t)
        except TypeError:
            t = np.array([t])

        r_10_I = (
            -self.mu * np.array([[np.cos(t)], [np.sin(t)], [np.zeros(len(t))]])[:, 0, :]
        )
        r_20_I = (1 - self.mu) * np.array(
            [[np.cos(t)], [np.sin(t)], [np.zeros(len(t))]]
        )[:, 0, :]

        r_E2_I = (
            self.a_earth
            * np.array(
                [
                    [np.cos(self.w_moon * t)],
                    [np.sin(self.w_moon * t) * np.cos(0)],
                    [np.sin(self.w_moon * t) * np.sin(0)],
                ]
            )[:, 0, :]
        )
        r_E0_I = r_E2_I + r_20_I

        # relative positions of P
        r_P1_I = r_P0_I - r_10_I
        r_PE_I = r_P0_I - r_E0_I

        d_P1_I = np.linalg.norm(r_P1_I, axis=0)
        d_PE_I = np.linalg.norm(r_PE_I, axis=0)

        # equations of motion
        Ia_P0_I = (
            -(1 - self.mu) * r_P1_I / d_P1_I**3 - self.mu_earth * r_PE_I / d_PE_I**3
        )

        modTimes = self.convertTime_to_dim(t).to("d")
        absTimes = self.equinox + modTimes
        tRange = absTimes - absTimes[0] if len(t) > 0 else [0]

        if SRP:
            fSRP = self.SRPforce(TL, sInd, absTimes[0], tRange)
            Ia_P0_I += fSRP

        if Moon:
            fMoon = self.lunarPerturbation(TL, sInd, absTimes[0], tRange)
            Ia_P0_I += fMoon.value

        # full equations
        f_P0_I = np.vstack([Iv_P0_I, Ia_P0_I])

        if integrate:
            f_P0_I = f_P0_I.flatten()

        return f_P0_I

    def SRPforce(self, TL, sInd, currentTime, tRange, radius=36):
        """Solar radiation pressure force for starshade

        This method calculate the solar radiation pressure force on a starshade
        on a nominal trajectory aligned with some star sInd from the target
        list TL.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m
            radius (float array):
                Radius of the starshade in meter

        Returns:
            f_SRP (float 6xn array):
                Solar radiation pressure force in canonical units
        """

        # absolute times (Note: equinox is start time of Halo AND when inertial frame
        # and rotating frame match)
        absTimes = currentTime + tRange  # mission times  in jd
        modTimes = (
            np.mod(absTimes.value, self.equinox.value) * u.d
        )  # mission times relative to equinox )
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonical units

        b1, b2, b3 = self.Bframe(TL, sInd, currentTime, tRange)
        r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I, Iv_ST_I, Ia_ST_I = self.starshadeKinematics(
            TL, sInd, currentTime, tRange
        )

        # positions of the Earth and Sun
        r_10_I = (
            -self.mu * np.array([[np.cos(t)], [np.sin(t)], [np.zeros(len(t))]])[:, 0, :]
        )

        # relative positions of P
        r_S1_I = r_S0_I - r_10_I
        u_S1_I, d_S1 = self.unitVector(r_S1_I)

        cosA = np.array([np.matmul(a, b) for a, b in zip(u_S1_I.T, b3.T)])

        R = radius * u.m
        A = np.pi * R**2.0  # starshade cross-sectional area
        P0 = 4.563 * u.uN / u.m**2 * (1 / d_S1) ** 2
        PA = self.convertAcc_to_canonical(P0 * A / self.scMass)

        f_SRP = 2 * PA * cosA * (self.a1 * u_S1_I + (self.a2 * cosA + self.a3) * b3)

        return f_SRP

    def lunarPerturbation(self, TL, sInd, currentTime, tRange, nodalRegression=True):
        """Lunar gravity force for starshade

        This method calculates the lunar gravity force on a starshade
        on a nominal trajectory aligned with some star sInd from the target
        list TL. Assumes a perfectly circular lunar orbit about the Earth
        which is inclined at 5.15 degrees from the ecliptic plane and has a
        period of 29.53 days. We also include precession of the lunar nodes
        when calculating the lunar position.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            tRange (float ndarray):
                Array of times relative to currentTime to calculate values.
                The array has size m
            nodalRegression (bool):
                Dummy input

        Returns:
            f_Moon (float 6xn array):
                Lunar gravity force in canonical units
        """

        # absolute times (Note: equinox is start time of Halo AND when inertial frame
        # and rotating frame match)
        absTimes = currentTime + tRange  # mission times  in jd
        modTimes = (
            np.mod(absTimes.value, self.equinox.value) * u.d
        )  # mission times relative to equinox )
        t = (
            self.convertTime_to_canonical(modTimes) * u.rad
        )  # modTimes in canonicaEXOSIMS/Observatory/SotoStarshade_SKi.pyl units

        b1, b2, b3 = self.Bframe(TL, sInd, currentTime, tRange)
        r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I, Iv_ST_I, Ia_ST_I = self.starshadeKinematics(
            TL, sInd, currentTime, tRange
        )

        # positions of the Earth and Sun
        r_20_I = (1 - self.mu) * np.array(
            [[np.cos(t)], [np.sin(t)], [np.zeros(len(t))]]
        )[:, 0, :]

        r_32_I = (
            -self.a_moon
            * np.array(
                [
                    [
                        np.sin(self.dO_moon * t)
                        * np.sin(self.w_moon * t)
                        * np.cos(self.i_moon)
                        + np.cos(self.dO_moon * t) * np.cos(self.w_moon * t)
                    ],
                    [
                        -np.sin(self.dO_moon * t) * np.cos(self.w_moon * t)
                        + np.sin(self.w_moon * t)
                        * np.cos(self.i_moon)
                        * np.cos(self.dO_moon * t)
                    ],
                    [np.sin(self.w_moon * t) * np.sin(self.i_moon)],
                ]
            )[:, 0, :]
        )  # already assume retrograde lunar nodal precession
        r_30_I = r_32_I + r_20_I

        # relative positions of P
        r_S3_I = r_S0_I - r_30_I
        u_S3_I, d_S3 = self.unitVector(r_S3_I)

        f_Moon = -self.mu_moon * r_S3_I / d_S3**3

        return f_Moon

    def equationsOfMotion_aboutS(
        self, t, state, TL, sInd, trajStartTime, integrate=False, SRP=False, Moon=False
    ):
        """Equations of motion of starshade relative to nominal trajectory

        Equations of motion for a starshade relative to the nominal trajectory,
        which is defined as following the LOS perfectly to a star sInd from
        target list TL. Motion is defined relative to the nominal point S; the
        offset motion is labeled as O and origin of the solar system barycenter
        is 0. All components are given in inertial frame components, all vector
        derivatives are inertial frame derivatives. Units are canonical units.

        Args:
            t (float):
                Times in normalized units
            state (float 6xn array):
                State vector consisting of stacked position and velocity vectors
                in normalized units
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            integrate (bool):
                If true, output array is flattened to ensure it is proper input
                in solve_ivp. Typically have it set to False if using solve_bvp
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force


        Returns:
            ds (float 6xn array):
                First derivative of the state vector consisting of stacked
                relative velocity and acceleration vectors in normalized units
        """

        # original state of S' wrt to S in R-frame components and derivatives
        x, y, z, dx, dy, dz = state

        r_OS_I = state[0:3]
        Iv_OS_I = state[3:6]

        # reference epoch in canonical units
        t0 = self.convertTime_to_canonical(
            np.mod(trajStartTime.value, self.equinox.value) * u.d
        )[0]
        tF = t0 + t

        # current time in dimensional units
        currentTime = trajStartTime + self.convertTime_to_dim(t)

        # retrieving acceleration of point S rel O at currentTime
        r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I, Iv_ST_I, Ia_ST_I = self.starshadeKinematics(
            TL, sInd, currentTime
        )

        # determining position and velocity of S' rel O at currentTime
        r_O0_I = r_S0_I[:, 0] + r_OS_I
        Iv_O0_I = Iv_S0_I[:, 0] + Iv_OS_I
        s0 = np.hstack([r_O0_I, Iv_O0_I])

        # calculating force on S' at currentTime
        f_O0_I = self.equationsOfMotion_CRTBPInertial(
            tF, s0, TL, sInd, integrate=True, SRP=SRP, Moon=Moon
        )[3:6]

        # setting final second derivatives and stuff
        dr = [dx, dy, dz]
        ddr = f_O0_I - Ia_S0_I.flatten()
        ds = np.vstack([dr, ddr])
        ds = ds.flatten()

        return ds

    # =============================================================================
    # Playing Tennis
    # =============================================================================

    def crossThreshholdEvent(
        self, t, s, TL, sInd, trajStartTime, latDist, SRP=False, Moon=False
    ):
        """Event function for when starshade crosses deadbanding limit

        This method is used as an event function in solve_ivp and returns the
        current distance of the starshade centroid from the lateral deadbanding
        limit for observations. Takes an input latDist which can be changed
        if the user selects inner and outer thresholds.

        Args:
            t (float):
                Times in normalized units
            s (float 6xn array):
                State vector consisting of stacked position and velocity vectors
                in normalized units
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            latDist (float Quantity):
                The lateral deadbanding boundary for observations in meters
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force

        Returns:
            distanceFromLim (float):
                Distance from the deadbanding radius
        """
        currentTime = trajStartTime + self.convertTime_to_dim(t)
        # converting state vectors from R-frame to C-frame
        r_OS_C, Iv_OS_C = self.rotateComponents2NewFrame(
            TL, sInd, currentTime, s, np.array([0]), final_frame="C", SRP=SRP, Moon=Moon
        )
        # converting units to meters
        r_OS_C_dim = self.convertPos_to_dim(r_OS_C).to("m")

        # distance from S (in units of m)
        delta_r = r_OS_C_dim[0:2, :]
        dr = np.linalg.norm(delta_r, axis=0)

        # distance from inner threshhold
        distanceFromLim = (dr - latDist).value
        # self.vprint(latDist,distanceFromLim)
        return distanceFromLim

    def drift(
        self,
        TL,
        sInd,
        trajStartTime,
        dt=20 * u.min,
        freshStart=True,
        s0=None,
        fullSol=False,
        SRP=False,
        Moon=False,
    ):
        """Method to simulate drift between deadbanding burns for a starshade

        This method simulates drifting between deadbanding burns during a
        starshade observation. Creates event functions for lateral deadbanding
        threshold crossings, both with inner and outer thresholds. Integrates
        relative equations of motion until event is triggered and resolves that
        event to see where the crossing happened.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            dt (float Quantity):
                The initial guess for lateral drift time in minutes
            freshStart (bool):
                Toggles whether starshade starts at the optimal initial point
                if True or at some given s0 if False
            s0 (float 6 ndarray):
                The initial state of the drift, set to None as default. Given in
                canonical units and in I-frame components
            fullSol (bool):
                Optional flag, default False, set True to return additional information
                Returns full state solutions if True or just the crossing states
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force

        Returns:
            tuple:
                cross (float):
                    Flag where 0,1,or 2 mean Tolerance Not Crossed, Lateral Cross,
                    or Axial Cross
                driftTime (float astropy.units.Quantity):
                    Amount of time between threshold crossings in minutes
                t_cross (float numpy.ndarray):
                    Time of lateral limit crossing in canonical units.
                    If fullSol is True, this is t_full - Full time history of drift in
                    canonical units
                r_cross (float numpy.ndarray):
                    Position of lateral limit crossing in canonical units in C-frame
                    components.
                    If fullSol is True, this is r_full - Full position history of drift
                    in canonical units in C-frame  components
                v_cross (float numpy.ndarray):
                    Velocity of lateral limit crossing in canonical units in C-frame
                    components but inertial derivatives
                    If fullSol is True, this is v_full - Full velocity history of drift
                    in canonical units in C-frame components but inertial derivatives

        """

        # defining equations of motion
        EoM = lambda t, s: self.equationsOfMotion_aboutS(
            t, s, TL, sInd, trajStartTime, integrate=True, SRP=SRP, Moon=Moon
        )

        # event function for crossing inner boundary
        crossInner = lambda t, s: self.crossThreshholdEvent(
            t, s, TL, sInd, trajStartTime, self.latDist
        )
        crossInner.terminal = False
        crossInner.direction = 1

        # event function for crossing outer boundary
        crossOuter = lambda t, s: self.crossThreshholdEvent(
            t, s, TL, sInd, trajStartTime, self.latDistOuter
        )
        crossOuter.terminal = (
            True  # this one is terminal, should end integration after trigger
        )
        crossOuter.direction = 1

        # defining times
        t0 = self.convertTime_to_canonical(
            np.mod(trajStartTime.value, self.equinox.value) * u.d
        )[0]
        tF = t0 + self.convertTime_to_canonical(dt)
        tInt = np.linspace(0, tF - t0, 5000)

        # remember that these states are relative to the nominal track S/0
        # either start exactly on nominal track (freshStart) or else place it yourself
        if freshStart:
            r0_C, v0_C = self.starshadeInjectionVelocity(TL, sInd, trajStartTime, SRP)
            s0_C = np.hstack([r0_C, v0_C])
            # rotate to inertial frame
            r0_I, v0_I = self.rotateComponents2NewFrame(
                TL,
                sInd,
                trajStartTime,
                s0_C,
                np.array([0]),
                SRP=SRP,
                Moon=Moon,
                final_frame="I",
            )
            s0 = np.hstack([r0_I.flatten(), v0_I.flatten()])

        # integrate forward to t0 + dt
        res = solve_ivp(
            EoM,
            [tInt[0], tInt[-1]],
            s0,
            t_eval=tInt,
            events=(
                crossInner,
                crossOuter,
            ),
            rtol=1e-13,
            atol=1e-13,
            method="Radau",
        )
        t_int = res.t
        y_int = res.y
        x, y, z, dx, dy, dz = y_int

        # gotta rotate trajectory array from R-frame to C-frame components
        r_OS_C, Iv_OS_C = self.rotateComponents2NewFrame(
            TL, sInd, trajStartTime, y_int, t_int, SRP=SRP, Moon=Moon, final_frame="C"
        )

        # there should be 2 events, even if they weren't triggered (they'd be empty)
        t_innerEvent = res.t_events[0]
        t_outerEvent = res.t_events[1]
        s_innerEvent = res.y_events[0][-1] if t_innerEvent.size > 0 else np.zeros(6)
        s_outerEvent = res.y_events[1][0] if t_outerEvent.size > 0 else np.zeros(6)
        r_C_outer = -np.ones(3)
        if t_outerEvent.size > 0:
            r_C_outer, v_C_outer = self.rotateComponents2NewFrame(
                TL,
                sInd,
                trajStartTime,
                s_outerEvent,
                t_outerEvent,
                SRP=SRP,
                Moon=Moon,
                final_frame="C",
            )

        runItAgain = False
        # outer event triggered
        if t_outerEvent.size > 0:
            # outer event triggered in positive y -> we need to burn!
            if r_C_outer[1] > 0:
                self.vprint("crossed outer")
                cross = 2
                driftTime = self.convertTime_to_dim(t_outerEvent[0]).to("min")
                # rotate states at crossing event to C frame
                r_cross, v_cross = self.rotateComponents2NewFrame(
                    TL,
                    sInd,
                    trajStartTime,
                    s_outerEvent,
                    t_outerEvent,
                    SRP=SRP,
                    Moon=Moon,
                    final_frame="C",
                )
                if fullSol:
                    tEndInd = np.where(t_int <= t_outerEvent)[0][-1]
                    t_input = t_int[0:tEndInd]
                    s_input = y_int[:, 0:tEndInd]
                    s_interp = interp.interp1d(t_input, s_input)

                    t_full = np.linspace(t_input[0], t_input[-1], len(tInt))
                    s_interped = s_interp(t_full)
                    r_full, v_full = self.rotateComponents2NewFrame(
                        TL,
                        sInd,
                        trajStartTime,
                        s_interped,
                        t_full,
                        SRP=SRP,
                        Moon=Moon,
                        final_frame="C",
                    )

            # outer event triggered in negative y -> let's see what happens....
            else:
                # if we trigger in negative y, we should then just resolve the
                # inner boundary crossing instead
                # inner event triggered AND => outer event triggered only in negative y
                if t_innerEvent.size > 0:
                    if np.any(t_innerEvent) > 0:
                        self.vprint("crossed inner")
                        cross = 1
                        driftTime = self.convertTime_to_dim(t_innerEvent[-1]).to("min")
                        # rotate states at crossing event to C frame
                        t_inner = (
                            t_innerEvent
                            if t_innerEvent.size == 0
                            else np.array([t_innerEvent[-1]])
                        )
                        r_cross, v_cross = self.rotateComponents2NewFrame(
                            TL,
                            sInd,
                            trajStartTime,
                            s_innerEvent,
                            t_inner,
                            SRP=SRP,
                            Moon=Moon,
                            final_frame="C",
                        )
                        if fullSol:
                            tEndInd = np.where(t_int <= t_inner)[0][-1]
                            t_input = t_int[0:tEndInd]
                            s_input = y_int[:, 0:tEndInd]
                            s_interp = interp.interp1d(t_input, s_input)

                            t_full = np.linspace(t_input[0], t_input[-1], len(tInt))
                            s_interped = s_interp(t_full)
                            r_full, v_full = self.rotateComponents2NewFrame(
                                TL,
                                sInd,
                                trajStartTime,
                                s_interped,
                                t_full,
                                SRP=SRP,
                                Moon=Moon,
                                final_frame="C",
                            )
                    else:
                        estr = "ERROR: crossed outer but not inner crossing broke!"
                        self.vprint(estr)
                        # let's check if the inner boundary was triggered badly by the
                        # integrator
                        ds = np.linalg.norm(r_OS_C[0:2, :], axis=0)
                        ds_dim = self.convertPos_to_dim(ds).to("m")

                        if np.any(ds_dim[1:] > self.latDist):
                            self.vprint("something DID break")
                            outsideInnerRadius = np.where(ds_dim > self.latDist)[0]

                            if outsideInnerRadius.size > 0:
                                self.vprint("crossed inner - fixed")
                                jumpsInInnerRadius = np.diff(outsideInnerRadius)
                                lastInnerCrossing = outsideInnerRadius[
                                    int(np.where(jumpsInInnerRadius > 1)[0][-1] + 1)
                                ]
                                # we got em
                                cross = 1
                                y_int = y_int[:, 0:lastInnerCrossing]
                                t_int = t_int[0:lastInnerCrossing]
                                t_inner = np.array([t_int[-1]])
                                driftTime = self.convertTime_to_dim(t_int[-1]).to("min")
                                # rotate states at crossing event to C frame
                                r_cross, v_cross = self.rotateComponents2NewFrame(
                                    TL,
                                    sInd,
                                    trajStartTime,
                                    y_int[:, -1],
                                    t_inner,
                                    SRP=SRP,
                                    Moon=Moon,
                                    final_frame="C",
                                )

                                if fullSol:
                                    s_interp = interp.interp1d(t_int, y_int)

                                    t_full = np.linspace(t_int[0], t_int[-1], len(tInt))
                                    s_interped = s_interp(t_full)
                                    r_full, v_full = self.rotateComponents2NewFrame(
                                        TL,
                                        sInd,
                                        trajStartTime,
                                        s_interped,
                                        t_full,
                                        SRP=SRP,
                                        Moon=Moon,
                                        final_frame="C",
                                    )

                            else:
                                runItAgain = True

                        else:
                            runItAgain = True
                # something messed up with the event function!
                # we triggered outer function in the negative y direction
                # BUT somehow didnt trigger the inner boundary?!?!
                # I swear this happens.
                else:
                    self.vprint("ERROR: crossed outer but not inner!!")
                    # let's check if the inner boundary was supposed to be
                    # triggered but wasn't caught by the integrator
                    ds = np.linalg.norm(r_OS_C[0:2, :], axis=0)
                    ds_dim = self.convertPos_to_dim(ds).to("m")

                    if np.any(ds_dim[1:] > self.latDist):
                        self.vprint("something DID break")
                        outsideInnerRadius = np.where(ds_dim > self.latDist)[0]

                        if outsideInnerRadius.size > 0:
                            self.vprint("crossed inner - fixed")
                            jumpsInInnerRadius = np.diff(outsideInnerRadius)
                            lastInnerCrossing = outsideInnerRadius[
                                int(np.where(jumpsInInnerRadius > 1)[0][-1] + 1)
                            ]
                            # we got em
                            cross = 1
                            y_int = y_int[:, 0:lastInnerCrossing]
                            t_int = t_int[0:lastInnerCrossing]
                            t_inner = np.array([t_int[-1]])
                            driftTime = self.convertTime_to_dim(t_int[-1]).to("min")
                            # rotate states at crossing event to C frame
                            r_cross, v_cross = self.rotateComponents2NewFrame(
                                TL,
                                sInd,
                                trajStartTime,
                                y_int[:, -1],
                                t_inner,
                                SRP=SRP,
                                Moon=Moon,
                                final_frame="C",
                            )

                            if fullSol:
                                s_interp = interp.interp1d(t_int, y_int)

                                t_full = np.linspace(t_int[0], t_int[-1], len(tInt))
                                s_interped = s_interp(t_full)
                                r_full, v_full = self.rotateComponents2NewFrame(
                                    TL,
                                    sInd,
                                    trajStartTime,
                                    s_interped,
                                    t_full,
                                    SRP=SRP,
                                    Moon=Moon,
                                    final_frame="C",
                                )

                        else:
                            runItAgain = True

                    else:
                        runItAgain = True

        # outer event not triggered at all
        else:
            # BUT inner event WAS triggered.
            if t_innerEvent.size > 0:
                self.vprint("crossed inner")
                cross = 1
                driftTime = self.convertTime_to_dim(t_innerEvent[-1]).to("min")
                # rotate states at crossing event to C frame
                t_inner = (
                    t_innerEvent
                    if t_innerEvent.size == 0
                    else np.array([t_innerEvent[-1]])
                )
                r_cross, v_cross = self.rotateComponents2NewFrame(
                    TL,
                    sInd,
                    trajStartTime,
                    s_innerEvent,
                    t_inner,
                    SRP=SRP,
                    Moon=Moon,
                    final_frame="C",
                )

                if fullSol:
                    tEndInd = np.where(t_int <= t_inner)[0][-1]
                    t_input = t_int[0:tEndInd]
                    s_input = y_int[:, 0:tEndInd]
                    s_interp = interp.interp1d(t_input, s_input)

                    t_full = np.linspace(t_input[0], t_input[-1], len(tInt))
                    s_interped = s_interp(t_full)
                    r_full, v_full = self.rotateComponents2NewFrame(
                        TL,
                        sInd,
                        trajStartTime,
                        s_interped,
                        t_full,
                        SRP=SRP,
                        Moon=Moon,
                        final_frame="C",
                    )

            # no events were triggered, need to run drift for longer
            else:
                runItAgain = True

        if runItAgain:
            self.vprint("crossed nothing")
            cross = 0
            driftTime = self.convertTime_to_dim(t_int[-1]).to("min")
            r_cross = r_OS_C[:, -1]
            v_cross = Iv_OS_C[:, -1]

            r_full = r_OS_C
            v_full = Iv_OS_C
            t_full = t_int

        if fullSol:
            return cross, driftTime, t_full, r_full, v_full
        else:
            return cross, driftTime, t_int, r_cross, v_cross

    def guessAParabola(
        self,
        TL,
        sInd,
        trajStartTime,
        r_OS_C,
        Iv_OS_C,
        latDist=0.9 * u.m,
        fullSol=False,
        SRP=False,
        Moon=False,
        axlBurn=True,
    ):
        """Method to simulate ideal starshade drift with parabolic motion

        This method assumes an ideal, unperturbed trajectory in between lateral
        deadbanding burns. It assumes that the differential lateral force is
        constant throughout the entire trajectory and therefore motion is parabolic.
        Everything is calculated in C-frame components but I-frame derivatives.


        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            r_OS_C (float Quantity):
                The initial guess for lateral drift time in minutes
            Iv_OS_C (bool):
                Toggles whether starshade starts at the optimal initial point
                if True or at some given s0 if False
            latDist (float 6 ndarray):
                The initial state of the drift, set to None as default. Given in
                canonical units and in I-frame components
            fullSol (bool):
                Optional flag, default False, set True to return additional information:
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force
            axlBurn (bool):
                Set True for purely axial burn (no dz velocity). Defaults true.

        Returns:
            tuple:
                dt_newTOF (float):
                    New time of flight for parabolic trajectory in canonical units of
                    CRTBP
                Iv_PS_C_newIC (float 3 numpy.ndarray):
                    New velocity of parabolic trajectory (P) relative to nominal
                    starshade (S) starting at previous lateral burn in canonical
                    units of CRTBP
                dv_dim (float numpy.ndarray):
                    Delta-v of initial lateral burn in dimensional units of m/s
                r_PS_C (float ndarray):
                    Full time history of drift in canonical units (just x-y plane
                    of the C-frame in canonical units of CRTBP). Only returned if
                    fullSol is True.

        """

        psi, dfL_I, df_A, df_S0_I, f_S0_I = self.starshadeIdealDynamics(
            TL, sInd, trajStartTime, SRP=SRP, Moon=Moon
        )

        a = np.linalg.norm(dfL_I)
        DU = self.convertPos_to_canonical(latDist)
        VU = np.sqrt(a * DU)
        TU = np.sqrt(DU / a)

        # converting initial state from canonical units to C-frame units
        x0, y0, z0 = r_OS_C[:, -1] / DU
        dx0_, dy0_, dz0_ = Iv_OS_C[:, -1] / VU

        # finding intercept point
        if y0 > 0.5:
            xi = x0
            yi = y0
        else:
            xi = (
                np.sqrt(1 + y0) / np.sqrt(2)
                if x0 > 0
                else -np.sqrt(1 + y0) / np.sqrt(2)
            )
            yi = np.sqrt(1 - y0) / np.sqrt(2)

        # initial velocities of the second parabola
        dx0 = (
            np.sqrt(yi * (1 - yi)) / np.sqrt(2)
            if x0 < 0
            else -np.sqrt(yi * (1 - yi)) / np.sqrt(2)
        )
        dy0 = -dx0 * xi / yi + (xi - x0) / dx0 if dx0 != 0 else 0

        # tof and velocities if we're really close to the well
        if np.abs(x0) < 0.08:
            dy0 = (
                (3 * yi - 1) * np.sqrt((1 + yi) / (2 * yi))
                if y0 < 0.5
                else (1 - yi) * np.sqrt((1 + yi) / (2 * yi))
            )

            ymax = dy0**2 / 2 + y0
            tof = np.sqrt(2) * (np.sqrt(ymax - y0) + np.sqrt(ymax + 1))
            dx0 = -x0 / tof
        else:
            # time of flight in C-frame units
            tof = np.abs(x0 / dx0)

        # if burnHard:
        #     dy0 = -(3*yi-1)*np.sqrt( (1+yi)/(2*yi) )
        #     dx0 = -np.sign(x0) * (3*yi-1)*np.sqrt( (1+yi)/(2*yi) )
        #     tof = 4

        # planar parabolic trajectories in C-frame units
        # time range from 0 to full time of flight
        t = np.linspace(0, tof, 1000)

        # x and y throughout entire new parabolic trajectory
        xP = dx0 * t + x0
        yP = -0.5 * t**2 + dy0 * t + y0

        r_PS_C = np.vstack([xP, yP]) * DU
        if axlBurn:
            Iv_PS_C_newIC = np.array([dx0, dy0, 0]) * VU
        else:
            Iv_PS_C_newIC = np.array([dx0, dy0, dz0_]) * VU
        dt_newTOF = tof * TU

        # calculate delta v
        sNew_C = np.hstack([r_OS_C[:, -1], Iv_PS_C_newIC])
        r0new, v0new_I = self.rotateComponents2NewFrame(
            TL,
            sInd,
            trajStartTime,
            sNew_C,
            np.array([0]),
            SRP=SRP,
            Moon=Moon,
            final_frame="I",
        )

        sOld_C = np.hstack([r_OS_C[:, -1], Iv_OS_C[:, -1]])
        r0old, v0old_I = self.rotateComponents2NewFrame(
            TL,
            sInd,
            trajStartTime,
            sOld_C,
            np.array([0]),
            SRP=SRP,
            Moon=Moon,
            final_frame="I",
        )
        dv = np.linalg.norm(v0new_I - v0old_I)
        dv_dim = self.convertVel_to_dim(dv).to("m/s")

        if fullSol:
            return dt_newTOF, Iv_PS_C_newIC, dv_dim, r_PS_C
        else:
            return dt_newTOF, Iv_PS_C_newIC, dv_dim

    def starshadeInjectionVelocity(
        self, TL, sInd, trajStartTime, SRP=False, Moon=False
    ):
        """Method to find injection velocity of starshade to start observation

        This method returns the ideal injection velocity and position of a starshade
        to begin an observation with a star sInd from target list TL. Position
        and velocity are given in C-frame components but I-frame derivatives.


        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force

        Returns:
            r_OS_C (float):
                New position of offset trajectory (O) relative to nominal
                starshade (S) starting at previous lateral burn in canonical
                units of CRTBP
            Iv_OS_C_newIC (float 3 ndarray):
                New velocity of offset trajectory (O) relative to nominal
                starshade (S) starting at previous lateral burn in canonical
                units of CRTBP and inertial derivatives

        """

        latDist = self.latDist
        startingY = self.convertPos_to_canonical(latDist)
        r_OS_C = np.array([0, -1, 0]).reshape(3, 1) * startingY
        Iv_OS_C = np.array([1, 1, 1]).reshape(3, 1) * startingY
        dt, Iv_OS_C_newIC, dv_dim = self.guessAParabola(
            TL,
            sInd,
            trajStartTime,
            r_OS_C,
            Iv_OS_C,
            latDist=latDist,
            fullSol=False,
            SRP=SRP,
            Moon=Moon,
        )

        r_OS_C = r_OS_C.flatten()
        return r_OS_C, Iv_OS_C_newIC

    def stationkeep(
        self,
        TL,
        sInd,
        trajStartTime,
        dt=30 * u.min,
        simTime=1 * u.hr,
        SRP=False,
        Moon=False,
        axlBurn=True,
    ):
        """Method to simulate full stationkeeping with a given star

        This method simulates a full observation for a star sInd in target list
        TL. It calculates drifts in a sequence until the allotted simulation
        time simTime is over. It then logs various metrics including delta-v,
        drift times and number of thruster firings to be catalogued by the user.

        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer index of some target star
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            dt (float Quantity):
                First guess of trajectory drift time in units of minutes
            simTime (float Quantity):
                Total simulated observation time in units of hours
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force
            axlBurn (bool):
                Set True for purely axial burn (no dz velocity). Defaults true.

        Returns:
            tuple:
                nBounces (float):
                    Number of thruster firings throughout observation
                timeLeft (float Quantity):
                    Amount of time left when simulation ended in units of hours
                dvLog (float n Quantity):
                    Log of delta-v's with size n where n is equal to nBounces and
                    units of m/s
                dvAxialLog (float n Quantity):
                    Log of delta-v's purely in the axial direction with size n
                    where n is equal to nBounces and units of m/s
                driftLog (float n Quantity):
                    Log of all drift times with size n where n is equal to nBounces
                    and units of minutes
        """
        # drift for the first time, calculates correct injection speed within
        cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(
            TL,
            sInd,
            trajStartTime,
            dt=dt,
            freshStart=True,
            fullSol=True,
            SRP=SRP,
            Moon=Moon,
        )

        # counter for re-do's
        reDo = 0
        # if we didn't cross any threshold, let's increase the drift time
        if cross == 0:
            while cross == 0:
                self.vprint("redo!")
                # increase time
                dt *= 4
                cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(
                    TL,
                    sInd,
                    trajStartTime,
                    dt=dt,
                    freshStart=True,
                    fullSol=True,
                    SRP=SRP,
                    Moon=Moon,
                )

                # augment counter
                reDo += 1
                # if we've gone through this 5 times, something is probably wrong...
                if reDo > 5:
                    break

        # initiate arrays to log results and times
        timeLeft = simTime - driftTime
        nBounces = 1
        dvLog = np.array([])
        dvAxialLog = np.array([])
        driftLog = np.array([driftTime.to("min").value])

        # running deadbanding simulation
        while timeLeft.to("min").value > 0:
            self.vprint(timeLeft.to("min"))
            trajStartTime += driftTime
            latDist = (
                self.latDist if cross == 1 else self.latDistOuter if cross == 2 else 0
            )

            dt_new, Iv_PS_C_newIC, dv_dim, r_PS_C = self.guessAParabola(
                TL,
                sInd,
                trajStartTime,
                r_OS_C,
                Iv_OS_C,
                latDist,
                fullSol=True,
                SRP=SRP,
                Moon=Moon,
                axlBurn=axlBurn,
            )
            dt_newGuess = self.convertTime_to_dim(dt_new).to("min") * 2

            s0_Cnew = np.hstack([r_OS_C[:, -1], Iv_PS_C_newIC])
            r0, v0 = self.rotateComponents2NewFrame(
                TL,
                sInd,
                trajStartTime,
                s0_Cnew,
                np.array([0]),
                SRP=SRP,
                Moon=Moon,
                final_frame="I",
            )
            s0_new = np.hstack([r0.flatten(), v0.flatten()])

            cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(
                TL,
                sInd,
                trajStartTime,
                dt=dt_newGuess,
                freshStart=False,
                s0=s0_new,
                fullSol=True,
                SRP=SRP,
                Moon=Moon,
            )

            reDo = 0
            if cross == 0:
                while cross == 0:
                    self.vprint("redo!")
                    dt_newGuess *= 4
                    cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(
                        TL,
                        sInd,
                        trajStartTime,
                        dt=dt_newGuess,
                        freshStart=False,
                        s0=s0_new,
                        fullSol=True,
                        SRP=SRP,
                        Moon=Moon,
                    )

                    reDo += 1
                    if reDo > 5:
                        break

            # update everything
            nBounces += 1
            timeLeft -= driftTime
            dvLog = np.hstack([dvLog, dv_dim.to("m/s").value])
            driftLog = np.hstack([driftLog, driftTime.to("min").value])
            dvAxialLog = np.hstack(
                [
                    dvAxialLog,
                    self.convertVel_to_dim(np.abs(Iv_OS_C[2, -1])).to("m/s").value,
                ]
            )

        dvLog = dvLog * u.m / u.s
        dvAxialLog = dvAxialLog * u.m / u.s
        driftLog = driftLog * u.min
        axDriftLog = self.convertPos_to_dim(np.abs(r_OS_C[2, -1])).to("km")

        return nBounces, timeLeft, dvLog, dvAxialLog, driftLog, axDriftLog

    def globalStationkeep(
        self,
        TL,
        trajStartTime,
        tau=0 * u.d,
        dt=30 * u.min,
        simTime=1 * u.hr,
        SRP=False,
        Moon=False,
        axlBurn=True,
    ):
        """Method to simulate global stationkeeping with all target list stars

        This method simulates full observations in a loop for all stars in a
        target list. It logs the same metrics as the stationkeep method and saves
        it onto a file specified in the body of the method. This method returns
        nothing.

        Args:
            TL (TargetList module):
                TargetList class object
            trajStartTime (astropy Time array):
                Current absolute mission time in MJD
            tau (float Quantity):
                Time relative to trajStartTime at which to simulate observations
                in units of days
            dt (float Quantity):
                First guess of trajectory drift time in units of minutes
            simTime (float Quantity):
                Total simulated observation time in units of hours
            SRP (bool):
                Toggles whether or not to include solar radiation pressure force
            Moon (bool):
                Toggles whether or not to include lunar gravity force
            axlBurn (bool):
                Set True for purely axial burn (no dz velocity). Defaults true.

        Returns:
            None

        """

        # drift times
        tDriftMax_Log = np.zeros(TL.nStars) * u.min
        tDriftMean_Log = np.zeros(TL.nStars) * u.min
        tDriftStd_Log = np.zeros(TL.nStars) * u.min
        # full delta v's
        dvMean_Log = np.zeros(TL.nStars) * u.m / u.s
        dvMax_Log = np.zeros(TL.nStars) * u.m / u.s
        dvStd_Log = np.zeros(TL.nStars) * u.m / u.s
        # axial delta v's (counteract axial drifts)
        dvAxlMean_Log = np.zeros(TL.nStars) * u.m / u.s
        dvAxlMax_Log = np.zeros(TL.nStars) * u.m / u.s
        dvAxlStd_Log = np.zeros(TL.nStars) * u.m / u.s
        # number of burns
        bounce_Log = np.zeros(TL.nStars)
        axlDrift_Log = np.zeros(TL.nStars) * u.km

        # relative to some trajStartTime
        currentTime = trajStartTime + tau

        # just so we can catalogue it
        latDist = self.latDist
        latDistOuter = self.latDistOuter

        sInds = np.zeros(TL.nStars)

        tic = time.perf_counter()
        for sInd in range(TL.nStars):
            self.vprint(tau, " -- start time: ", trajStartTime)
            self.vprint(sInd, " / ", TL.nStars)

            # let's try to stationkeep!
            good = True
            try:
                (
                    nBounces,
                    timeLeft,
                    dvLog,
                    dvAxialLog,
                    driftLog,
                    axDriftLog,
                ) = self.stationkeep(
                    TL,
                    sInd,
                    currentTime,
                    dt=dt,
                    simTime=simTime,
                    SRP=SRP,
                    Moon=Moon,
                    axlBurn=axlBurn,
                )
            except:  # noqa: E722
                # stationkeeping didn't work! sad. just skip that index, then.
                good = False

            # stationkeeping worked!
            sInds[sInd] = good
            if good and len(driftLog) > 0 and len(dvLog) > 0 and len(dvAxialLog) > 0:
                self.vprint("stationkeeping worked!")
                bounce_Log[sInd] = nBounces
                axlDrift_Log[sInd] = axDriftLog
                tDriftMax_Log[sInd] = np.max(driftLog)
                tDriftMean_Log[sInd] = np.mean(driftLog)
                tDriftStd_Log[sInd] = np.std(driftLog)
                dvMax_Log[sInd] = np.max(dvLog)
                dvMean_Log[sInd] = np.mean(dvLog)
                dvStd_Log[sInd] = np.std(dvLog)
                dvAxlMax_Log[sInd] = np.max(dvAxialLog)
                dvAxlMean_Log[sInd] = np.mean(dvAxialLog)
                dvAxlStd_Log[sInd] = np.std(dvAxialLog)
            else:
                if good:
                    bounce_Log[sInd] = nBounces
                    axlDrift_Log[sInd] = axDriftLog
                    tDriftMax_Log[sInd] = simTime
                    tDriftMean_Log[sInd] = simTime
                    tDriftStd_Log[sInd] = 0
                    dvMax_Log[sInd] = 0
                    dvMean_Log[sInd] = 0
                    dvStd_Log[sInd] = 0
                    dvAxlMax_Log[sInd] = 0
                    dvAxlMean_Log[sInd] = 0
                    dvAxlStd_Log[sInd] = 0

            # NOMENCLATURE:
            # m  - model:   (IN - inertial) (RT - rotating)
            # ic - initial conditions: (CNV - centered, neutral velocity)
            #                           (WIP - well, ideal parabola)
            # lm - lunar model (C - circular) (NP - nodal precession)
            # ac - axial control law (CD - cancel drift) (NC - no control)
            # n  - number of stars
            # ld - lateral distance (in meters * 10 )
            # ms - reference epoch for mission start (in mjd)
            # t  - time since reference epoch (in days)

            burnStr = "CD" if axlBurn else "NC"

            filename = (
                "skMap_mIN_icWIP_lmCNP_ac"
                + burnStr
                + "_n"
                + str(int(TL.nStars))
                + "_ld"
                + str(int(latDist.value * 10))
                + "_ms"
                + str(int(trajStartTime.value))
                + "_t"
                + str(int((tau).value))
                + "_SRP"
                + str(int(SRP))
                + "_Moon"
                + str(int(Moon))
            )

            timePath = os.path.join(self.cachedir, filename + ".skmap")
            toc = time.perf_counter()

            A = {
                "simTime": simTime,
                "compTime": toc - tic,
                "bounces": bounce_Log,
                "axialDrift": axlDrift_Log,
                "SRP": SRP,
                "Moon": Moon,
                "dt": dt,
                "tDriftMax": tDriftMax_Log,
                "tDriftMean": tDriftMean_Log,
                "tDriftStd": tDriftStd_Log,
                "dvMax": dvMax_Log,
                "dvMean": dvMean_Log,
                "dvStd": dvStd_Log,
                "dvAxlMax": dvAxlMax_Log,
                "dvAxlMean": dvAxlMean_Log,
                "dvAxlStd": dvAxlStd_Log,
                "dist": TL.dist,
                "lon": TL.coords.lon,
                "lat": TL.coords.lat,
                "missionStart": trajStartTime,
                "tau": tau,
                "latDist": latDist,
                "latDistOuter": latDistOuter,
                "trajStartTime": currentTime,
                "axlBurn": axlBurn,
                "sInds": sInds,
            }

            with open(timePath, "wb") as f:
                pickle.dump(A, f)
