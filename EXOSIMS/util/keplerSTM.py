import numpy as np
import sys

try:
    import EXOSIMS.util.KeplerSTM_C.CyKeplerSTM
except ImportError:
    pass


class planSys:
    """
    Kepler State Transition Matrix

    Class container for defining a planetary system (or group of planets in multiple
    systems) via their gravitational parameters and state vectors.  Contains methods
    for propagating state vectors forward in time via the Kepler state transition
    matrix.

    Args:
        x0 (~numpy.ndarray(float)):
            6n vector of stacked positions and velocities for n planets
        mu (~numpy.ndarray(float)):
            n vector of standard gravitational parameters mu = G(m+m_s) where m is
            the planet mass, m_s is the star mass and G is the gravitational
            constant
        epsmult (float):
            default multiplier on floating point precision, used as convergence
            metric.  Higher values mean faster convergence, but sacrifice precision.
        prefVallado (bool):
            If True, always try the Vallado algorithm first, otherwise try Shepherd
            first. Defaults False;
        noc (bool):
            Do not attempt to use cythonized code even if found.  Defaults False.


    .. note::

        All units must be complementary (i.e., if position is AU and velocity
        is AU/day, mu must be in AU^3/day^2.

    .. note::

        Two algorithms are implemented, both using Batting/Goodyear universal variables.
        The first is from Shepperd (1984), using continued fraction to solve the Kepler
        equation. The second is from Vallado (2004), using Newton iteration to solve the
        time equation. One algorithm is used preferentially, and the other is called
        only in the case of convergence failure on the first.  All convergence is
        calculated to machine precision of the data type and variable size, scaled by a
        user-selected multiple.

    """

    def __init__(self, x0, mu, epsmult=4.0, prefVallado=False, noc=False):
        # determine number of planets and validate input
        nplanets = x0.size / 6.0
        if nplanets - np.floor(nplanets) > 0:
            raise Exception("The length of x0 must be a multiple of 6.")

        if mu.size != nplanets:
            raise Exception("The length of mu must be the length of x0 divided by 6")

        self.nplanets = int(nplanets)
        self.mu = np.squeeze(mu)
        if self.mu.size == 1:
            self.mu = np.array(mu)

        self.epsmult = epsmult

        if prefVallado:
            self.algOrder = [self.calcSTM_vallado, self.calcSTM]
        else:
            self.algOrder = [self.calcSTM, self.calcSTM_vallado]

        # create position and velocity index matrices
        tmp = np.reshape(np.arange(self.nplanets * 6), (self.nplanets, 6)).T
        self.rinds = tmp[0:3]
        self.vinds = tmp[3:6]

        if not (noc) and ("EXOSIMS.util.KeplerSTM_C.CyKeplerSTM" in sys.modules):
            self.havec = True
        else:
            self.havec = False

        self.updateState(np.squeeze(x0))

    def updateState(self, x0):
        """Update internal state variable and associated constants

        Args:
            x0 (~numpy.ndarray(float)):
                6n vector of stacked positions and velocities for n planets

        """

        self.x0 = x0
        r0 = self.x0[self.rinds]
        v0 = self.x0[self.vinds]

        # constants
        self.r0norm = np.sqrt(np.sum(r0**2.0, 0))  # ||r0||
        self.v0norm2 = np.sum(v0 * v0, 0)  # ||v0||^2
        self.nu0 = np.sum(r0 * v0, 0)  # r0 \cdot v0
        self.beta = 2 * self.mu / self.r0norm - self.v0norm2  # -2E
        self.alpha = self.beta / self.mu
        self.nu0osmu = self.nu0 / np.sqrt(self.mu)

    def takeStep(self, dt):
        """Propagate state by input time

        Args:
            dt (float):
                Time step

        """
        if self.havec:
            try:
                tmp = EXOSIMS.util.KeplerSTM_C.CyKeplerSTM.CyKeplerSTM(
                    self.x0, dt, self.mu, self.epsmult
                )
                self.updateState(tmp)
                return
            except:  # noqa: E722
                print("Cython propagation failed.  Falling back to python.")

        try:
            Phi = self.algOrder[0](dt)
        except ValueError as detail:
            print("First algorithm error: %s\n Trying second algorithm." % (detail))
            Phi = self.algOrder[1](dt)

        self.updateState(np.dot(Phi, self.x0))

    def calcSTM(self, dt):
        """Compute STM for input time

        Args:
            dt (float):
                Time step

        Returns:
            ~numpy.ndarray(float):
                6x6 STM

        """

        # allocate
        u = np.zeros(self.nplanets)
        deltaU = np.zeros(self.beta.size)
        t = np.zeros(self.nplanets)
        counter = 0

        # For elliptic orbits, calculate period effects
        eorbs = self.beta > 0
        if any(eorbs):
            P = 2 * np.pi * self.mu[eorbs] * self.beta[eorbs] ** (-3.0 / 2.0)
            n = np.floor((dt + P / 2 - 2 * self.nu0[eorbs] / self.beta[eorbs]) / P)
            deltaU[eorbs] = 2 * np.pi * n * self.beta[eorbs] ** (-5.0 / 2.0)

        # loop until convergence of the time array to the time step
        while (np.max(np.abs(t - dt)) > self.epsmult * np.spacing(dt)) and (
            counter < 1000
        ):
            q = self.beta * u**2.0 / (1 + self.beta * u**2.0)
            U0w2 = 1.0 - 2.0 * q
            U1w2 = 2.0 * (1.0 - q) * u
            temp = self.contFrac(q)
            U = 16.0 / 15.0 * U1w2**5.0 * temp + deltaU
            U0 = 2.0 * U0w2**2.0 - 1.0
            U1 = 2.0 * U0w2 * U1w2
            U2 = 2.0 * U1w2**2.0
            U3 = self.beta * U + U1 * U2 / 3.0
            r = self.r0norm * U0 + self.nu0 * U1 + self.mu * U2
            t = self.r0norm * U1 + self.nu0 * U2 + self.mu * U3
            u = u - (t - dt) / (4.0 * (1.0 - q) * r)
            counter += 1

        if counter == 1000:
            raise ValueError(
                "Failed to converge on t: %e/%e"
                % (np.max(np.abs(t - dt)), self.epsmult * np.spacing(dt))
            )

        # Kepler solution
        f = 1 - self.mu / self.r0norm * U2
        g = self.r0norm * U1 + self.nu0 * U2
        F = -self.mu * U1 / r / self.r0norm
        G = 1 - self.mu / r * U2

        Phi = np.zeros([6 * self.nplanets] * 2)
        for j in np.arange(self.nplanets):
            st = j * 6
            Phi[st : st + 6, st : st + 6] = np.vstack(
                (
                    np.hstack((np.eye(3) * f[j], np.eye(3) * g[j])),
                    np.hstack((np.eye(3) * F[j], np.eye(3) * G[j])),
                )
            )

        return Phi

    def contFrac(self, x, a=5.0, b=0.0, c=5.0 / 2.0):
        """Compute continued fraction

        Args:
            x (~numpy.ndarray(float)):
                iterant
            a (float):
                a parameter
            b (float):
                b parameter
            c (float):
                c parameter

        Returns:
            ~numpy.ndarray(float):
                converged iterant

        """

        # initialize
        k = 1 - 2 * (a - b)
        l = 2 * (c - 1)
        d = 4 * c * (c - 1)
        n = 4 * b * (c - a)
        A = np.ones(x.size)
        B = np.ones(x.size)
        G = np.ones(x.size)

        Gprev = np.zeros(x.size) + 2
        counter = 0
        # loop until convergence of continued fraction
        while (np.max(np.abs(G - Gprev)) > self.epsmult * np.max(np.spacing(G))) and (
            counter < 1000
        ):
            k = -k
            l = l + 2.0
            d = d + 4.0 * l
            n = n + (1.0 + k) * l
            A = d / (d - n * A * x)
            B = (A - 1.0) * B
            Gprev = G
            G = G + B
            counter += 1

        if counter == 1000:
            raise ValueError(
                (
                    "Failed to converge on G, most likely due to divergence in "
                    "continued fractions."
                )
            )

        return G

    def calcSTM_vallado(self, dt):
        """Compute STM for input time

        Args:
            dt (float):
                Time step

        Returns:
            ~numpy.ndarray(float):
                6x6 STM

        """

        # classify orbits
        epsval = 1e-12

        eorbs = self.alpha >= epsval
        porbs = np.abs(self.alpha) < epsval
        horbs = self.alpha <= -epsval

        xi = np.zeros(self.nplanets)
        if np.any(eorbs):
            atmp = self.alpha[eorbs]
            tmp = np.sqrt(self.mu[eorbs]) * dt * atmp
            circinds = np.abs(atmp - 1) > epsval
            if any(circinds):
                tmp[circinds] *= 0.97

            xi[eorbs] = tmp

        if np.any(porbs):
            r0 = self.x0[self.rinds]
            v0 = self.x0[self.vinds]

            h = np.cross(r0[:, porbs].T, v0[:, porbs].T).T
            p = np.sum(h * h, 0) / self.mu[porbs]

            s = np.arctan2(1.0, (3.0 * np.sqrt(self.mu[porbs] / p**3.0) * dt)) / 2.0
            w = np.arctan((np.tan(s)) ** (1.0 / 3.0))
            xi[porbs] = np.sqrt(p) * 2.0 / np.tan(2 * w)
            self.alpha[porbs] = 0

        if np.any(horbs):
            a = 1.0 / (self.alpha[horbs])
            xi[horbs] = (
                np.sign(dt)
                * np.sqrt(-a)
                * np.log(
                    -2
                    * self.mu[horbs]
                    * self.alpha[horbs]
                    * dt
                    / (
                        self.nu0[horbs]
                        + np.sign(dt)
                        * np.sqrt(-self.mu[horbs] * self.alpha[horbs])
                        * (1.0 - self.r0norm[horbs] * self.alpha[horbs])
                    )
                )
            )

        # loop
        counter = 0
        r = self.r0norm
        xiup = 10.0 * np.max(np.abs(np.hstack((xi, r))))
        while (
            np.max(np.abs(xiup))
            > self.epsmult * np.spacing(np.max(np.abs(np.hstack((xi, r)))))
        ) and (counter < 1000):
            ps = xi**2.0 * self.alpha
            c2, c3 = self.psi2c2c3(ps)
            r = (
                xi**2.0 * c2
                + self.nu0osmu * xi * (1 - ps * c3)
                + self.r0norm * (1 - ps * c2)
            )
            xiup = (
                np.sqrt(self.mu) * dt
                - xi**3.0 * c3
                - self.nu0osmu * xi**2.0 * c2
                - self.r0norm * xi * (1 - ps * c3)
            ) / r
            xi += xiup
            counter += 1

        if counter == 1000:
            raise ValueError(
                "Failed to converge on xi: %e/%e"
                % (
                    np.max(np.abs(xiup)),
                    self.epsmult * np.spacing(np.max(np.abs(np.hstack((xi, r))))),
                )
            )

        # kepler solution
        f = 1.0 - xi**2.0 / self.r0norm * c2
        g = dt - xi**3.0 / np.sqrt(self.mu) * c3
        F = np.sqrt(self.mu) / r / self.r0norm * xi * (ps * c3 - 1.0)
        G = 1.0 - xi**2.0 / r * c2

        Phi = np.zeros([6 * self.nplanets] * 2)
        for j in np.arange(self.nplanets):
            st = j * 6
            Phi[st : st + 6, st : st + 6] = np.vstack(
                (
                    np.hstack((np.eye(3) * f[j], np.eye(3) * g[j])),
                    np.hstack((np.eye(3) * F[j], np.eye(3) * G[j])),
                )
            )

        return Phi

    def psi2c2c3(self, psi0):
        """Compute c_2 and c_3 values given psi

        Args:
            psi0 (float):
                Input psi value

        Returns:
            tuple:
                c2 (float):
                    c2 value
                c3 (float):
                    c3 value
        """

        c2 = np.zeros(len(psi0))
        c3 = np.zeros(len(psi0))

        psi12 = np.sqrt(np.abs(psi0))
        pos = psi0 >= 0
        neg = psi0 < 0
        if np.any(pos):
            c2[pos] = (1 - np.cos(psi12[pos])) / psi0[pos]
            c3[pos] = (psi12[pos] - np.sin(psi12[pos])) / psi12[pos] ** 3.0
        if any(neg):
            c2[neg] = (1 - np.cosh(psi12[neg])) / psi0[neg]
            c3[neg] = (np.sinh(psi12[neg]) - psi12[neg]) / psi12[neg] ** 3.0

        tmp = c2 + c3 == 0
        if any(tmp):
            c2[tmp] = 1.0 / 2.0
            c3[tmp] = 1.0 / 6.0

        return c2, c3
