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
        Algorithm from Shepperd, 1984, using Goodyear's universal variables
        and continued fraction to solve the Kepler equation.

    """

    def __init__(self, x0, mu, epsmult=4.0, noc=False):
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

        if not (noc) and ("EXOSIMS.util.KeplerSTM_C.CyKeplerSTM" in sys.modules):
            self.havec = True
            self.x0 = np.squeeze(x0)
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

        # create position and velocity matrices
        # tmp = np.reshape(self.x0,(self.nplanets,6)).T
        # r0 = tmp[0:3]
        # v0 = tmp[3:6]

        tmp = np.reshape(np.arange(len(self.x0)), (self.nplanets, 6)).T
        self.rinds = tmp[0:3]
        self.vinds = tmp[3:6]
        r0 = self.x0[self.rinds]
        v0 = self.x0[self.vinds]

        # constants and allocation
        self.r0norm = np.sqrt(sum(r0**2.0, 0))
        self.nu0 = sum(r0 * v0, 0)
        self.beta = 2 * self.mu / self.r0norm - sum(v0**2, 0)

    def takeStep(self, dt):
        """Propagate state by input time

        Args:
            dt (float):
                Time step

        """

        if self.havec:
            self.x0 = EXOSIMS.util.KeplerSTM_C.CyKeplerSTM.CyKeplerSTM(
                self.x0, dt, self.mu, self.epsmult
            )
        else:
            tmp = np.zeros(self.x0.shape)
            for j in range(self.nplanets):
                Phi = self.calcSTM(dt, j)
                tmp[j * 6 : (j + 1) * 6] = np.dot(Phi, self.x0[j * 6 : (j + 1) * 6])

            self.updateState(tmp)

    def calcSTM(self, dt, j):
        """Compute STM for input time for one body

        Args:
            dt (float):
                Time step
            j (int):
                Index of body to propagate

        Returns:
            ~numpy.ndarray(float):
                6x6 STM

        """

        # allocate
        u = 0
        deltaU = 0
        t = 0
        counter = 0

        # For elliptic orbits, calculate period effects
        if self.beta[j] > 0:
            P = 2 * np.pi * self.mu[j] * self.beta[j] ** (-3.0 / 2.0)
            n = np.floor((dt + P / 2 - 2 * self.nu0[j] / self.beta[j]) / P)
            deltaU = 2 * np.pi * n * self.beta[j] ** (-5.0 / 2.0)

        # loop until convergence of the time array to the time step
        while (np.max(np.abs(t - dt)) > self.epsmult * np.spacing(dt)) and (
            counter < 1000
        ):
            q = self.beta[j] * u**2.0 / (1 + self.beta[j] * u**2.0)
            U0w2 = 1.0 - 2.0 * q
            U1w2 = 2.0 * (1.0 - q) * u
            temp = self.contFrac(q)
            U = 16.0 / 15.0 * U1w2**5.0 * temp + deltaU
            U0 = 2.0 * U0w2**2.0 - 1.0
            U1 = 2.0 * U0w2 * U1w2
            U2 = 2.0 * U1w2**2.0
            U3 = self.beta[j] * U + U1 * U2 / 3.0
            r = self.r0norm[j] * U0 + self.nu0[j] * U1 + self.mu[j] * U2
            t = self.r0norm[j] * U1 + self.nu0[j] * U2 + self.mu[j] * U3
            u = u - (t - dt) / (4.0 * (1.0 - q) * r)
            counter += 1

        if counter == 1000:
            raise ValueError(
                "Failed to converge on t: %e/%e"
                % (np.max(np.abs(t - dt)), self.epsmult * np.spacing(dt))
            )

        # Kepler solution
        f = 1 - self.mu[j] / self.r0norm[j] * U2
        g = self.r0norm[j] * U1 + self.nu0[j] * U2
        F = -self.mu[j] * U1 / r / self.r0norm[j]
        G = 1 - self.mu[j] / r * U2

        Phi = np.vstack(
            (
                np.hstack((np.eye(3) * f, np.eye(3) * g)),
                np.hstack((np.eye(3) * F, np.eye(3) * G)),
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
