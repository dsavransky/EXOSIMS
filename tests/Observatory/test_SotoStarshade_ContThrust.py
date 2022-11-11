from EXOSIMS.Observatory.SotoStarshade_ContThrust import SotoStarshade_ContThrust as sss
import math as m
import unittest
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
from scipy.optimize import basinhopping
import scipy.interpolate as interp
import scipy.integrate as intg
from scipy.integrate import solve_bvp
from copy import deepcopy
import time
import os
import pickle


class TestSotoStarshadeContThrust(unittest.TestCase):

    """
    Sonny Rappaport, July 2021, Cornell

    This class tests particular methods from SotoStarshade_Ski.

    """

    def test_DCM_r2i_i2r(self):

        """
        tests DCM_r2i against a manually created 6x6 rotation matrix, with
        the basis vectors being the rotational position and velocity vectors.
        Arbitary inputs are used for the angle t.

        Also tests DCM_i2r. The inverse of a matrix is its transpose, so the
        transpose is simply taken from DCM_r2i_manual.

        matrix taken from Gabe's thesis ("ORBITAL DESIGN TOOLS AND SCHEDULING
        TECHNIQUES FOR OPTIMIZING SPACE SCIENCE AND EXOPLANET-FINDING MISSIONS"),
        page 81.
        """

        def DCM_r2i_manual(t):
            x = [m.cos(t), -m.sin(t), 0, 0, 0, 0]
            y = [m.sin(t), m.cos(t), 0, 0, 0, 0]
            z = [0, 0, 1, 0, 0, 0]
            dx = [-m.sin(t), -m.cos(t), 0, m.cos(t), -m.sin(t), 0]
            dy = [m.cos(t), -m.sin(t), 0, m.sin(t), m.cos(t), 0]
            dz = [0, 0, 0, 0, 0, 1]

            return np.vstack([x, y, z, dx, dy, dz])

        def DCM_i2r_manual(t):
            return np.linalg.inv(DCM_r2i_manual(t))

        for t in np.arange(0, 101):
            np.testing.assert_allclose(sss.DCM_r2i(self, t), DCM_r2i_manual(t))
            np.testing.assert_allclose(sss.DCM_i2r(self, t), DCM_i2r_manual(t))

    def test_DCM_r2i_9(self):
        """
        tests DCM_r2i_9 against a manually created 6x6 rotation matrix, with
        the basis vectors being the rotational position, velocity, and
        accerelation vector.

        Arbitary inputs are used for the angle t.

        matrix taken from Gabe's thesis ("ORBITAL DESIGN TOOLS AND SCHEDULING
        TECHNIQUES FOR OPTIMIZING SPACE SCIENCE AND EXOPLANET-FINDING MISSIONS"),
        page 81.
        """

        def DCM_r2i_9_manual(t):
            x = np.array([m.cos(t), -m.sin(t), 0, 0, 0, 0, 0, 0, 0])
            y = np.array([m.sin(t), m.cos(t), 0, 0, 0, 0, 0, 0, 0])
            z = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
            dx = np.array([-m.sin(t), -m.cos(t), 0, m.cos(t), -m.sin(t), 0, 0, 0, 0])
            dy = np.array([m.cos(t), -m.sin(t), 0, m.sin(t), m.cos(t), 0, 0, 0, 0])
            dz = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
            ddx = np.array([0, 0, 0, 0, 0, 0, m.cos(t), -m.sin(t), 0]) - 2 * dy + x
            ddy = np.array([0, 0, 0, 0, 0, 0, m.sin(t), m.cos(t), 0]) + 2 * dx + y
            ddz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

            return np.vstack([x, y, z, dx, dy, dz, ddx, ddy, ddz])

        for t in np.arange(0, 101):
            np.testing.assert_allclose(sss.DCM_r2i_9(self, t), DCM_r2i_9_manual(t))
