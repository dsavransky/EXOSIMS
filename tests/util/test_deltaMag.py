import unittest

from astropy import units as u
from astropy import constants as const
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np

r"""DeltaMag module unit tests

Paul Nunez, JPL, Aug. 2016
"""


class Test_deltaMag(unittest.TestCase):
    def test1(self):
        r"""Testing some limiting cases."""
        p = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        Rp = np.array([1.0, 0.0, 1.0, 1.0, 1.0]) * u.kilometer
        d = np.array([1.0, 1.0, 0.0, 1.0, np.inf]) * u.kilometer
        Phi = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        # suppress division-by-zero warnings
        with np.errstate(divide="ignore"):
            result = deltaMag(p, Rp, d, Phi)
        expected = np.array([np.inf, np.inf, -np.inf, np.inf, np.inf])
        np.testing.assert_allclose(expected, result, rtol=1e-1, atol=0)

    def test2(self):
        r"""Testing a couple of specific cases."""
        p = np.array([0.1, 0.2])
        Rp = np.array([1.0, 2.0]) * const.R_earth
        d = np.array([1.0, 2.0]) * u.au
        Phi = np.array([0.1, 0.5])
        result = deltaMag(p, Rp, d, Phi)
        expected = np.array([26.85, 24.35])
        np.testing.assert_allclose(expected, result, rtol=1e-2, atol=0.0)


if __name__ == "__main__":
    unittest.main()
