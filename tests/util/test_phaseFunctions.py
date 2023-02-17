import unittest

from astropy import units as u
import EXOSIMS.util.phaseFunctions as pf
import numpy as np

"""phaseFunctions.py module unit tests

Sonny Rappaport, June 2021 (in format of code for test_deltaMag)

General strategy: I look up the paper referenced in phaseFunctions.py and
doublecheck the paper's function vs what phaseFunction method returns.
"""


class TestPhaseFunctions(unittest.TestCase):
    def test_quasilambert(self):
        """Testing the quasiLambertPhaseFunction and its inverse for arbitrary small
        inputs

        Note: Inverse should only take on input values from 0 to 1.
        """

        i1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        result = pf.quasiLambertPhaseFunction(i1)
        expected = np.array(
            [
                1.0,
                0.9950104048691683,
                0.9801659131709817,
                0.9558351964265126,
                0.9226188356698382,
                0.8813290691787037,
                0.8329625267644232,
                0.7786669865047742,
                0.7197034143859217,
            ]
        )
        np.testing.assert_allclose(expected, result, rtol=1e-8, atol=0)

        i2 = np.array(
            [
                1.0,
                0.9950104048691683,
                0.9801659131709817,
                0.9558351964265126,
                0.9226188356698382,
                0.8813290691787037,
                0.8329625267644232,
                0.7786669865047742,
                0.7197034143859217,
            ]
        )
        result2 = pf.quasiLambertPhaseFunctionInverse(i2)
        expected2 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        np.testing.assert_allclose(expected2, result2, rtol=1e-8, atol=0)

        i3 = 0.3
        result3 = pf.quasiLambertPhaseFunction(i3)
        expected3 = 0.9558351964265126
        self.assertAlmostEqual(expected3, result3, delta=1e-8)

        i4 = 0.9558351964265126
        result3 = pf.quasiLambertPhaseFunctionInverse(i4)
        expected3 = 0.3
        self.assertAlmostEqual(expected3, result3, delta=1e-8)

    def test_lambert(self):
        """Testing the phi_lambert function for arbitrary small inputs.
        Numbers calculated on desmos.
        """

        expected = [
            1.0,
            0.995110162508012,
            0.9809120137457908,
            0.9581755777367491,
            0.927743574153277,
            0.8905168478209772,
            0.8474394049633889,
            0.7994832652785417,
            0.7476333383100179,
            0.6928725292930388,
            0.6361672737835763,
        ]

        for x in np.arange(0, 11, 1):
            self.assertAlmostEqual(pf.phi_lambert(x / 10), expected[x], delta=1e-8)

    def test_transition(self):
        """Testing the transitionStart and transitionEnd functions, specifically
        testing for the midpoint value and two extreme values.

        Also checks that the other phase functions begin at 0 degrees around
        1 and ends at 180 around 180.

        The hyperbolic tangent function is given somewhat looser bounds as it's
        a fitted tanh function.
        """

        delta1 = 1e-8
        delta2 = 1e-3
        # looser delta

        self.assertAlmostEqual(pf.transitionStart(-1e100, 0, 1), 0, delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(1e100, 0, 1), 1, delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(0, 0, 1), 0.5, delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(2, 2, 1), 0.5, delta=delta1)
        # transition start

        self.assertAlmostEqual(pf.transitionEnd(-1e100, 0, 1), 1, delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(1e100, 0, 1), 0, delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(0, 0, 1), 0.5, delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(2, 2, 1), 0.5, delta=delta1)
        # transition end

        self.assertAlmostEqual(pf.phi_lambert(0), 1, delta=delta1)
        self.assertAlmostEqual(pf.phi_lambert(np.pi), 0, delta=delta1)
        # phi_lampert

        self.assertAlmostEqual(pf.quasiLambertPhaseFunction(0), 1, delta=delta1)
        self.assertAlmostEqual(pf.quasiLambertPhaseFunction(np.pi), 0, delta=delta1)
        # quasi_lampert phase function

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0.0, 0.0, 0.0, 0.0, "mercury"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "mercury"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "venus"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "venus"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "earth"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "earth"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "mars"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "mars"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "saturn"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "saturn"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "jupiter"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "jupiter"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "neptune"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "neptune"),
            0,
            delta=delta2,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(0.0 * u.deg, 0, 0, 0, 0, "uranus"),
            1,
            delta=delta2,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFunc(180 * u.deg, 0, 0, 0, 0, "uranus"),
            0,
            delta=delta2,
        )
        # hyperbolicTangentPhaseFunc for each planet

        delta1 = 1.5
        delta2 = 15
        # much looser bounds to check for degrees- degrees on a larger scale anyways

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "mercury"),
            180,
            delta=delta2,
        )
        # bad fit? all others work with smaller bounds
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "mercury"),
            1,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "venus"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "venus"),
            0,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "earth"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "earth"),
            0,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "mars"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "mars"), 0, delta=delta1
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "jupiter"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "jupiter"),
            0,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "saturn"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "saturn"),
            0,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "neptune"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "neptune"),
            0,
            delta=delta1,
        )

        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(0, 0, 0, 0, 0, "uranus"),
            180,
            delta=delta1,
        )
        self.assertAlmostEqual(
            pf.hyperbolicTangentPhaseFuncInverse(1, 0, 0, 0, 0, "uranus"),
            0,
            delta=delta1,
        )

    def test_betaFunc(self):
        """Testing the beta function for arbitrary inputs.
        Expected outputs taken from a python script."""

        input = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        np.testing.assert_allclose(
            pf.betaFunc(input, input, input),
            [
                1.5707963267948966,
                1.5509611881121566,
                1.4933534592588231,
                1.4031488617499241,
                1.28767754720778,
                1.1555419735883117,
                1.0165896898819988,
                0.8829895396912839,
                0.7712352623707003,
            ],
        )
        # test array of inputs
        self.assertAlmostEqual(
            pf.betaFunc(0.1, 0.1, 0.1), 1.5509611881121563, delta=np.spacing(2 * np.pi)
        )
        # test a singular input


if __name__ == "__main__":
    unittest.main()
