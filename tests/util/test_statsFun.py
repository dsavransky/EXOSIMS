import unittest
from EXOSIMS.util import statsFun
import numpy as np
import scipy.stats
from unittest.mock import patch, call


class TestStatsFun(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simpSample(self):
        """Test simple rejection sampler

        Test method: use KS-statistic for two continuous distributions
        and ensure that generated values correctly correlate with each one
        """

        # uniform dist
        ulim = [0, 1]
        ufun = lambda x: 1.0 / np.diff(ulim)

        n = int(1e5)
        usample = statsFun.simpSample(ufun, n, ulim[0], ulim[1])
        self.assertGreaterEqual(usample.min(), ulim[0])
        self.assertLessEqual(usample.max(), ulim[1])

        nlim = [-10, 10]
        nfun = lambda x: np.exp(-(x**2.0) / 2.0) / np.sqrt(2.0 * np.pi)
        nsample = statsFun.simpSample(nfun, n, nlim[0], nlim[1])
        self.assertGreaterEqual(nsample.min(), nlim[0])
        self.assertLessEqual(nsample.min(), nlim[1])

        self.assertGreaterEqual(
            scipy.stats.kstest(usample, "uniform")[1],
            0.01,
            "Uniform sample does not look uniform.",
        )
        self.assertGreaterEqual(
            scipy.stats.kstest(nsample, "norm")[1],
            0.01,
            "Normal sample does not look normal.",
        )
        self.assertLessEqual(
            scipy.stats.kstest(nsample, "uniform")[1],
            0.01,
            "Normal sample looks too uniform.",
        )
        self.assertLessEqual(
            scipy.stats.kstest(usample, "norm")[1],
            0.01,
            "Uniform sample looks too normal.",
        )

    def test_simpSample_error(self):
        """Test simple rejection sampler with trivial inputs, looking to raise
        max iteration exception

        Test method: set up sampling with equal upper and lower bounds, with a worst
        case scenario (approximate) spike function

        Sonny Rappaport, Cornell, 2021
        """

        ulim = [0, 1]
        ufun = lambda x: 1.0 / np.exp(-1e8 * x**2)

        n = 10000

        with self.assertRaises(Exception):
            sample = statsFun.simpSample(ufun, n, -1, 1)

    def test_simpSample_trivial(self):
        """Test simple rejection sampler with trivial inputs

        Test method: set up sampling with equal upper and lower bounds
        """

        ulim = [0, 1]
        ufun = lambda x: 1.0 / np.diff(ulim)

        n = 10000
        sample = statsFun.simpSample(ufun, n, 0.5, 0.5)

        self.assertEqual(len(sample), n)
        self.assertTrue(np.all(sample == 0.5))

    @patch("builtins.print")
    def test_simpSample_verb(self, mocked_print):
        """Test simple rejection sampler with trivial inputs, checking verb = True

        Test method: set up mock python printing and test that mock console output
        contains contains iteration information. Uses a simple uniform distribution
        so it just finishes in one iteration

        Sonny Rappaport, Cornell, 2021
        """

        ulim = [0, 1]
        ufun = lambda x: 1.0

        n = 10000
        sample = statsFun.simpSample(ufun, n, 0, 1, verb=True)

        self.assertEqual(mocked_print.mock_calls, [call("Finished in 1 iterations.")])

    def test_eqLogSample(self):
        """test eqLogSample with trivial inputs

        Test method: ensure that eqLogSample returns expected array. For this test,
        see if a uniform distribution transforms correctly.

        Sonny Rappaport, Cornell, 2021
        """

        # uniform distribution from 0 to 1
        ufun = lambda x: 1.0

        sample = statsFun.eqLogSample(ufun, numTest=10000, xMin=1, xMax=2)
        # generate uniform distribution points after a log transformation

        loguni = scipy.stats.loguniform.rvs(1, 2, size=10000)
        # generate a log uniform distribution from scipy

        ks_result = scipy.stats.kstest(sample, loguni)

        self.assertGreater(ks_result[1], 0.01)
        # assert that the p value is greater than .01


if __name__ == "__main__":
    unittest.main()
