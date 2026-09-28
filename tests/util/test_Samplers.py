import unittest
from EXOSIMS.util.RejectionSampler import RejectionSampler as RS
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler as ITS
import numpy as np
import scipy.stats
import os
from unittest.mock import patch, call


class TestSamplers(unittest.TestCase):
    """Test rejection sampler and inverse transform sampler since both have
    same set up
    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        self.mods = [RS, ITS]

    def tearDown(self):
        self.dev_null.close()

    def test_samplers(self):
        """Test samplers using KS-statistic for two continuous distributions
        and ensure that generated values correctly correlate with each one
        """

        # uniform dist
        ulim = [0, 1]
        ufun = lambda x: 1.0 / np.diff(ulim)

        n = int(1e5)

        # normal/Gaussian dist
        nlim = [-10, 10]
        nfun = lambda x: np.exp(-(x**2.0) / 2.0) / np.sqrt(2.0 * np.pi)

        for mod in self.mods:
            print(
                "Testing uniform and normal distributions for sampler: %s"
                % mod.__name__
            )
            # test uniform distribution
            usampler = mod(ufun, ulim[0], ulim[1])
            usample = usampler(n)
            self.assertGreaterEqual(
                usample.min(),
                ulim[0],
                "Uniform sampler does not obey lower limit for %s." % mod.__name__,
            )
            self.assertLessEqual(
                usample.max(),
                ulim[1],
                "Uniform sampler does not obey upper limit for %s." % mod.__name__,
            )

            # test normal/Gaussian distribution
            nsampler = mod(nfun, nlim[0], nlim[1])
            nsample = nsampler(n)
            self.assertGreaterEqual(
                nsample.min(),
                nlim[0],
                "Normal sampler does not obey lower limit for %s." % mod.__name__,
            )
            self.assertLessEqual(
                nsample.min(),
                nlim[1],
                "Normal sampler does not obey upper limit for %s." % mod.__name__,
            )

            # test that uniform sample is not normal and normal is not uniform
            # this test is probabilistic and may fail
            nu = scipy.stats.kstest(nsample, "uniform")[1]
            if nu > 0.01:
                # test fails, so try resampling to get it to pass
                nsample = nsampler(n)
                nu = scipy.stats.kstest(nsample, "uniform")[1]
            self.assertLessEqual(
                nu, 0.01, "Normal sample looks too uniform for %s." % mod.__name__
            )

            # this test is also probabilistic and may fail
            un = scipy.stats.kstest(usample, "norm")[1]
            if un > 0.01:
                # test fails, so try resampling to get it to pass
                usample = usampler(n)
                un = scipy.stats.kstest(usample, "norm")[1]
            self.assertLessEqual(
                un, 0.01, "Uniform sample looks too normal for %s." % mod.__name__
            )

            # this test is probabilistic and may fail
            pu = scipy.stats.kstest(usample, "uniform")[1]
            if pu < 0.01:
                # test fails, so try resampling to get it to pass
                usample = usampler(n)
                pu = scipy.stats.kstest(usample, "uniform")[1]
            self.assertGreaterEqual(
                pu, 0.01, "Uniform sample does not look uniform for %s." % mod.__name__
            )

            # this test is also probabilistic and may fail
            pn = scipy.stats.kstest(nsample, "norm")[1]
            if pn < 0.01:
                # test fails, try resampling to get it to pass
                nsample = nsampler(n)
                pn = scipy.stats.kstest(nsample, "norm")[1]
            self.assertGreaterEqual(
                pn, 0.01, "Normal sample does not look normal for %s." % mod.__name__
            )

    def test_samplers_trivial(self):
        """Test simple rejection sampler with trivial inputs

        Test method: set up sampling with equal upper and lower bounds
        """

        ulim = [0, 1]
        ufun = lambda x: 1.0 / np.diff(ulim)
        ufun2 = lambda x: np.ndarray.tolist(ufun)  # to trigger conversion to ndarray

        n = 10000

        for mod in self.mods:
            print("Testing trivial input for sampler: %s" % mod.__name__)
            sampler = mod(ufun, 0.5, 0.5)
            sample = sampler(n)
            sampler2 = mod(ufun2, 0.5, 0.5)
            sample2 = sampler2(n)

            self.assertEqual(
                len(sample),
                n,
                "Sampler %s does not return all same value" % mod.__name__,
            )
            self.assertTrue(
                np.all(sample == 0.5),
                "Sampler %s does not return all values at 0.5" % mod.__name__,
            )
            self.assertEqual(
                len(sample2),
                n,
                "Sampler %s does not return all same value" % mod.__name__,
            )
            self.assertTrue(
                np.all(sample2 == 0.5),
                "Sampler %s does not return all values at 0.5" % mod.__name__,
            )

    def test_RejectionSampler_error(self):
        """Test rejection sampler max iteration exception

        Test method: set up sampling with a worst case scenario (approximate) spike
        function, which should fail to converge and raise an exception.

        Sonny Rappaport, Cornell, 2021
        """

        ufun = lambda x: 1.0 / np.exp(-1e8 * x**2)

        n = 10000

        with np.errstate(over="ignore"), self.assertRaises(Exception):
            RS(ufun, -1, 1)(n)

    @patch("builtins.print")
    def test_RejectionSampler_verb(self, mocked_print):
        """Test rejection sampler with verb = True

        Test method: set up mock python printing and test that mock console output
        contains contains iteration information. Uses a simple uniform distribution
        so it just finishes in one iteration

        Sonny Rappaport, Cornell, 2021
        """

        ufun = lambda x: 1.0

        n = 10000
        RS(ufun, 0, 1)(n, verb=True)

        self.assertEqual(mocked_print.mock_calls, [call("Finished in 1 iterations.")])

    def test_RejectionSampler_seeded(self):
        """Test rejection sampler reproducibility with a fixed seed

        Test method: compare seeded samples to a reference rejection sampling loop
        drawing from np.random.uniform, which must be bitwise identical.
        """

        nfun = lambda x: np.exp(-(x**2.0) / 2.0)
        xMin, xMax = -3.0, 3.0
        n = 10000

        sampler = RS(nfun, xMin, xMax)
        np.random.seed(42)
        sample = sampler(n)

        np.random.seed(42)
        nSamp = max(2 * n, 1000 * 1000)
        xd = np.random.uniform(low=xMin, high=xMax, size=nSamp)
        yd = np.random.uniform(low=0, high=sampler.M, size=nSamp)
        expected = xd[yd < nfun(xd)][:n]

        self.assertTrue(np.array_equal(sample, expected))


if __name__ == "__main__":
    unittest.main()
