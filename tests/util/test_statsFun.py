import unittest
from EXOSIMS.util import statsFun
import numpy as np
import scipy.stats

class TestStatsFun(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simpSample(self):
        """ Test simple rejection sampler

        Test method: use KS-statistic for two continuous distributions
        and ensure that generated values correctly correlate with each one
        """

        #uniform dist
        ulim = [0,1]
        ufun = lambda x: 1.0/np.diff(ulim)

        n = int(1e5)
        usample = statsFun.simpSample(ufun,n,ulim[0],ulim[1])
        self.assertGreaterEqual(usample.min(), ulim[0])
        self.assertLessEqual(usample.max(), ulim[1])

        nlim = [-10,10]
        nfun = lambda x: np.exp(-x**2./2.0)/np.sqrt(2.0*np.pi)
        nsample = statsFun.simpSample(nfun,n,nlim[0],nlim[1])
        self.assertGreaterEqual(nsample.min(), nlim[0])
        self.assertLessEqual(nsample.min(), nlim[1])

        self.assertGreaterEqual(scipy.stats.kstest(usample,'uniform')[1],0.01)
        self.assertGreaterEqual(scipy.stats.kstest(nsample,'norm')[1],0.01)
        self.assertLessEqual(scipy.stats.kstest(nsample,'uniform')[1],0.01)
        self.assertLessEqual(scipy.stats.kstest(usample,'norm')[1],0.01)

    def test_simpSample_trivial(self):
        """ Test simple rejection sampler with trivial inputs

        Test method: set up sampling with equal upper and lower bounds
        """

        ulim = [0,1]
        ufun = lambda x: 1.0/np.diff(ulim)

        n = 10000
        sample = statsFun.simpSample(ufun,n,0.5,0.5)

        self.assertEqual(len(sample),n)
        self.assertTrue(np.all(sample == 0.5))

