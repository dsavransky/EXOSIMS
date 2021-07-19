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

        self.assertGreaterEqual(scipy.stats.kstest(usample,'uniform')[1],0.01,'Uniform sample does not look uniform.')
        self.assertGreaterEqual(scipy.stats.kstest(nsample,'norm')[1],0.01,'Normal sample does not look normal.')
        self.assertLessEqual(scipy.stats.kstest(nsample,'uniform')[1],0.01,'Normal sample looks too uniform.')
        self.assertLessEqual(scipy.stats.kstest(usample,'norm')[1],0.01,'Uniform sample looks too normal.')

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

    def test_norm_array(self):
        """Test array normalizer with trivial inputs

        Test methods: Ensure that 1D arrays passed in will sum to one after method called
        """

        self.assertEqual(statsFun.norm_array(np.array([3,3,26,43,62,4])).sum(),1)
        self.assertEqual(statsFun.norm_array(np.array([5,7,3,8,2,9,6])).sum(),1)

    def test_eqLogSample(self):
        """test eqLogSample with trivial inputs 
        
        Test method: ensure that eqLogSample returns expected array"""

        #uniform distribution from 0 to 1 
        ufun = lambda x: 1.0

        crit = scipy.stats.chi2.ppf(1-.01,99)

        sample = statsFun.eqLogSample(ufun, numTest= 10000 , xMin=1,xMax= 2)
        hist_sample = np.histogram(sample,100,density=True)

        print("\n" + "\n" + "\n")
        print(hist_sample)
        print("\n" + "\n" + "\n")

        loguni = scipy.stats.loguniform.rvs(1,2,size=10000)
        hist_loguni = np.histogram(loguni,hist_sample[1])

        print("\n" + "\n" + "\n")
        print(hist_loguni)
        print("\n" + "\n" + "\n")

        #critical value chi^2: chi^2 must be smaller than this value for .01 signifiance
        chi2 = scipy.stats.chisquare(hist_sample[0],hist_loguni[0])
        self.assertLess(chi2[0], crit)
        #assert that chi^2 is less than critical value 

if __name__ == '__main__':
    unittest.main()

