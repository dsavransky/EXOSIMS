import unittest
from EXOSIMS.util.RejectionSampler import RejectionSampler as RS
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler as ITS
import numpy as np
import scipy.stats
import os

class TestSamplers(unittest.TestCase):
    """Test rejection sampler and inverse transform sampler since both have
    same set up
    """

    def setUp(self):
        self.dev_null = open(os.devnull, 'w')
        self.mods = [RS,ITS]

    def tearDown(self):
        pass

    def test_simpSample(self):
        """Test samplers using KS-statistic for two continuous distributions
        and ensure that generated values correctly correlate with each one
        """
        
        #uniform dist
        ulim = [0,1]
        ufun = lambda x: 1.0/np.diff(ulim)

        n = int(1e5)
        
        #normal/Gaussian dist
        nlim = [-10,10]
        nfun = lambda x: np.exp(-x**2./2.0)/np.sqrt(2.0*np.pi)
        
        for mod in self.mods:
            print('Testing uniform and normal distributions for sampler: %s'%mod.__name__)
            #test uniform distribution
            usampler = mod(ufun,ulim[0],ulim[1])
            usample = usampler(n)
            self.assertGreaterEqual(usample.min(), ulim[0],'Uniform sampler does not obey lower limit for %s.'%mod.__name__)
            self.assertLessEqual(usample.max(), ulim[1],'Uniform sampler does not obey upper limit for %s.'%mod.__name__)
            
            #test normal/Gaussian distribution
            nsampler = mod(nfun,nlim[0],nlim[1])
            nsample = nsampler(n)
            self.assertGreaterEqual(nsample.min(), nlim[0],'Normal sampler does not obey lower limit for %s.'%mod.__name__)
            self.assertLessEqual(nsample.min(), nlim[1],'Normal sampler does not obey upper limit for %s.'%mod.__name__)
            
            # test that uniform sample is not normal and normal is not uniform
            # this test is probabilistic and may fail
            nu = scipy.stats.kstest(nsample,'uniform')[1]
            if nu > 0.01:
                # test fails, so try resampling to get it to pass
                nsample = nsampler(n)
                nu = scipy.stats.kstest(nsample,'uniform')[1]
            self.assertLessEqual(nu,0.01,'Normal sample looks too uniform for %s.'%mod.__name__)
            
            # this test is also probabilistic and may fail
            un = scipy.stats.kstest(usample,'norm')[1]
            if un > 0.01:
                # test fails, so try resampling to get it to pass
                usample = usampler(n)
                un = scipy.stats.kstest(usample,'norm')[1]
            self.assertLessEqual(un,0.01,'Uniform sample looks too normal for %s.'%mod.__name__)
            
            # this test is probabilistic and may fail
            pu = scipy.stats.kstest(usample,'uniform')[1]
            if pu < 0.01:
                # test fails, so try resampling to get it to pass
                usample = usampler(n)
                pu = scipy.stats.kstest(usample,'uniform')[1]
            self.assertGreaterEqual(pu,0.01,'Uniform sample does not look uniform for %s.'%mod.__name__)
            
            # this test is also probabilistic and may fail
            pn = scipy.stats.kstest(nsample,'norm')[1]
            if pn < 0.01:
                # test fails, try resampling to get it to pass
                nsample = nsampler(n)
                pn = scipy.stats.kstest(nsample,'norm')[1]
            self.assertGreaterEqual(pn,0.01,'Normal sample does not look normal for %s.'%mod.__name__)

    def test_simpSample_trivial(self):
        """ Test simple rejection sampler with trivial inputs

        Test method: set up sampling with equal upper and lower bounds
        """
        
        ulim = [0,1]
        ufun = lambda x: 1.0/np.diff(ulim)

        n = 10000
        
        for mod in self.mods:
            print('Testing trivial input for sampler: %s'%mod.__name__)
            sampler = mod(ufun,0.5,0.5)
            sample = sampler(n)
            
            self.assertEqual(len(sample),n,'Sampler %s does not return all same value'%mod.__name__)
            self.assertTrue(np.all(sample == 0.5),'Sampler %s does not return all values at 0.5'%mod.__name__)