r"""Test code for PostProcessing module within EXOSIMS.

As of 10/2016, these tests are skipped, because the API changed.

Cate Liu, IPAC, 2016"""

import unittest
import numpy as np
import EXOSIMS
from EXOSIMS.Prototypes.PostProcessing import PostProcessing

class TestPostProcessing_others(unittest.TestCase):
    def setUp(self):
        self.spec = {"modules":{"BackgroundSources" : "BackgroundSources"}}
        pass
    
    def tearDown(self):
        pass
    
    @unittest.skip('API changed')
    def test_det_occur1(self):
        obj = PostProcessing(MDP=-0.1, **self.spec)
        
        observables = [True, False, False, True, False]
        res = obj.det_occur(observables)
        expected = [False, True, False, False]
        np.testing.assert_allclose(res, expected)
               
    @unittest.skip('API changed')
    def test_det_occur2(self):
        obj = PostProcessing(MDP=1.1, **self.spec)
        
        observables = [True, False, False, True, False]
        res = obj.det_occur(observables)
        expected = [False, False, True, False]
        np.testing.assert_allclose(res, expected)
        
    @unittest.skip('API changed')
    def test_det_occur3(self):
        obj = PostProcessing(FAP=-0.2, **self.spec)
        
        observables = [False, False, False, False, False]
        res = obj.det_occur(observables)
        expected = [False, False, False, True]
        np.testing.assert_allclose(res, expected)
        
    @unittest.skip('API changed')
    def test_det_occur4(self):
        obj = PostProcessing(FAP=1.1, **self.spec)
        
        observables = [False, False, False, False, False]
        res = obj.det_occur(observables)
        expected = [True, False, False, False]
        np.testing.assert_allclose(res, expected)
        
if __name__ == "__main__":
    unittest.main()

