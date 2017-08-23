import unittest
import numpy as np
import os
import EXOSIMS
from EXOSIMS.Prototypes import PostProcessing
import numpy as np

r"""PostProcessing module unit tests

Paul Nunez, JPL, Aug. 2016
"""

# need a dummy BackgroundSources
specs = {'modules':{'BackgroundSources':''}}


class Test_PostProcessing_prototype(unittest.TestCase):
    
    #Testing some limiting cases below
    def test_zeroFAP(self):
        #case that false alarm probability is 0 and missed det prob is 0  
        obj = PostProcessing.PostProcessing(FAP=0.0,MDP=0.0,**specs)

        SNRin = np.array([1.0,2.0,3.0,4.0,5.0,5.1,6.0])
        SNRmin = 5.0
        expected_MDresult = [True,True,True,True,False,False,False] 
        FA, MD = obj.det_occur(SNRin,SNRmin)     
        for x,y in zip(MD,expected_MDresult):
            self.assertEqual( x,y )
        self.assertEqual( FA, False)

    #another limiting case
    def test_oneFAP(self):
        #case that false alarm probability is 1 and missed det prob is 0
        obj = PostProcessing.PostProcessing(FAP=1.0,MDP=0.0,**specs)

        SNRin = np.array([1.0,2.0,3.0,4.0,5.0,5.1,6.0])
        SNRmin = 5.0
        expected_MDresult = [True,True,True,True,False,False,False] 
        FA, MD = obj.det_occur(SNRin,SNRmin)        
        for x,y in zip(MD,expected_MDresult):
            self.assertEqual( x,y )
        self.assertEqual( FA, True)


    def test_nontrivialFAP(self):
        obj = PostProcessing.PostProcessing(FAP=0.1,MDP=0.0,**specs)

        #Test a case for which the false alarm prob is 0.1
        FAs = np.zeros(1000,dtype=bool)
        for j in range(len(FAs)):
            FA,_ = obj.det_occur(np.array([5]),5)
            FAs[j] = FA
        #~0.3% of the time this test should fail! due to random number gen.
        np.testing.assert_allclose(len(FAs)*obj.FAP, len(np.where(FAs)[0]), rtol=0.3, atol=0.)



if __name__ == '__main__':
    unittest.main()

    
