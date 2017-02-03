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

########################################
#
# turmon 9/2016 -- the API for PostProcessing has changed,
# invalidating these tests.  perhaps the test strategy
# is still OK, or salvageable.
# set all tests to skipped.
#
########################################

class Test_PostProcessing_prototype(unittest.TestCase):
    
    #Testing some limiting cases below
    @unittest.skip("API changed.  Skip.")
    def test1(self):
        #case that false alarm probability is 0 and missed det prob is 0  
        obj = PostProcessing.PostProcessing(**specs)
        obj.FAP = 0.0 #Different from default values
        obj.MDP = 0.0

        observation_possible = [False]*10 #no possible detections
        #boolean result: False Alarm, Detection, Missed Det, Null Det
        result = obj.det_occur(observation_possible)        
        expected = False, False, False, True 
        self.assertEqual( result , expected)

        observation_possible1 = [True]*10 #all possible detections
        result1 = obj.det_occur(observation_possible1)
        expected1 = False, True, False, False        
        self.assertEqual( result1 , expected1)

        observation_possible2 = [False, True, False, False]#1 possible detection
        result2 = obj.det_occur(observation_possible1)
        expected2 = False, True, False, False        
        self.assertEqual( result2 , expected2)

    #another limiting case
    @unittest.skip("API changed.  Skip.")
    def test2(self):
        #case that false alarm probability is 1 and missed det prob is 0
        obj = PostProcessing.PostProcessing(**specs)
        obj.FAP = 1.0
        obj.MDP = 0.0

        observation_possible = [False]*10
        result = obj.det_occur(observation_possible)
        expected = True, False, False, False
        self.assertEqual( result , expected)

        observation_possible1 = [True]*10
        result1 = obj.det_occur(observation_possible1)
        expected1 = False, True, False, False
        self.assertEqual( result1 , expected1)

        observation_possible2 = [False, False, True, False]
        result2 = obj.det_occur(observation_possible2)        
        expected2 = False, True, False, False
        self.assertEqual( result2 , expected2)

    #limiting case
    @unittest.skip("API changed.  Skip.")
    def test3(self):
        #case of false alarm probability of 0 and missed det prob of 1
        obj = PostProcessing.PostProcessing(**specs)
        obj.FAP = 0.0
        obj.MDP = 1.0

        observation_possible = [False]*10
        result = obj.det_occur(observation_possible)
        expected = False, False, False, True
        self.assertEqual( result , expected)
        
        observation_possible1 = [True]*10
        result1 = obj.det_occur(observation_possible1)
        #expected1 = True, False, False, False #check this
        expected1 = False, False, True, False #check this
        self.assertEqual( result1 , expected1)
        
        observation_possible2 = [False, False, True, False]
        result2 = obj.det_occur(observation_possible2)        
        expected2 = False, False, True, False
        self.assertEqual( result2 , expected2)


    @unittest.skip("API changed.  Skip.")
    def test_non_trivial(self):
        obj = PostProcessing.PostProcessing(**specs)

        #Test a case for which the false alarm prob is 0.1
        obj.FAP = 0.1
        obj.MDP = 0.0
        observation_possible = [False]
        counter = 0
        for i in range(0,10000):
            result = obj.det_occur(observation_possible)
            if(result == (True, False, False, False)):
                counter = counter + 1
        #counter should be ~1000. Use ~3 sigma tolerance 
        #~0.3% of the time this test should fail! due to random number gen.
        np.testing.assert_allclose(1000, counter, rtol=0., atol=90)


        #Test a case for which the missed det prob is 0.2
        obj.FAP = 0
        obj.MDP = 0.2            
        observation_possible = [True]
        counter = 0
        for i in range(0,10000):
            result = obj.det_occur(observation_possible)
            if(result == (False, False, True, False)):
                counter = counter + 1
        np.testing.assert_allclose(2000, counter, rtol=0., atol=134)
        
        #Test default values
        #explicitly write default values or small values
        obj1 =  PostProcessing.PostProcessing()
        observation_possible = [True] #Should give a few missed det
        counter = 0
        for i in range(0,10000):
            result = obj1.det_occur(observation_possible)
            if(result == (False, False, True, False)):
                counter = counter + 1
        np.testing.assert_allclose(10, counter, rtol=0., atol=9)
        
        observation_possible = [False] #should give a few false alarms
        counter = 0
        for i in range(0,100000):
            result = obj1.det_occur(observation_possible)
            if(result == (True, False, False, False)):
                counter = counter + 1
        print counter
        np.testing.assert_allclose(3, counter, rtol=0., atol=5)


if __name__ == '__main__':
    unittest.main()

    
