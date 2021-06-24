import unittest

from astropy import units as u
from astropy import constants as const
import EXOSIMS.util.phaseFunctions as pf
import numpy as np

"""phaseFunctions.py module unit tests

Sonny Rappaport, June 2021 (in format of code for test_deltaMag)

General strategy: I look up the paper referenced in phaseFunctions.py and 
doublecheck the paper's function vs what phaseFunction method returns. 
"""

class Test_phaseFunctions(unittest.TestCase):
    
    def test1(self):
        """Testing the quasiLambertPhaseFunction and its inverse for arbitrary small inputs
          Note: Inverse should only take on input values from 0 to 1.
          """

        i1 = np.array([0,.1,.2,.3,.4])
        result = pf.quasiLambertPhaseFunction(i1)
        expected = np.array([1.,0.9950,0.9802,0.9558,0.9226])
        np.testing.assert_allclose(expected, result, rtol=1e-1, atol=0)

        i2 = np.array([1.,0.9950,0.9802,0.9558,0.9226])
        result2 = pf.quasiLambertPhaseFunctionInverse(i2)
        expected2 = np.array([0,.1,.2,.3,.4]) 
        np.testing.assert_allclose(expected2, result2, rtol=1e-1, atol=0)

    def test2(self): 
        """ Testing the hyperbolicTangentPhaseFunc.""" 

    def test3(self): 
        """ Testing the phi_lambert function for arbitrary small inputs.
        Numbers calculated on desmos. 
        """ 

        expected = [1,.995,.981,.9582,.9277,.8910,.8474,.7995,.7476,.6929,.6362]

        for x in np.arange(0,11,1): 
          self.assertAlmostEqual (pf.phi_lambert(x/10),expected[x],delta=1e-2)

    def test4(self): 
        """ Testing the transitionStart and transitionEnd functions, specifically
        testing for the midpoint value and two extreme values. """ 

        self.assertAlmostEqual(pf.transitionStart(-1e100,0,1),0,delta=.001)
        self.assertAlmostEqual(pf.transitionStart(1e100,0,1),1,delta=.001)
        self.assertAlmostEqual(pf.transitionStart(0,0,1),.5,delta=.001)
        self.assertAlmostEqual(pf.transitionStart(2,2,1),.5,delta=.001)

        self.assertAlmostEqual(pf.transitionEnd(-1e100,0,1),1,delta=.001)
        self.assertAlmostEqual(pf.transitionEnd(1e100,0,1),0,delta=.001)
        self.assertAlmostEqual(pf.transitionEnd(0,0,1),.5,delta=.001)
        self.assertAlmostEqual(pf.transitionEnd(2,2,1),.5,delta=.001)
          

if __name__ == '__main__':
    unittest.main()