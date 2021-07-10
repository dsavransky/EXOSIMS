import unittest

from astropy import units as u
from astropy import constants as const
import EXOSIMS.util.phaseFunctions as pf
import EXOSIMS.PlanetPhysicalModel as ppm
import numpy as np
import math

"""phaseFunctions.py module unit tests

Sonny Rappaport, June 2021 (in format of code for test_deltaMag)

General strategy: I look up the paper referenced in phaseFunctions.py and 
doublecheck the paper's function vs what phaseFunction method returns. 
"""

class TestPhaseFunctions(unittest.TestCase):
    
    def test1(self):
        """Testing the quasiLambertPhaseFunction and its inverse for arbitrary small inputs
          Note: Inverse should only take on input values from 0 to 1.
          """

        i1 = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8])
        result = pf.quasiLambertPhaseFunction(i1)
        expected = np.array([1.0, 0.9950104048691683, 0.9801659131709817, 0.9558351964265126, 0.9226188356698382, 0.8813290691787037, 0.8329625267644232, 0.7786669865047742, 0.7197034143859217])
        np.testing.assert_allclose(expected, result, rtol=1e-8, atol=0)

        i2 = np.array([1.0, 0.9950104048691683, 0.9801659131709817, 0.9558351964265126, 0.9226188356698382, 0.8813290691787037, 0.8329625267644232, 0.7786669865047742, 0.7197034143859217])
        result2 = pf.quasiLambertPhaseFunctionInverse(i2)
        expected2 = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8]) 
        np.testing.assert_allclose(expected2, result2, rtol=1e-8, atol=0)

    def test2(self): 
        """ Testing the phi_lambert function for arbitrary small inputs.
        Numbers calculated on desmos. 
        """ 

        # expected = [1,.995,.981,.9582,.9277,.8910,.8474,.7995,.7476,.6929,.6362]

        # for x in np.arange(0,11,1): 
        #   self.assertAlmostEqual (pf.phi_lambert(x/10),expected[x],delta=1e-8)

    def test3(self): 
        """ Testing the transitionStart and transitionEnd functions, specifically
        testing for the midpoint value and two extreme values. 
        
        Also checks that the other phase functions begin at 0 degrees around 
        1 and ends at 180 around 180.

        The hyperbolic tangent function is given somewhat looser bounds as it's
        a fitted tanh function. 
        """ 

        delta1 = 1e-8
        delta2 = 1e-3
        #looser delta 

        self.assertAlmostEqual(pf.transitionStart(-1e100,0,1),0,delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(1e100,0,1),1,delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(0,0,1),.5,delta=delta1)
        self.assertAlmostEqual(pf.transitionStart(2,2,1),.5,delta=delta1)
        #transition start

        self.assertAlmostEqual(pf.transitionEnd(-1e100,0,1),1,delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(1e100,0,1),0,delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(0,0,1),.5,delta=delta1)
        self.assertAlmostEqual(pf.transitionEnd(2,2,1),.5,delta=delta1)
        #transition end

        self.assertAlmostEqual(pf.phi_lambert(0),1,delta=delta1)
        self.assertAlmostEqual(pf.phi_lambert(np.pi),0,delta=delta1)
        #phi_lampert

        self.assertAlmostEqual(pf.quasiLambertPhaseFunction(0),1,delta=delta1)
        self.assertAlmostEqual(pf.quasiLambertPhaseFunction(np.pi),0,delta=delta1)
        #quasi_lampert phase function


        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0.,0.,0.,0.,'mercury'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'mercury'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'venus'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'venus'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'earth'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'earth'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'mars'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'mars'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'jupiter'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'jupiter'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'neptune'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'neptune'),0,delta=delta2)

        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(0.*u.deg,0,0,0,0,'uranus'),1,delta=delta2)
        self.assertAlmostEqual(pf.hyperbolicTangentPhaseFunc(180*u.deg,0,0,0,0,'uranus'),0,delta=delta2)
        #hyperbolicTangentPhaseFunc for each planet 

    def test4(self):
        """ Testing the hyperbolicTangentPhaseFunc for each of the predefined
        inputs (the various planets.) Uses the functions from Mallama2018 using
        some of their sample input/outputs. Uses a loose bound for the test 
        as average values are used.
        
        For each of the nested functions:

        args: 
            r (float): Planet's distance from the sun.
            d (float): Planet's distance from the earth.
            a (float): Planet phase angle in degrees. Requires 0 < a < 180. 
        
        Returns: apparant magnitude, as described via mallama2018's various 
        functions for the different planets.
        """
          

if __name__ == '__main__':
    unittest.main()