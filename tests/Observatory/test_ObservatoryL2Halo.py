import unittest
from tests.TestSupport.Info import resource_path
import EXOSIMS.Observatory.ObservatoryL2Halo as obs
import pkgutil, os, json, sys, copy
from EXOSIMS.util.get_module import get_module
import numpy as np
from astropy.time import Time
import astropy.units as u
from tests.TestSupport.Utilities import RedirectStreams

# Python 3 compatibility:
if sys.version_info[0] > 2:
    from io import StringIO
else:
    from StringIO import StringIO

class TestObservatoryL2Halo(unittest.TestCase):
    """

    ObservatoryL2Halo test.
    Made specifically (for now) to test the jacobian_CRTBP calculation. 

    """

    def setUp(self):

        self.fixture = obs.ObservatoryL2Halo()

    def tearDown(self):
      del self.fixture

    def test_jacobian_CRTBP(self): 
        """
        Strategy: Make sure the output is of the right type/size. Arg 't' doesn't
        seem to do anything yet in ObservatoryL2Halo. 
        """

    #     Z = np.zeros([3,3,1])
    #     E = np.full_like(Z,np.eye(3).reshape(3,3,1))
    #     w = np.array([[ 0. , 2. , 0.],
    #                   [-2. , 0. , 0.],
    #                   [ 0. , 0. , 0.]])
    #     W = np.full_like(Z,w.reshape(3,3,1))
    #     t=0
    #     obs = self.fixture
    #     #should be the same for each output 

    #     i1 = np.array([[1],[1],
    #     [1],[1],
    #     [1],[1]])
    #     result1 = obs.jacobian_CRTBP(t,i1)

    #     expected3x3 = -np.array([[[9.99999705e-01], [1.92449115e-01], [1.92449115e-01]],
    #    [[1.92449115e-01], [1.00000015e+00], [1.92450142e-01]],
    #    [[1.92449115e-01], [1.92450142e-01], [1.47392514e-07]]])
    #     row1 = np.hstack( [ Z , E ])
    #     row2 = np.hstack( [ expected3x3 , W ])
    #     expected = np.vstack( [ row1, row2 ])

    #     np.testing.assert_array_almost_equal_nulp(expected,result1)
    

        # i2 = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],
        #         [1,2,3,4,5,6],[1,2,3,4,5,6],
        #         [1,2,3,4,5,6],[1,2,3,4,5,6]])
        # result2 = obs.jacobian_CRTBP(t,i2)
        # self.assertEqual( result1.shape, (6,6,6) ) 
