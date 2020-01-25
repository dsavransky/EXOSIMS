from EXOSIMS.Observatory.SotoStarshade_ContThrust import SotoStarshade_ContThrust
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
from scipy.optimize import basinhopping
import scipy.interpolate as interp
import scipy.integrate as intg
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os
try:
    import _pickle as pickle
except:
    import pickle

EPS = np.finfo(float).eps


class SotoStarshade_SK(SotoStarshade_ContThrust):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,orbit_datapath=None,**specs): 

        SotoStarshade_ContThrust.__init__(self,**specs)  

    # converting angular velocity
    def convertAngVel_to_canonical(self,angvel):
        """ Convert velocity to canonical units
        """
        angvel = angvel.to('rad/yr')
        return angvel.value / (2*np.pi)

    def convertAngVel_to_dim(self,angvel):
        """ Convert velocity to canonical units
        """
        angvel = angvel * (2*np.pi)
        return angvel * u.rad / u.yr