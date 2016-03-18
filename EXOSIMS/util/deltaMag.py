# -*- coding: utf-8 -*-
import numpy as np

def deltaMag(p,R,r,Phi):
    """ Calculates delta magnitudes for a set of planets, based on their albedo, 
    radius, and position with respect to host star.

    Args:
        p:
            (1D numpy ndarray) planet albedo
        R:
            (1D numpy ndarray) planet radius, default astropy units of km
        r:
            (1D numpy ndarray) planet-star distance, default astropy units of AU
        Phi:
            (1D numpy ndarray) Lambert phase

    Returns:
        dMag:
            (1D numpy ndarray) planet delta magnitudes

    """

    dMag = -2.5*np.log10((p*(R/r)**2*Phi).decompose().value)
    
    return dMag
