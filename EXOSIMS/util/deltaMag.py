# -*- coding: utf-8 -*-
import numpy as np

def deltaMag(p,Rp,d,Phi):
    """ Calculates delta magnitudes for a set of planets, based on their albedo, 
    radius, and position with respect to host star.
    
    Args:
        p (ndarray):
            Planet albedo
        Rp (astropy Quantity array):
            Planet radius in units of km
        d (astropy Quantity array):
            Planet-star distance in units of AU
        Phi (ndarray):
            Planet phase function
    
    Returns:
        dMag (ndarray):
            Planet delta magnitudes
    
    """
    dMag = -2.5*np.log10(p*(Rp/d).decompose()**2*Phi).value
    
    return dMag
