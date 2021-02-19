"""
Planet Star Separation
Written By: Dean Keithly
Written On: 11/13/2020
"""
import numpy as np

def planet_star_separation(a,e,v,w,i):
    """ Following directly from Keithly 2021. Calculates planet star separation given KOE
    Args:
        a (numpy array):
            planet semi-major axis in AU
        e (numpy array):
            planet eccentricity
        v (numpy array):
            planet true anomaly rad
        w (numpy array):
            planet argument of periapsis rad
        i (numpy array):
            planet inclination rad
    Returns:
        s (numpy array):
            planet-star separations in AU
    """
    r = a*(1.-e**2.)/(1.+e*np.cos(v))
    s = r*np.sqrt(np.sin(v+w)**2.*np.cos(i)**2 + np.cos(v+w)**2.) #this is from threeDellipseToTwoDellipse.ipynb
    return s
