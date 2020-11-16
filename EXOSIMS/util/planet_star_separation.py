# Planet Star Separation
# Written By: Dean Keithly
# Written On: 11/13/2020
import numpy as np

def planet_star_separation(a,e,v,w,i):
    """ Following directly from Keithly 2021. Calculates planet star separation given KOE
    Args:
        a (numpy array):
        e (numpy array):
        v (numpy array):
        w (numpy array):
        i (numpy array):
    Returns:
        s (numpy array):
            planet-star separations in AU
    """
    r = a*(1.-e**2.)/(1.+e*np.cos(v))
    s = r/np.sqrt(2.)*np.sqrt(1+np.cos(i)**2. + (1.-np.cos(i)**2)*np.cos(2.*(w+v)))
    return s