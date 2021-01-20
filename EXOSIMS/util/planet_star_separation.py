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
    #I couldn't find where this next equation is
    #s = r/np.sqrt(2.)*np.sqrt(1+np.cos(i)**2. + (1.-np.cos(i)**2)*np.cos(2.*(w+v)))
    s = r*np.sqrt(np.sin(v+w)**2.*np.cos(i)**2 + np.cos(v+w)**2.) #this is from threeDellipseToTwoDellipse
    return s