"""
Planet Star Separation
Written By: Dean Keithly
Written On: 11/13/2020
"""

import numpy as np


def planet_star_separation(a, e, v, w, i):
    """Following directly from Keithly 2021. Calculates planet star separation given KOE

    Args:
        a (numpy.ndarray(float)):
            planet semi-major axis in AU
        e (numpy.ndarray(float)):
            planet eccentricity
        v (numpy.ndarray(float)):
            planet true anomaly rad
        w (numpy.ndarray(float)):
            planet argument of periapsis rad
        i (numpy.ndarray(float)):
            planet inclination rad

    Returns:
        numpy.ndarray(float):
            planet-star separations in AU
    """

    r = a * (1.0 - e**2.0) / (1.0 + e * np.cos(v))
    s = r * np.sqrt(np.sin(v + w) ** 2.0 * np.cos(i) ** 2 + np.cos(v + w) ** 2.0)

    return s
