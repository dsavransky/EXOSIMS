# -*- coding: utf-8 -*-
import numpy as np

def calc_Blam(lam, temp):
    """ Calculates the spectral radiance of a body at absolute temperature T
    using Planck's law Blambda(lambda,T) in CGS units

    Args:
        lam:
            wavelength in nm
        temp:
            temperature in Kelvin

    Returns:
        Blam:
            spectral radiance in erg cm^-3 s^-1 steradian^-1 (CGS units)

    """

    lb = lam*1e-7                   # wavelength in cm
    h = 6.6260755e-27               # Planck constant in erg s
    c = 2.99792458e10               # speed of light in vacuum in cm s-1
    kB = 1.38064852e-16             # Boltzmann constant in erg K-1
    ephoton = h*c/lb
    Blam = 2*h*c**2/lb**5 / (np.exp(ephoton/kB/temp) - 1)

    return Blam
