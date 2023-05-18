# -*- coding: utf-8 -*-
import numpy as np
import scipy
import astropy.units as u


def deltaMag(p, Rp, d, Phi):
    """Calculates delta magnitudes for a set of planets, based on their albedo,
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
        ~numpy.ndarray:
            Planet delta magnitudes

    """
    dMag = -2.5 * np.log10(p * (Rp / d).decompose() ** 2 * Phi).value

    return dMag


def betaStar_Lambert():
    """Compute the Lambert phase function deltaMag-maximizing phase angle

    Args:
        None

    Returns:
        float:
            Value of beta^* in radians
    """

    betastarexpr = (
        lambda beta: -(np.pi - beta) * np.sin(beta) ** 3 / np.pi
        + 2
        * ((np.pi - beta) * np.cos(beta) + np.sin(beta))
        * np.sin(beta)
        * np.cos(beta)
        / np.pi
    )
    betastar = scipy.optimize.fsolve(betastarexpr, 63 * np.pi / 180)[0]

    return betastar


def min_deltaMag_Lambert(Completeness, s=None):
    """Calculate the minimum deltaMag at given separation(s) assuming a Lambert phase
    function

    Args:
        Completeness (BrownCompleteness):
            BrownCompleteness object
        s (float or ~numpy.ndarray, optional):
            Projected separations (in AU)  to compute minimum delta mag at.
            If None (default) then uses Completeness.xnew

    Returns:
        ~numpy.ndarray:
            Minimum deltaMag values

    """
    # this is the output of betaStar_Lambert (no need to recompute every time)
    betastar = 1.1047288186445432

    # if no s input supplied, use the full array of separations from Completeness
    if s is None:
        s = Completeness.xnew

    # allocate output
    dmagmin = np.zeros(s.size)

    # idenitfy breakpoints
    Ppop = Completeness.PlanetPopulation
    bp1 = Ppop.rrange.min().to(u.AU).value * np.sin(betastar)
    bp2 = Ppop.rrange.max().to(u.AU).value * np.sin(betastar)

    # compute minimum delta mags
    dmagmin[s < bp1] = -2.5 * np.log10(
        Ppop.prange.max()
        * ((Ppop.Rprange.max() / Ppop.rrange.min()).decompose().value) ** 2
        * Completeness.PlanetPhysicalModel.calc_Phi(
            (np.arcsin(s[s < bp1] / Ppop.rrange.min().value)) * u.rad
        )
    )
    inds = (s >= bp1) & (s < bp2)
    dmagmin[inds] = -2.5 * np.log10(
        Ppop.prange.max()
        * ((Ppop.Rprange.max().to(u.AU).value / s[inds])) ** 2
        * Completeness.PlanetPhysicalModel.calc_Phi(betastar * u.rad)
        * np.sin(betastar) ** 2
    )
    dmagmin[s >= bp2] = -2.5 * np.log10(
        Ppop.prange.max()
        * ((Ppop.Rprange.max() / Ppop.rrange.max()).decompose().value) ** 2
        * Completeness.PlanetPhysicalModel.calc_Phi(
            (np.arcsin(s[s >= bp2] / Ppop.rrange.max().value)) * u.rad
        )
    )

    return dmagmin


def max_deltaMag_Lambert(Completeness, s=None):
    """Calculate the maximum deltaMag at given separation(s) assuming a Lambert phase
    function

    Args:
        Completeness (BrownCompleteness):
            BrownCompleteness object
        s (float or ~numpy.ndarray, optional):
            Projected separations (in AU)  to compute minimum delta mag at.
            If None (default) then uses Completeness.xnew

    Returns:
        ~numpy.ndarray:
            Maximum deltaMag values

    """
    Ppop = Completeness.PlanetPopulation

    dmagmax = -2.5 * np.log10(
        Ppop.prange.min()
        * ((Ppop.Rprange.min() / Ppop.rrange.max()).decompose().value) ** 2
        * Completeness.PlanetPhysicalModel.calc_Phi(
            (np.pi - np.arcsin(s / Ppop.rrange.max().value)) * u.rad
        )
    )

    return dmagmax
