"""
Phase Functions

See also [Keithly2021]_
"""

import numpy as np
from astropy import units as u


def phi_lambert(beta, phiIndex=np.asarray([])):
    """Lambert phase function (isotropic scattering)

    See: [Sobolev1975]_

    Args:
        beta (astropy.units.quantity.Quantity or numpy.ndarray):
            phase angle array in radians
        phiIndex (numpy.ndarray):
            array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        numpy.ndarray:
            Phi, phase function values between 0 and 1
    """
    if hasattr(beta, "value"):
        beta = beta.to("rad").value

    phi = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    return phi


def transitionStart(x, a, b):
    """Smoothly transition from one 0 to 1

    Args:
        x (numpy.ndarray):
            x, in deg input value in deg, floats
        a (numpy.ndarray):
            transition midpoint in deg, floats
        b (numpy.ndarray):
            transition slope
    Returns:
        numpy.ndarray:
            s, Transition value from 0 to 1, floats
    """
    s = 0.5 + 0.5 * np.tanh((x - a) / b)
    return s


def transitionEnd(x, a, b):
    """Smoothly transition from one 1 to 0

    Smaller b is sharper step a is midpoint, s(a)=0.5

    Args:
        x (numpy.ndarray):
            x, in deg input value in deg, floats
        a (numpy.ndarray):
            a, transition midpoint in deg, floats
        b (numpy.ndarray):
            transition slope

    Returns:
        numpy.ndarray:
            s, transition value from 1 to 0
    """
    s = 0.5 - 0.5 * np.tanh((x - a) / b)
    return s


def quasiLambertPhaseFunction(beta, phiIndex=np.asarray([])):
    """Quasi Lambert Phase Function from [Agol2007]_

    Args:
        beta (astropy.units.quantity.Quantity or numpy.ndarray):
            planet phase angles in radians
        phiIndex (numpy.ndarray):
            array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        numpy.ndarray:
            Phi, phase function value
    """
    if hasattr(beta, "value"):
        beta = beta.to("rad").value

    Phi = np.cos(beta / 2.0) ** 4
    return Phi


def quasiLambertPhaseFunctionInverse(Phi, phiIndex=np.asarray([])):
    """Quasi Lambert Phase Function Inverse

    Args:
        Phi (numpy.ndarray):
            Phi, phase function value, floats
        phiIndex (numpy.ndarray):
            array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        numpy.ndarray:
            beta, planet phase angles in rad, floats
    """

    beta = 2.0 * np.arccos((Phi) ** (1.0 / 4.0))
    return beta


def hyperbolicTangentPhaseFunc(beta, A, B, C, D, planetName=None):
    """
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison using
    mallama2018PlanetProperties.py:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018
    comparison using mallama2018PlanetProperties.py:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213

    Args:
        beta (astropy.units.quantity.Quantity or numpy.ndarray):
            Phase Angle in radians
        A (float):
            Hyperbolic phase function parameter
        B (float):
            Hyperbolic phase function paramter
        C (float):
            Hyperbolic phase function parameter
        D (float):
            Hyperbolic phase function parameter
        planetName (string or None):
            planet name string all lower case for one of 8 solar system planets

    Returns:
        numpy.ndarray:
            Phi, phase angle in degrees
    """
    if planetName is None:
        return None  # do nothing
    elif planetName == "mercury":
        A, B, C, D = (
            0.9441564,
            0.3852919,
            2.59291159,
            -0.67540991,
        )  # 0.93940195,  0.40446512,  2.47034733, -0.64361749
    elif planetName == "venus":
        A, B, C, D = (
            1.20324931,
            1.57402581,
            0.60683886,
            0.87010846,
        )  # 1.26116739, 1.53204409, 0.61961161, 0.84075693
    elif planetName == "earth":
        A, B, C, D = (
            0.78414986,
            1.86890464,
            0.52958938,
            1.07587225,
        )  # 0.78415   , 1.86890455, 0.5295894 , 1.07587213
    elif planetName == "mars":
        A, B, C, D = (
            1.89881785,
            0.48220465,
            2.02299497,
            -1.02612681,
        )  # 2.02856459,  0.29590061,  3.32324214, -1.71048535
    elif planetName == "jupiter":
        A, B, C, D = (
            1.3622441,
            1.45676529,
            0.64490468,
            0.7800663,
        )  # 1.3761512 , 1.45852349, 0.64157352, 0.7983722
    elif planetName == "saturn":
        A, B, C, D = (
            5.02672862,
            0.41588155,
            1.80205383,
            -1.74369974,
        )  # 5.49410541,  0.37274869,  2.00119662, -2.1551928
    elif planetName == "uranus":
        A, B, C, D = (
            1.54388146,
            1.18304642,
            0.79972526,
            0.37288376,
        )  # 1.56866334, 1.16284633, 0.81250327, 0.34759469
    elif planetName == "neptune":
        A, B, C, D = (
            1.31369238,
            1.41437107,
            0.67584636,
            0.65077278,
        )  # 1.37105297, 1.36886173, 0.69506274, 0.609515

    if hasattr(beta, "value"):
        beta = beta.to("rad").value

    Phi = -np.tanh((beta - D) / A) / B + C
    return Phi


def hyperbolicTangentPhaseFuncInverse(Phi, A, B, C, D, planetName=None):
    """
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison
    using mallama2018PlanetProperties.py:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018
    comparison using mallama2018PlanetProperties.py:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213

    Args:
        Phi (numpy.ndarray):
            phase angle in degrees
        A (float):
            Hyperbolic phase function parameter
        B (float):
            Hyperbolic phase function paramter
        C (float):
            Hyperbolic phase function parameter
        D (float):
            Hyperbolic phase function parameter
        planetName (string or None):
            planet name string all lower case for one of 8 solar system planets

    Returns:
        numpy.ndarray:
            beta, Phase Angle  in degrees
    """
    if planetName is None:
        return None  # do nothing
    elif planetName == "mercury":
        A, B, C, D = (
            0.9441564,
            0.3852919,
            2.59291159,
            -0.67540991,
        )  # 0.93940195,  0.40446512,  2.47034733, -0.64361749
    elif planetName == "venus":
        A, B, C, D = (
            1.20324931,
            1.57402581,
            0.60683886,
            0.87010846,
        )  # 1.26116739, 1.53204409, 0.61961161, 0.84075693
    elif planetName == "earth":
        A, B, C, D = (
            0.78414986,
            1.86890464,
            0.52958938,
            1.07587225,
        )  # 0.78415   , 1.86890455, 0.5295894 , 1.07587213
    elif planetName == "mars":
        A, B, C, D = (
            1.89881785,
            0.48220465,
            2.02299497,
            -1.02612681,
        )  # 2.02856459,  0.29590061,  3.32324214, -1.71048535
    elif planetName == "jupiter":
        A, B, C, D = (
            1.3622441,
            1.45676529,
            0.64490468,
            0.7800663,
        )  # 1.3761512 , 1.45852349, 0.64157352, 0.7983722
    elif planetName == "saturn":
        A, B, C, D = (
            5.02672862,
            0.41588155,
            1.80205383,
            -1.74369974,
        )  # 5.49410541,  0.37274869,  2.00119662, -2.1551928
    elif planetName == "uranus":
        A, B, C, D = (
            1.54388146,
            1.18304642,
            0.79972526,
            0.37288376,
        )  # 1.56866334, 1.16284633, 0.81250327, 0.34759469
    elif planetName == "neptune":
        A, B, C, D = (
            1.31369238,
            1.41437107,
            0.67584636,
            0.65077278,
        )  # 1.37105297, 1.36886173, 0.69506274, 0.609515

    beta = ((A * np.arctanh(-B * (Phi - C)) + D) * u.radian).to("deg").value
    return beta


def betaFunc(inc, v, w):
    """Calculate the planet phase angle

    Args:
        inc (float or numpy.ndarray):
            planet inclination in rad
        v (numpy.ndarray):
            planet true anomaly in rad
        w (numpy.ndarray):
            planet argument of periapsis

    Returns:
        numpy.ndarray:
            beta, planet phase angle
    """
    beta = np.arccos(np.sin(inc) * np.sin(v + w))
    return beta


def phase_Mercury(beta):
    """Mercury phase function
    Valid from 0 to 180 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (
        -0.4
        * (
            6.3280e-02 * beta
            - 1.6336e-03 * beta**2.0
            + 3.3644e-05 * beta**3.0
            - 3.4265e-07 * beta**4.0
            + 1.6893e-09 * beta**5.0
            - 3.0334e-12 * beta**6.0
        )
    )
    return phase


def phase_Venus_1(beta):
    """Venus phase function
    Valid from 0 to 163.7 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (
        -0.4
        * (
            -1.044e-03 * beta
            + 3.687e-04 * beta**2.0
            - 2.814e-06 * beta**3.0
            + 8.938e-09 * beta**4.0
        )
    )
    return phase


def phase_Venus_2(beta):
    """Venus phase function
    Valid from 163.7 to 179 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (-0.4 * (-2.81914e-00 * beta + 8.39034e-03 * beta**2.0))
    # 1 Scale Properly
    h1 = phase_Venus_1(163.7) - 0.0  # Total height desired over range
    h2 = 10.0 ** (
        -0.4 * (-2.81914e-00 * 163.7 + 8.39034e-03 * 163.7**2.0)
    ) - 10.0 ** (-0.4 * (-2.81914e-00 * 179.0 + 8.39034e-03 * 179.0**2.0))
    phase = phase * h1 / h2  # Scale so height is proper
    # 2 Lateral movement to make two functions line up
    difference = phase_Venus_1(163.7) - h1 / h2 * (
        10.0 ** (-0.4 * (-2.81914e-00 * 163.7 + 8.39034e-03 * 163.7**2.0))
    )
    phase = phase + difference
    return phase


def phase_Venus_melded(beta):
    """
    Venus phae function
    Valid from 0 to 180 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = (
        transitionEnd(beta, 163.7, 5.0) * phase_Venus_1(beta)
        + transitionStart(beta, 163.7, 5.0)
        * transitionEnd(beta, 179.0, 0.5)
        * phase_Venus_2(beta)
        + transitionStart(beta, 179.0, 0.5) * phi_lambert(beta * np.pi / 180.0)
        + 2.766e-04
    )
    # 2.666e-04 ensures the phase function is entirely positive
    # (near 180 deg phase, there is a small region
    # where phase goes negative) This small addition fixes this
    return phase


def phase_Earth(beta):
    """Earth phase function
    Valid from 0 to 180 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (-0.4 * (-1.060e-3 * beta + 2.054e-4 * beta**2.0))
    return phase


def phase_Mars_1(beta):
    """Mars phase function
    Valid from 0 to 50 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (
        -0.4 * (0.02267 * beta - 0.0001302 * beta**2.0 + 0.0 + 0.0)
    )  # L(λe) + L(LS)
    return phase


def phase_Mars_2(beta):
    """Mars phase function
    Valid from 50 to 180 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = (
        phase_Mars_1(50.0)
        / 10.0 ** (-0.4 * (-0.02573 * 50.0 + 0.0003445 * 50.0**2.0))
        * 10.0 ** (-0.4 * (-0.02573 * beta + 0.0003445 * beta**2.0 + 0.0 + 0.0))
    )  # L(λe) + L(Ls)
    return phase


def phase_Mars_melded(beta):
    """Mars phase function
    Valid from 0 to 180 degrees

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = transitionEnd(beta, 50.0, 5.0) * phase_Mars_1(beta) + transitionStart(
        beta, 50.0, 5.0
    ) * phase_Mars_2(beta)
    return phase


def phase_Jupiter_1(beta):
    """Jupiter phase function
    Valid from 0 to 12 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (-0.4 * (-3.7e-04 * beta + 6.16e-04 * beta**2.0))
    return phase


def phase_Jupiter_2(beta):
    """Jupiter phase function
    Valid from 12 to 130 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    # inds = np.where(beta > 180.0)[0]
    # beta[inds] = [180.0] * len(inds)
    # assert np.all(
    #     (
    #         1.0
    #         - 1.507 * (beta / 180.0)
    #         - 0.363 * (beta / 180.0) ** 2.0
    #         - 0.062 * (beta / 180.0) ** 3.0
    #         + 2.809 * (beta / 180.0) ** 4.0
    #         - 1.876 * (beta / 180.0) ** 5.0
    #     )
    #     >= 0.0
    # ), "error in beta input"

    difference = phase_Jupiter_1(12.0) - 10.0 ** (
        -0.4
        * (
            -2.5
            * np.log10(
                1.0
                - 1.507 * (12.0 / 180.0)
                - 0.363 * (12.0 / 180.0) ** 2.0
                - 0.062 * (12.0 / 180.0) ** 3.0
                + 2.809 * (12.0 / 180.0) ** 4.0
                - 1.876 * (12.0 / 180.0) ** 5.0
            )
        )
    )
    phase = difference + 10.0 ** (
        -0.4
        * (
            -2.5
            * np.log10(
                1.0
                - 1.507 * (beta / 180.0)
                - 0.363 * (beta / 180.0) ** 2.0
                - 0.062 * (beta / 180.0) ** 3.0
                + 2.809 * (beta / 180.0) ** 4.0
                - 1.876 * (beta / 180.0) ** 5.0
            )
        )
    )
    return phase


def phase_Jupiter_melded(beta):
    """Jupiter phase function
    Valid from 0 to 130 degrees

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = (
        transitionEnd(beta, 12.0, 5.0) * phase_Jupiter_1(beta)
        + transitionStart(beta, 12.0, 5.0)
        * transitionEnd(beta, 130.0, 5.0)
        * phase_Jupiter_2(beta)
        + transitionStart(beta, 130.0, 5.0) * phi_lambert(beta * np.pi / 180.0)
    )
    return phase


def phase_Saturn_2(beta):
    """Saturn phase function (Globe Only Earth Observations)
    Valid beta from 0 to 6.5 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (-0.4 * (-3.7e-04 * beta + 6.16e-04 * beta**2.0))
    return phase


def phase_Saturn_3(beta):
    """Saturn phase function (Globe Only Pioneer Observations)
    Valid beta from 6 to 150. deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    difference = phase_Saturn_2(6.5) - 10.0 ** (
        -0.4
        * (
            2.446e-4 * 6.5
            + 2.672e-4 * 6.5**2.0
            - 1.505e-6 * 6.5**3.0
            + 4.767e-9 * 6.5**4.0
        )
    )
    phase = difference + 10.0 ** (
        -0.4
        * (
            2.446e-4 * beta
            + 2.672e-4 * beta**2.0
            - 1.505e-6 * beta**3.0
            + 4.767e-9 * beta**4.0
        )
    )
    return phase


def phase_Saturn_melded(beta):
    """
    Saturn phase function

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = (
        transitionEnd(beta, 6.5, 5.0) * phase_Saturn_2(beta)
        + transitionStart(beta, 6.5, 5.0)
        * transitionEnd(beta, 150.0, 5.0)
        * phase_Saturn_3(beta)
        + transitionStart(beta, 150.0, 5.0) * phi_lambert(beta * np.pi / 180.0)
    )
    return phase


def phiprime_phi(phi):
    """Helper method for Uranus phase function
    Valid for phi from -82 to 82 deg

    Args:
        phi (numpy.ndarray):
            phi, planet rotation axis offset in degrees, floats

    Returns:
        numpy.ndarray:
            phiprime, in deg, floats
    """
    f = 0.0022927  # flattening of the planet
    phiprime = np.arctan2(np.tan(phi * np.pi / 180.0), (1.0 - f) ** 2.0) * 180.0 / np.pi
    return phiprime


def phase_Uranus(beta, phi=-82.0):
    """Uranus phase function
    Valid for beta 0 to 154 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees
        phi (float or numpy.ndarray):
            phi, planet rotation axis offset in degrees, floats

    Returns:
        numpy.ndarray:
            phase function values
    """

    phase = 10.0 ** (
        -0.4 * (-8.4e-04 * phiprime_phi(phi) + 6.587e-3 * beta + 1.045e-4 * beta**2.0)
    )
    return phase


def phase_Uranus_melded(beta):
    """Uranus phase function

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = transitionEnd(beta, 154.0, 5.0) * phase_Uranus(beta) + transitionStart(
        beta, 154.0, 5.0
    ) * phi_lambert(beta * np.pi / 180.0)
    return phase


def phase_Neptune(beta):
    """Neptune phase function
    Valid for beta 0 to 133.14 deg

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = 10.0 ** (-0.4 * (7.944e-3 * beta + 9.617e-5 * beta**2.0))
    return phase


def phase_Neptune_melded(beta):
    """Neptune phase function

    Args:
        beta (numpy.ndarray):
            beta, phase angle in degrees

    Returns:
        numpy.ndarray:
            phase function values
    """
    phase = transitionEnd(beta, 133.14, 5.0) * phase_Neptune(beta) + transitionStart(
        beta, 133.14, 5.0
    ) * phi_lambert(beta * np.pi / 180.0)
    return phase


def realSolarSystemPhaseFunc(beta, phiIndex=np.asarray([])):
    """Uses the phase functions from Mallama 2018 implemented in
    mallama2018PlanetProperties.py

    Args:
        beta (astropy.units.quantity.Quantity or numpy.ndarray):
            phase angle array in degrees
        phiIndex (numpy.ndarray):
            array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        numpy.ndarray:
            Phi, phase function values between 0 and 1
    """
    if len(phiIndex) == 0:  # Default behavior is to use the lambert phase function
        Phi = np.zeros(len(beta))  # instantiate initial array
        Phi = phi_lambert(beta)
    else:
        if hasattr(beta, "unit"):
            beta = beta.to("rad").value
        beta = beta * 180.0 / np.pi  # convert to phase angle in degrees

        if not len(phiIndex) == 0 and (len(beta) == 1 or len(beta) == 0):
            beta = np.ones(len(phiIndex)) * beta
        Phi = np.zeros(len(beta))

        # Find indicies of where to use each phase function
        mercuryInds = np.where(phiIndex == 0)[0]
        venusInds = np.where(phiIndex == 1)[0]
        earthInds = np.where(phiIndex == 2)[0]
        marsInds = np.where(phiIndex == 3)[0]
        jupiterInds = np.where(phiIndex == 4)[0]
        saturnInds = np.where(phiIndex == 5)[0]
        uranusInds = np.where(phiIndex == 6)[0]
        neptuneInds = np.where(phiIndex == 7)[0]

        if not len(mercuryInds) == 0:
            Phi[mercuryInds] = phase_Mercury(beta[mercuryInds])
        if not len(venusInds) == 0:
            Phi[venusInds] = phase_Venus_melded(beta[venusInds])
        if not len(earthInds) == 0:
            Phi[earthInds] = phase_Earth(beta[earthInds])
        if not len(marsInds) == 0:
            Phi[marsInds] = phase_Mars_melded(beta[marsInds])
        if not len(jupiterInds) == 0:
            Phi[jupiterInds] = phase_Jupiter_melded(beta[jupiterInds])
        if not len(saturnInds) == 0:
            Phi[saturnInds] = phase_Saturn_melded(beta[saturnInds])
        if not len(uranusInds) == 0:
            Phi[uranusInds] = phase_Uranus_melded(beta[uranusInds])
        if not len(neptuneInds) == 0:
            Phi[neptuneInds] = phase_Neptune_melded(beta[neptuneInds])

    return Phi
