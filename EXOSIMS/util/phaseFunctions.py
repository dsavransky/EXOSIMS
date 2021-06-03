"""
Phase Functions
Written By: Dean Keithly
"""
import numpy as np

def phi_lambert(alpha, phiIndex=np.asarray([])):
    """ Lambert phase function most easily found in Garrett2016 and initially presented in Sobolev 1975
    Args:
        ndarray:
            alpha, phase angle in radians, float
        ndarray:
            phiIndex, array of indicies of type of exoplanet phase function to use, ints 0-7
            
    Returns:
        ndarray:
            Phi, phase function values between 0 and 1
    """
    if hasattr(alpha,'value'):
        alpha = alpha.to('rad').value
    phi = (np.sin(alpha) + (np.pi-alpha)*np.cos(alpha))/np.pi
    return phi

def transitionStart(x,a,b):
    """ Smoothly transition from one 0 to 1
    Args:
        ndarray:
            x, in deg input value in deg, floats
        ndarray:
            a, transition midpoint in deg, floats
    Returns:
        ndarray:
            s, Transition value from 0 to 1, floats
    """
    s = 0.5+0.5*np.tanh((x-a)/b)
    return s

def transitionEnd(x,a,b):
    """ Smoothly transition from one 1 to 0
    Smaller b is sharper step
    a is midpoint, s(a)=0.5
    Args:
        ndarray:
            x, in deg input value in deg, floats
        ndarray:
            a, transition midpoint in deg, floats
    Returns:
        ndarray:
            s, Transition value from 0 to 1, floats
    """
    s = 0.5-0.5*np.tanh((x-a)/b)
    return s

def quasiLambertPhaseFunction(beta, phiIndex=np.asarray([])):
    """ Quasi Lambert Phase Function as presented
    Analytically Invertible Phase function from Agol 2007, 'Rounding up the wanderers: optimizing
    coronagraphic searches for extrasolar planets'
    Args:
        beta (numpy array):
            planet phase angles in radians
        ndarray:
            phiIndex, array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        ndarray:
            Phi, phase function value
    """
    Phi = np.cos(beta/2.)**4
    return Phi

def quasiLambertPhaseFunctionInverse(Phi, phiIndex=np.asarray([])):
    """ Quasi Lambert Phase Function Inverses'
    Args:
        ndarray:
            Phi, phase function value, floats
        ndarray:
            phiIndex, array of indicies of type of exoplanet phase function to use, ints 0-7

    Returns:
        ndarray:
            beta, planet phase angles, floats
    """
    beta = 2.*np.arccos((Phi)**(1./4.))
    return beta

def hyperbolicTangentPhaseFunc(beta,A,B,C,D,planetName=None):
    """
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties.py:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties.py:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213
    Args:
        beta (float):
            Phase Angle  in degrees
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
        float:
            Phi, phase angle in degrees
    """
    if planetName == None:
        None #do nothing
    elif planetName == 'mercury':
        A, B, C, D = 0.9441564 ,  0.3852919 ,  2.59291159, -0.67540991#0.93940195,  0.40446512,  2.47034733, -0.64361749
    elif planetName == 'venus':
        A, B, C, D = 1.20324931, 1.57402581, 0.60683886, 0.87010846#1.26116739, 1.53204409, 0.61961161, 0.84075693
    elif planetName == 'earth':
        A, B, C, D = 0.78414986, 1.86890464, 0.52958938, 1.07587225#0.78415   , 1.86890455, 0.5295894 , 1.07587213
    elif planetName == 'mars':
        A, B, C, D = 1.89881785,  0.48220465,  2.02299497, -1.02612681#2.02856459,  0.29590061,  3.32324214, -1.71048535
    elif planetName == 'jupiter':
        A, B, C, D = 1.3622441 , 1.45676529, 0.64490468, 0.7800663#1.3761512 , 1.45852349, 0.64157352, 0.7983722
    elif planetName == 'saturn':
        A, B, C, D = 5.02672862,  0.41588155,  1.80205383, -1.74369974#5.49410541,  0.37274869,  2.00119662, -2.1551928
    elif planetName == 'uranus':
        A, B, C, D = 1.54388146, 1.18304642, 0.79972526, 0.37288376#1.56866334, 1.16284633, 0.81250327, 0.34759469
    elif planetName == 'neptune':
        A, B, C, D = 1.31369238, 1.41437107, 0.67584636, 0.65077278#1.37105297, 1.36886173, 0.69506274, 0.609515
    beta = beta.to('rad').value
    Phi = -np.tanh((beta-D)/A)/B+C
    return Phi

def hyperbolicTangentPhaseFuncInverse(Phi,A,B,C,D,planetName=None):
    """
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties.py:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties.py:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213
    Args:
        Phi (float):
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
        float:
            beta, Phase Angle  in degrees
    """
    if planetName == None:
        None #do nothing
    elif planetName == 'mercury':
        A, B, C, D = 0.9441564 ,  0.3852919 ,  2.59291159, -0.67540991#0.93940195,  0.40446512,  2.47034733, -0.64361749
    elif planetName == 'venus':
        A, B, C, D = 1.20324931, 1.57402581, 0.60683886, 0.87010846#1.26116739, 1.53204409, 0.61961161, 0.84075693
    elif planetName == 'earth':
        A, B, C, D = 0.78414986, 1.86890464, 0.52958938, 1.07587225#0.78415   , 1.86890455, 0.5295894 , 1.07587213
    elif planetName == 'mars':
        A, B, C, D = 1.89881785,  0.48220465,  2.02299497, -1.02612681#2.02856459,  0.29590061,  3.32324214, -1.71048535
    elif planetName == 'jupiter':
        A, B, C, D = 1.3622441 , 1.45676529, 0.64490468, 0.7800663#1.3761512 , 1.45852349, 0.64157352, 0.7983722
    elif planetName == 'saturn':
        A, B, C, D = 5.02672862,  0.41588155,  1.80205383, -1.74369974#5.49410541,  0.37274869,  2.00119662, -2.1551928
    elif planetName == 'uranus':
        A, B, C, D = 1.54388146, 1.18304642, 0.79972526, 0.37288376#1.56866334, 1.16284633, 0.81250327, 0.34759469
    elif planetName == 'neptune':
        A, B, C, D = 1.31369238, 1.41437107, 0.67584636, 0.65077278#1.37105297, 1.36886173, 0.69506274, 0.609515
    beta = A*np.arctanh(-B*(Phi-C))+D
    return beta
    
def betaFunc(inc,v,w):
    """ Calculated the planet phase angle
    Args:
        ndarray:
            inc, planet inclination in rad
        ndarray:
            v, planet true anomaly in rad
        ndarray:
            w, planet argument of periapsis
    Returns:
        ndarray:
            beta, planet phase angle
    """
    beta = np.arccos(np.sin(inc)*np.sin(v+w))
    return beta

def phase_Mercury(alpha):
    """ Valid from 0 to 180 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.))
    return phase

def phase_Venus_1(alpha):
    """ Valid from 0 to 163.7 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(- 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.))
    return phase

def phase_Venus_2(alpha):
    """ Valid from 163.7 to 179 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*( - 2.81914e-00*alpha + 8.39034e-03*alpha**2.))
    #1 Scale Properly
    h1 = phase_Venus_1(163.7) - 0. #Total height desired over range
    h2 = 10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)) - 10.**(-0.4*( - 2.81914e-00*179. + 8.39034e-03*179.**2.))
    phase = phase * h1/h2 #Scale so height is proper
    #2 Lateral movement to make two functions line up
    difference = phase_Venus_1(163.7) - h1/h2*(10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)))
    phase = phase + difference
    return phase

def phase_Venus_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,163.7,5.)*phase_Venus_1(alpha) + \
        transitionStart(alpha,163.7,5.)*transitionEnd(alpha,179.,0.5)*phase_Venus_2(alpha) + \
        transitionStart(alpha,179.,0.5)*phi_lambert(alpha*np.pi/180.)+2.766e-04
        #2.666e-04 ensures the phase function is entirely positive (near 180 deg phase, there is a small region 
        #where phase goes negative) This small addition fixes this
    return phase

def phase_Earth(alpha):
    """ Valid from 0 to 180 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(- 1.060e-3*alpha + 2.054e-4*alpha**2.))
    return phase

def phase_Mars_1(alpha):
    """ Valid from 0 to 50 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(0.02267*alpha - 0.0001302*alpha**2.+ 0. + 0.))#L(λe) + L(LS)
    return phase

def phase_Mars_2(alpha):
    """ Valid from 50 to 180 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = phase_Mars_1(50.)/10.**(-0.4*(- 0.02573*50. + 0.0003445*50.**2.)) * 10.**(-0.4*(- 0.02573*alpha + 0.0003445*alpha**2. + 0. + 0.)) #L(λe) + L(Ls)
    return phase

def phase_Mars_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,50.,5.)*phase_Mars_1(alpha) + \
        transitionStart(alpha,50.,5.)*phase_Mars_2(alpha)
    return phase

def phase_Jupiter_1(alpha):
    """ Valid from 0 to 12 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(- 3.7e-04*alpha + 6.16e-04*alpha**2.))
    return phase

def phase_Jupiter_2(alpha):
    """ Valid from 12 to 130 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    # inds = np.where(alpha > 180.)[0]
    # alpha[inds] = [180.]*len(inds)
    # assert np.all((1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.) >= 0.), "error in alpha input"
    difference = phase_Jupiter_1(12.) - 10.**(-0.4*(- 2.5*np.log10(1.0 - 1.507*(12./180.) - 0.363*(12./180.)**2. - 0.062*(12./180.)**3.+ 2.809*(12./180.)**4. - 1.876*(12./180.)**5.)))
    phase = difference + 10.**(-0.4*(- 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)))
    return phase

def phase_Jupiter_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,12.,5.)*phase_Jupiter_1(alpha) + \
        transitionStart(alpha,12.,5.)*transitionEnd(alpha,130.,5.)*phase_Jupiter_2(alpha) + \
        transitionStart(alpha,130.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase

def phase_Saturn_2(alpha):
    """ Valid alpha from 0 to 6.5 deg
    Saturn Globe Only Earth Observations
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(- 3.7e-04*alpha +6.16e-04*alpha**2.))
    return phase

def phase_Saturn_3(alpha):
    """ Valid alpha from 6 to 150. deg
    Saturn Globe Only Pioneer Observations
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    difference = phase_Saturn_2(6.5) - 10.**(-0.4*(2.446e-4*6.5 + 2.672e-4*6.5**2. - 1.505e-6*6.5**3. + 4.767e-9*6.5**4.))
    phase = difference + 10.**(-0.4*(2.446e-4*alpha + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**4.))
    return phase

def phase_Saturn_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,6.5,5.)*phase_Saturn_2(alpha) + \
                transitionStart(alpha,6.5,5.)*transitionEnd(alpha,150.,5.)*phase_Saturn_3(alpha)  + \
                transitionStart(alpha,150.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase

def phiprime_phi(phi):
    """ Valid for phi from -82 to 82 deg
    Args:
        ndarray:
            phi, planet rotation axis offset in degrees, floats
    Returns:
        ndarray:
            phiprime, in deg, floats
    """
    f = 0.0022927 #flattening of the planet
    phiprime = np.arctan2(np.tan(phi*np.pi/180.),(1.-f)**2.)*180./np.pi
    return phiprime

def phase_Uranus(alpha,phi=-82.):
    """ Valid for alpha 0 to 154 deg
    Args:
        ndarray:
            alpha, phase angle in degrees
        float:
            phi, planet rotation axis offset in deg
    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(- 8.4e-04*phiprime_phi(phi) + 6.587e-3*alpha + 1.045e-4*alpha**2.))
    return phase

def phase_Uranus_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,154.,5.)*phase_Uranus(alpha) + \
        transitionStart(alpha,154.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase

def phase_Neptune(alpha):
    """ Valid for alpha 0 to 133.14 deg
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = 10.**(-0.4*(7.944e-3*alpha + 9.617e-5*alpha**2.))
    return phase

def phase_Neptune_melded(alpha):
    """
    Args:
        ndarray:
            alpha, phase angle in degrees

    Returns:
        ndarray:
            phase, phase function values
    """
    phase = transitionEnd(alpha,133.14,5.)*phase_Neptune(alpha) + \
        transitionStart(alpha,133.14,5.)*phi_lambert(alpha*np.pi/180.)
    return phase

def realSolarSystemPhaseFunc(beta,phiIndex=np.asarray([])):
    """ Uses the phase functions from Mallama 2018 implemented in mallama2018PlanetProperties.py
    Args:
        ndarray:
            beta, array of phase angles in radians, floats
        ndarray:
            phiIndex, array of indicies of type of exoplanet phase function to use, ints 0-7
    Returns:
        ndarray:
            Phi, phase function value, empty or array of ints
    """
    Phi = np.zeros(len(beta)) #instantiate initial array

    if len(phiIndex) == 0: #Default behavior is to use the lambert phase function
        Phi = phi_lambert(beta)
    else:
        if hasattr(beta,'unit'): #might not work properly if beta passed in isnt an ndarray
            beta = beta.to('rad').value
        alpha = beta*180./np.pi #convert to phase angle in degrees

        #Find indicies of where to use each phase function
        mercuryInds = np.where(phiIndex == 0)[0]
        venusInds = np.where(phiIndex == 1)[0]
        earthInds = np.where(phiIndex == 2)[0]
        marsInds = np.where(phiIndex == 3)[0]
        jupiterInds = np.where(phiIndex == 4)[0]
        saturnInds = np.where(phiIndex == 5)[0]
        uranusInds = np.where(phiIndex == 6)[0]
        neptuneInds = np.where(phiIndex == 7)[0]

        if not len(mercuryInds) == 0:
            Phi[mercuryInds] = phase_Mercury(alpha[mercuryInds])
        if not len(venusInds) == 0:
            Phi[venusInds] == phase_Venus_melded(alpha[venusInds])
        if not len(earthInds) == 0:
            Phi[earthInds] == phase_Earth(alpha[earthInds])
        if not len(marsInds) == 0:
            Phi[marsInds] == phase_Mars_melded(alpha[marsInds])
        if not len(jupiterInds) == 0:
            Phi[jupiterInds] == phase_Jupiter_melded(alpha[jupiterInds])
        if not len(saturnInds) == 0:
            Phi[saturnInds] == phase_Saturn_melded(alpha[saturnInds])
        if not len(uranusInds) == 0:
            Phi[uranusInds] == phase_Uranus_melded(alpha[uranusInds])
        if not len(neptuneInds) == 0:
            Phi[neptuneInds] == phase_Neptune_melded(alpha[neptuneInds])

    return Phi


