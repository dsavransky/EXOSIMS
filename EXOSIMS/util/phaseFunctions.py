""" Phase Functions

Written By: Dean Keithly
"""
import numpy as np

def quasiLambertPhaseFunction(beta):
    """ Quasi Lambert Phase Function as presented
    Analyticall Invertible Phase function from Agol 2007, 'Rounding up the wanderers: optimizing
    coronagraphic searches for extrasolar planets'
    Args:
        beta (numpy array):
            planet phase angles
    Returns:
        Phi (numpy array):
            phase function value
    """
    Phi = np.cos(beta/2)**4
    return Phi

def quasiLambertPhaseFunctionInverse(Phi):
    """ Quasi Lambert Phase Function Inverses'
    Args:
        Phi (numpy array):
            phase function value
        
    Returns:
        beta (numpy array):
            planet phase angles
    """
    beta = 2.*np.arccos((Phi)**(1./4.))
    return beta

#### Hyperbolic Phase Function
def hyperbolicTangentPhaseFunc(beta,A,B,C,D,planetName=None):
    """
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213
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
    Optimal Parameters for Earth Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties:
    A=1.85908529,  B=0.89598952,  C=1.04850586, D=-0.08084817
    Optimal Parameters for All Solar System Phase Function basedon mallama2018 comparison using mallama2018PlanetProperties:
    A=0.78415 , B=1.86890455, C=0.5295894 , D=1.07587213

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