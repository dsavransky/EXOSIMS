import numpy as np
import numbers

def eccanom(M, e):
    """Finds eccentric anomaly from mean anomaly and eccentricity
    
    This method uses algorithm 2 from Vallado to find the eccentric anomaly
    from mean anomaly and eccentricity.
    
    Args:
        M (float or ndarray):
            mean anomaly
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.
            
    Returns:
        E (float or ndarray):
            eccentric anomaly
    
    """
    
    assert isinstance(M,(np.ndarray,numbers.Number)),\
            "M must be a number or array."
    
    assert isinstance(e,(np.ndarray,numbers.Number)),\
            "e must be a number or array."
    
    retscalar = False
    if isinstance(M, numbers.Number):
        M = np.array([M])
        retscalar = True

    if isinstance(e, numbers.Number):
        e = np.array([e]*len(M))

    assert e.shape == M.shape, "Incompatible inputs."

    if np.all(e == 0):
        if retscalar:
            return M[0]
        else:
            return M


    #initial values for E
    E = M/(1-e)
    inds = E > np.sqrt(6*(1-e)/e)
    E[inds] = (6*M[inds]/e[inds])**(1./3)

    # Newton-Raphson setup
    tolerance = np.finfo(float).eps*4.01
    numIter = 0
    maxIter = 200
    err = 1.
    while err > tolerance and numIter < maxIter:

        E = E - (M - E + e*np.sin(E))/(e*np.cos(E)-1)
        err = np.max(abs(M - (E - e*np.sin(E))))
        numIter += 1
    
    if numIter == maxIter:
        raise Exception("Failed to converge.")

    if retscalar:
        E = E[0]
            
    return E

