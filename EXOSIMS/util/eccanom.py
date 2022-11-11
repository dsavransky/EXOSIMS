import numpy as np


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

    # make sure M and e are of the correct format.
    # if 1 value provided for e, array must match size of M
    M = np.array(M).astype(float)
    if not M.shape:
        M = np.array([M])
    e = np.array(e).astype(float)
    if not e.shape:
        e = np.array([e] * len(M))

    assert e.shape == M.shape, "Incompatible inputs."
    assert np.all((e >= 0) & (e < 1)), "e defined outside [0,1)"

    # initial values for E
    E = M / (1 - e)
    mask = e * E**2 > 6 * (1 - e)
    E[mask] = (6 * M[mask] / e[mask]) ** (1.0 / 3)

    # Newton-Raphson setup
    tolerance = np.finfo(float).eps * 4.01
    numIter = 0
    maxIter = 200
    err = 1.0
    while err > tolerance and numIter < maxIter:
        E = E - (M - E + e * np.sin(E)) / (
            e * np.cos(E) - 1
        )  # verbatim from first page of Vallado
        err = np.max(abs(M - (E - e * np.sin(E))))
        numIter += 1

    if numIter == maxIter:
        raise Exception("eccanom failed to converge. Final error of %e" % err)

    return E
