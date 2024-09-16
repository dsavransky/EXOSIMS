import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


def partitionSphere(N, d):
    """Find coordinates of N equally-spaced targets on the unit sphere

    Args:
        N (int):
            Number of points
        d (float):
            Distance to assign to targets (pc)

    Returns:
        ~astropy.coordinates.SkyCoord:
            Coordinates of equally-spaced targets

    """

    Atot = 4 * np.pi  # total surface area for 2D sphere
    Ar = Atot / N  # area of each partition

    # spherical caps have area Ar, begin at certain colatitude
    PHIc = np.arccos(1 - Atot / (2 * np.pi * N))

    # COLLAR ANGLES (LATITUDE)
    dI = np.sqrt(Ar)  # ideal collar angles measured as colatitudes
    nI = (np.pi - 2 * PHIc) / dI  # ideal # of collars

    # actual collar angles
    nF = int(np.max([1, np.round(nI)]))  # actual # of collars
    dF = (np.pi - 2 * PHIc) / nF  # 'fitting' angle for each collar

    collars = np.zeros(nF + 1)
    COLAT = np.zeros(nF + 2)  # colatitudes of each collar
    COLAT[1] = PHIc + dF / 2
    COLAT[-1] = np.pi

    # colatitudes of each collar
    for i in range(nF + 1):
        collars[i] = PHIc + i * dF

    # colatitudes of each star
    for i in np.arange(2, nF + 1):
        COLAT[i] = COLAT[i - 1] + dF

    # ZONE PARTITIONS (LONGITUDE)
    Acap = Ar
    Az = np.zeros(nF + 2)
    Az[0] = Acap
    Az[-1] = Acap
    for i in range(nF):
        Az[i + 1] = 2 * np.pi * (np.cos(collars[i]) - np.cos(collars[i + 1]))

    yJ = Az / Ar  # ideal zone partitions
    mJ = np.ones(nF + 2)
    aJ = np.zeros(nF + 2)

    for j in np.arange(1, nF + 2):
        mJ[j] = round(yJ[j] + aJ[j - 1])
        aJ[j] = aJ[j - 1] + (yJ[j] - mJ[j])

    # putting it all together
    theta = np.zeros(N)
    phi = np.zeros(N)
    S = 0
    for i in range(nF + 2):
        K = int(mJ[i])
        LONG = np.zeros(K)
        for k in range(K):
            LONG[k] = (np.pi / K) * (1 + 2 * k)
        S += K
        for s, k in zip(np.arange(S - K, S), range(K)):
            phi[s] = np.pi / 2 - COLAT[i]
            theta[s] = LONG[k]

    theta_0 = theta[0]
    phi_0 = phi[0]

    psi = np.arccos(
        np.cos(phi_0) * np.cos(phi) * np.cos(theta_0 - theta)
        + np.sin(phi_0) * np.sin(phi)
    )
    psi = np.sort(psi)
    theta_, phi_ = add_caps(np.max(np.abs(np.diff(psi))))

    theta = np.append(theta, theta_)
    phi = np.append(phi, phi_)

    coords = SkyCoord(theta * u.rad, phi * u.rad, d * np.ones(len(phi)) * u.pc)
    return coords


def add_caps(psi):
    """Helper method for partitionSphere. Computes coordinates on caps of sphere

    Args:
        psi (float):
            Cap angle (rad)

    Returns:
        tuple:
            theta (float):
                Azimuth angle (rad)
            phi (float):
                Elevation angle (rad)

    """

    As = 2 * np.pi * (1 - np.cos(psi))  # area of polar cap specified by psi
    At = 4 * np.pi  # total area
    f = As / At  # fraction of stars in cap
    N = int(round(1 / f) * 2)  # number of stars in the cap

    Ar = As / N  # area of each individual area
    dI = np.sqrt(Ar)  # ideal collar angle

    nI = psi / dI  # ideal number of collars
    n = int(np.max([1, np.round(nI)]))  # actual number of collars
    dF = (nI / n) * dI  # fitted collar angle

    COLAT = np.zeros(n - 1)

    # colatitudes of each star
    for i in np.arange(1, n - 1):
        COLAT[i] = COLAT[i - 1] + dF

    # ZONE PARTITIONS (LONGITUDE)
    Az = np.zeros(n - 1)
    for i in range(n - 2):
        Az[i + 1] = 2 * np.pi * (np.cos(COLAT[i]) - np.cos(COLAT[i + 1]))

    yJ = Az / Ar  # ideal zone partitions
    mJ = np.ones(n - 1)
    aJ = np.zeros(n - 1)

    for j in np.arange(1, n - 1):
        mJ[j] = round(yJ[j] + aJ[j - 1])
        aJ[j] = aJ[j - 1] + (yJ[j] - mJ[j])

    # putting it all together
    theta = np.zeros(N)
    phi = np.zeros(N)
    S = 0
    for i in range(n - 1):
        K = int(mJ[i])
        LONG = np.zeros(K)
        for k in range(K):
            LONG[k] = (np.pi / K) * (1 + 2 * k)
        S += K
        for s, k in zip(np.arange(S - K, S), range(K)):
            phi[s] = np.pi / 2 - COLAT[i]  # dec - phi
            theta[s] = LONG[k]  # ra  - theta

    goodIdx = np.where((phi > 0) & (phi is not np.pi / 2))
    theta = theta[goodIdx]
    phi = phi[goodIdx]

    theta = np.append(theta, theta)
    phi = np.append(phi, -phi)

    RAsort = np.argsort(theta)

    return theta[RAsort], phi[RAsort]
