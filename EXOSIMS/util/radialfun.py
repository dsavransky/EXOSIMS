"""
Utilities for radial computations on rectangular data arrays
"""

import numpy as np


def radial_average(im, center=None, nbins=None):
    """
    Compute radial average on an image

    Args:
        im (numpy.ndarray):
            The input image. Must be 2-dimensional
        center (list(float,float), optional):
            [x,y] pixel coordinates of center to compute average about.
            If None (default) use geometric center of input.
        nbins (int, optional):
            Number of bins to compute average in. If None (default) then set to
            floor(N/2) where N is the maximum dimension of the input image.

    Returns:
        tuple:
            means (numpy.ndarray):
                ``nbins`` element array with radial average values
            bins (numpy.nadarray):
                ``nbins+1`` element array with bin boundaries.
            bincents (numpy.ndarray):
                ``nbins`` elements array with bin midpoints. Equivalent to
                ``(bins[1:] + bins[:-1]) / 2``
    """

    # gather info
    dims = im.shape
    assert len(dims) == 2, "Only 2D images are supported."

    if center is None:
        center = [dims[0] / 2.0, dims[1] / 2.0]

    if nbins is None:
        nbins = int(np.floor(np.max(dims) / 2))

    # compute pixel distances from center
    y, x = np.indices(dims)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # define bins and digitize distanes
    bins = np.linspace(0, np.max(r), nbins + 1)
    bincents = (bins[1:] + bins[:-1]) / 2
    inds = np.digitize(r.ravel(), bins)
    # max value will be in its own bin, so put all matching pixels in the last valid bin
    inds[inds == nbins + 1] = nbins

    # compute means in each bin
    means = np.zeros(nbins)
    imflat = im.ravel()
    for j in range(1, nbins + 1):
        means[j - 1] = np.nanmean(imflat[inds == j])

    return means, bins, bincents
