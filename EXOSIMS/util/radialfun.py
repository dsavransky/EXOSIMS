"""
Utilities for radial computations on rectangular data arrays
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import optimize


def pixel_dists(dims, center):
    """
    Compute pixel distances from center of an image

    Args:
        dims (tuple(float,float)):
            Image dimensions
        center (list(float,float)):
            [x,y] pixel coordinates of center

    Returns:
        numpy.ndarray:
            Array of dimension dims with distance from center of each pixel
    """

    y, x = np.indices(dims)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    return r


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
    r = pixel_dists(dims, center)

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


def circ_aperture(im, rho, center, return_sum=False):
    """
    Extract pixels in circular aperture

    Args:
        im (numpy.ndarray):
            The input image. Must be 2-dimensional
        rho (float):
            Radius of aperture (in pixels)
        center (list(float,float):
            [x,y] pixel coordinates of center of aperture
        return_sum (bool):
            Return sum. Defaults False: returns all pixels in aperture.

    Returns:
        numpy.ndarray:
            1-dimensional array of pixel values inside aperture
    """

    # compute pixel distances from center
    r = pixel_dists(im.shape, center)

    pix = im[r <= rho]

    if return_sum:
        out = np.sum(pix[np.isfinite(pix)])
    else:
        out = pix

    return out


def com(im0, fill_val=0):
    """
    Find the center-of-mass centroid of an image

    Args:
        im0 (numpy.ndarray):
            The input image. Must be 2-dimensional
        fill_val (float):
            Replace any non finite values in the image with this value before
            resampling. Defaults to zero.

    Returns:
        list:
            [x,y] pixel coordinates of COM
    """

    # operate on input copy and zero out bad values
    im = im0.copy()
    im[~np.isfinite(im)] = fill_val

    y, x = np.indices(im.shape)

    return [np.sum(im * inds) / np.sum(im) for inds in [x, y]]


def genwindow(dims):
    """
    Create a 2D Hann window of the given dimensions

    Args:
        dims (tuple or list):
            2-element dimensions of window (can be the .shape output of an ndarray)

    Returns:
        numpy.ndarray:
            Window of dimensions ``dims``.

    """

    window = np.ones(dims)
    y, x = np.indices(dims)
    for dim, inds in zip(dims, [x, y]):
        window *= 0.5 - 0.5 * np.cos(2 * np.pi * inds / dim)
    window = np.sqrt(window)

    return window


def resample_image(im, resamp=2, fill_val=0):
    """
    Create a resampled image

    Args:
        im (numpy.ndarray):
            The input image. Must be 2-dimensional
        resamp (float):
            Resampling factor. Must be >=1. Defaults to 2
        fill_val (float):
            Replace any non finite values in the image with this value before
            resampling. Defaults to zero.

    Returns:
        numpy.ndarray:
            Resampled image.

    """

    assert resamp >= 1, "resamp must be >= 1"
    if resamp == 1:
        return im

    dims = im.shape
    newdims = [(d - 1) * resamp + 1 for d in dims]

    imc = im.copy()
    imc[~np.isfinite(imc)] = fill_val
    sp = RectBivariateSpline(np.arange(dims[0]), np.arange(dims[0]), imc)
    imresamp = sp(np.arange(newdims[0]) / resamp, np.arange(newdims[1]) / resamp)

    return imresamp


def gaussian(a, x0, y0, sx, sy):
    """Gaussian function

    Args:
        a (float):
            Amplitude
        x0 (float):
            Center (mean) x position
        y0 (float):
            Center (mean) y position
        sx (float):
            Standard deviation in x
        sy (float):
            Standard deviation in y

    Returns:
        lambda:
            Callable lambda function with input x,y returning value of Gaussian at those
            coordinates
    """

    return lambda x, y: a * np.exp(
        -((x - x0) ** 2) / 2 / sx**2 - (y - y0) ** 2 / 2 / sy**2
    )


def fitgaussian(im):
    """Fit a 2D Gaussian to data

    Args:
        im (numpy.ndarray):
            2D data array

    Returns:
        tuple:
            a (float):
                Amplitude
            x0 (float):
                Center (mean) x position
            y0 (float):
                Center (mean) y position
            sx (float):
                Standard deviation in x
            xy (float):
                Standard deviation in y
    """

    Y, X = np.indices(im.shape)
    total = im.sum()
    x0 = (X * im).sum() / total
    y0 = (Y * im).sum() / total
    col = im[:, int(x0)]
    sx0 = np.sqrt(np.abs((np.arange(col.size) - x0) ** 2 * col).sum() / col.sum() / 2)
    row = im[int(y0), :]
    sy0 = np.sqrt(np.abs((np.arange(row.size) - y0) ** 2 * row).sum() / row.sum() / 2)
    a0 = im.max()

    errorfunction = lambda p: np.ravel(gaussian(*p)(X, Y) - im)
    p, success = optimize.leastsq(errorfunction, (a0, x0, y0, sx0, sy0))
    return p
