"""
Various useful photometric models from the literature
"""
import astropy.units as u


def TraubStellarFlux(V, BV, lam):
    """
    Stellar photon flux at a given wavelength.

    This implements the model from [Traub2016]_ which has a stated valid range between
    0.4 and 1 um.

    .. warning::

        Values will be returned for wavelengths outside the valid range, but a warning
        will be generated.

    Args:
        V (float or ~numpy.ndarray(float)):
            Johnson V-band magnitude(s)
        BV (float or ~numpy.ndarray(float)):
            B-V color.  Must be same type and dimensionality as ``V``
        lam (astropy.units.Quantity):
            Wavelength at which to evaluate the flux.  Must be scalar.

    Returns:
        ~astropy.units.Quantity:
            Stellar flux(es) at wavelength lam. This will have the same dimensionality
            as the V and BV inputs. Default units are ph/s/cm^2/nm
    """

    a = 4.01 - (lam.to(u.um).value - 0.55) / 0.770
    b = 2.20 if lam < 0.55 * u.um else 1.54
    m = V + b * BV * (1 / lam.to(u.um).value - 1.818)
    f = 10 ** (a - 0.4 * m) * u.ph / u.s / u.cm**2 / u.nm

    return f
