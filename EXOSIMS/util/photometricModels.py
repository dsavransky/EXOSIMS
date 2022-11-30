"""
Various useful photometric models from the literature
"""
import astropy.units as u
from synphot.models import Box1D as Box1D_orig
from synphot.compat import ASTROPY_LT_5_0
from functools import partial
import numpy as np


def TraubStellarFluxDensity(V, BV, lam):
    """
    Stellar photon flux density at a given wavelength.

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
            Stellar flux desnsity at wavelength lam. This will have the same
            dimensionality as the V and BV inputs. Default units are ph/s/cm^2/nm
    """

    a = 4.01 - (lam.to(u.um).value - 0.55) / 0.770
    b = 2.20 if lam < 0.55 * u.um else 1.54
    m = V + b * BV * (1 / lam.to(u.um).value - 1.818)
    f = 10 ** (a - 0.4 * m) * u.ph / u.s / u.cm**2 / u.nm

    return f


class Box1D(Box1D_orig):
    """Same as `synphot.models.Box1D`, except with ``step`` input.
    """

    def __init__(self, step=0.01, *args, **kwargs):

        super(Box1D, self).__init__(*args, **kwargs)
        self.step = float(step)

    def sampleset(self, step=None, minimal=False):
        """Return ``x`` array that samples the feature.

        Parameters
        ----------
        step : float
            Distance of first and last points w.r.t. bounding box.

        minimal : bool
            Only return the minimal points needed to define the box;
            i.e., box edges and a point outside on each side.

        """
        if step is None:
            step = self.step

        if ASTROPY_LT_5_0:
            w1, w2 = self.bounding_box
        else:
            w1, w2 = tuple(self.bounding_box.bounding_box())

        if self._n_models == 1:
            w = self._calc_sampleset(w1, w2, step, minimal)
        else:
            w = list(
                map(partial(self._calc_sampleset, step=step, minimal=minimal), w1, w2)
            )

        return np.asarray(w)
