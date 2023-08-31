# -*- coding: utf-8 -*-
from EXOSIMS.ZodiacalLight.Stark import Stark
import numpy as np
import os
import astropy.units as u
from astropy.io import fits


class Mennesson(Stark):
    """Mennesson Zodiacal Light class"""

    def __init__(self, EZ_distribution="nominal_maxL_distribution.fits", **specs):
        Stark.__init__(self, **specs)
        if os.path.exists(os.path.normpath(os.path.expandvars(EZ_distribution))):
            self.EZ_distribution = os.path.normpath(os.path.expandvars(EZ_distribution))
        elif os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), EZ_distribution)
        ):
            self.EZ_distribution = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), EZ_distribution
            )
        else:
            raise ValueError(f"Could not locate EZ_distribution file {EZ_distribution}")
        self.fitsdata = fits.open(self.EZ_distribution)[0].data
        self._outspec["EZ_distribution"] = EZ_distribution

    def gen_systemnEZ(self, nStars):
        """Ranomly generates the number of Exo-Zodi
        Args:
            nStars (int):
                number of exo-zodi to generate
        Returns:
            nEZ (numpy array):
                numpy array of exo-zodi randomly selected from fitsdata
        """
        nEZ_seed = np.random.randint(len(self.fitsdata) - nStars)
        nEZ = 2 * self.fitsdata[nEZ_seed : (nEZ_seed + nStars)]
        return nEZ

    def zodi_latitudinal_correction_factor(self, theta, model=None, interp_at=135):
        """
        Compute zodiacal light latitudinal correction factor.  This is a multiplicative
        factor to apply to zodiacal light intensity to account for the orientation of
        the dust disk with respect to the observer.

        Args:
            theta (astropy.units.Quantity):
                Angle of disk. For local zodi, this is equivalent to the absolute value
                of the ecliptic latitude of the look vector. For exozodi, this is 90
                degrees minus the inclination of the orbital plane.
            model (str, optional):
                Model to use.  Options are Lindler2006, Stark2014, or interp
                (case insensitive). See :ref:`zodiandexozodi` for details.
                Defaults to None
            interp_at (float):
                If ``model`` is 'interp', interpolate Leinert Table 17 at this
                longitude. Defaults to 135.

        Returns:
            float or numpy.ndarray:
                Correction factor of zodiacal light at requested angles.
                Has same dimension as input.

        .. note::

            Unlike the color correction factor, this quantity is wavelength independent
            and thus does not change if using power or photon units.

        .. note::

            The systems in the data file are all at 60 degrees inclination, so we scale
            by the 90-60=30 degree value of the correction factor.

        """

        fbeta = super().zodi_latitudinal_correction_factor(
            theta, model=model, interp_at=interp_at
        )
        fbeta30 = super().zodi_latitudinal_correction_factor(
            30 * u.deg, model=model, interp_at=interp_at
        )

        return fbeta / fbeta30
