import numpy as np
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import astropy.units as u
from astropy.coordinates import SkyCoord


class BackgroundSources(object):
    """:ref:`BackgroundSources` Prototype

    Args:
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        _outspec (dict):
            :ref:`sec:outspec`

    """

    _modtype = "BackgroundSources"

    def __init__(self, cachedir=None, **specs):
        self._outspec = {}

        # cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        return

    def __str__(self):
        """String representation of Background Sources module"""

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Background Sources class object attributes"

    def dNbackground(self, coords, intDepths):
        """Returns background source number densities

        Args:
            coords (~astropy.coordinates.SkyCoord):
                SkyCoord object containing right ascension, declination, and
                distance to star of the planets of interest in units of deg, deg and pc
            intDepths (~numpy.ndarray(float)):
                Integration depths equal to the planet magnitude (Vmag+dMag),
                i.e. the V magnitude of the dark hole to be produced for each target.
                Must be of same length as coords.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                dN: Number densities of background sources for given targets in
                units of 1/arcmin2. Same length as inputs.

        """

        assert isinstance(
            intDepths, (tuple, list, np.ndarray)
        ), "intDepths is not array-like."
        if isinstance(coords, SkyCoord):
            assert coords.shape, "coords is not array-like."
        else:
            assert isinstance(
                coords, (tuple, list, np.ndarray)
            ), "coords is not array-like."
        assert len(coords) == len(intDepths), "Input size mismatch."

        dN = np.zeros(len(intDepths))

        return dN / u.arcmin**2
