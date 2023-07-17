# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


class StarCatalog(object):
    """:ref:`StarCatalog` Prototype

    Args:
        ntargs (int):
            Number of stars in catalog. Defaults to 1.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        VmagFill (float):
            Fill value for V magnitudes. Defaults to 0.1. Must be set to non-zero value
            or TargetList will fail to build.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        catalog_atts (list):
            All star catalog attributes that were copied in
        ntargs (int):
            Number of stars
        Name (~numpy.ndarray(str)):
            Star names
        Spec (~numpy.ndarray(str)):
            Star spectral types
        Umag (~numpy.ndarray(float)):
            U magnitude
        Bmag (~numpy.ndarray(float)):
            B magnitude
        Vmag (~numpy.ndarray(float)):
            V magnitude
        Rmag (~numpy.ndarray(float)):
            R magnitude
        Imag (~numpy.ndarray(float)):
            I magnitude
        Jmag (~numpy.ndarray(float)):
            J magnitude
        Hmag (~numpy.ndarray(float)):
            H magnitude
        Kmag (~numpy.ndarray(float)):
            K magnitude
        BV (~numpy.ndarray(float)):
            B-V Johnson magnitude
        MV (~numpy.ndarray(float)):
            Absolute V magnitude
        BC (~numpy.ndarray(float)):
            Bolometric correction
        L (~numpy.ndarray(float)):
            Stellar luminosity in Solar luminosities
        Binary_Cut (~numpy.ndarray(bool)):
            Boolean where True is a binary star with companion closer than 10 arcsec
        dist (~astropy.units.Quantity(~numpy.ndarray(float))):
            Distance to star in units of pc
        parx (~astropy.units.Quantity(~numpy.ndarray(float))):
            Parallax in units of mas
        coords (astropy.coordinates.SkyCoord):
            SkyCoord object (ICRS frame) containing right ascension, declination, and
            distance to star in units of deg, deg, and pc
        pmra (~astropy.units.Quantity(~numpy.ndarray(float))):
            Proper motion in right ascension in units of mas/year
        pmdec (~astropy.units.Quantity(~numpy.ndarray(float))):
            Proper motion in declination in units of mas/year
        rv (~astropy.units.Quantity(~numpy.ndarray(float))):
            Radial velocity in units of km/s
        cachedir (str):
            Path to cache directory

    .. note::

        The prototype will generate empty arrays for all attributes of size ntargs.
        These can then be either filled in or overwritten by inheriting implementations.

    """

    _modtype = "StarCatalog"

    def __init__(self, ntargs=1, cachedir=None, VmagFill=0.1, **specs):

        # start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # ntargs must be an integer >= 1
        self.ntargs = max(int(ntargs), 1)

        # list of astropy attributes
        self.dist = np.ones(ntargs) * u.pc  # distance
        self.parx = self.dist.to("mas", equivalencies=u.parallax())  # parallax
        self.coords = SkyCoord(
            ra=np.zeros(ntargs) * u.deg,
            dec=np.zeros(ntargs) * u.deg,
            distance=self.dist,
        )  # ICRS coordinates
        self.pmra = np.zeros(ntargs) * u.mas / u.yr  # proper motion in RA
        self.pmdec = np.zeros(ntargs) * u.mas / u.yr  # proper motion in DEC
        self.rv = np.zeros(ntargs) * u.km / u.s  # radial velocity

        # list of non-astropy attributes
        self.Name = np.array([f"Prototype Star {j}" for j in range(ntargs)])
        self.Spec = np.array(["G"] * ntargs)  # spectral types
        self.Umag = np.zeros(ntargs)  # U magnitude
        self.Bmag = np.zeros(ntargs)  # B magnitude
        self.Vmag = np.zeros(ntargs) + VmagFill  # V magnitude
        self.Rmag = np.zeros(ntargs)  # R magnitude
        self.Imag = np.zeros(ntargs)  # I magnitude
        self.Jmag = np.zeros(ntargs)  # J magnitude
        self.Hmag = np.zeros(ntargs)  # H magnitude
        self.Kmag = np.zeros(ntargs)  # K magnitude
        self.BV = np.zeros(ntargs)  # B-V Johnson magnitude
        self.MV = np.zeros(ntargs)  # absolute V magnitude
        self.BC = np.zeros(ntargs)  # bolometric correction
        self.L = np.ones(ntargs)  # stellar luminosity in solar units
        self.Binary_Cut = np.zeros(ntargs, dtype=bool)  # binary closer than 10 arcsec

        # populate outspecs
        self._outspec["ntargs"] = self.ntargs
        self._outspec["VmagFill"] = VmagFill

        # define list of provided catalog attributes
        self.catalog_atts = [
            "Name",
            "Spec",
            "parx",
            "dist",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "BV",
            "MV",
            "BC",
            "L",
            "Binary_Cut",
        ]

    def __str__(self):
        """String representation of the StarCatalog object

        When the command 'print' is used on the StarCatalog object, this method
        will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Star Catalog class object attributes"
