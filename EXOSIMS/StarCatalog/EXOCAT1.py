# -*- coding: utf-8 -*-
import os
import inspect
import warnings
import numpy as np
import astropy
import astropy.units as u
from astropy.constants import R_sun
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import re
import astropy.io.ascii


class EXOCAT1(StarCatalog):
    """
    EXOCAT Catalog class

    This class populates the star catalog used in EXOSIMS from
    Margaret Turnbull's EXOCAT catalog, retrieved from the
    NASA Exoplanet Archive as a VOTABLE.
    Documentation of fields available at:
    https://exoplanetarchive.ipac.caltech.edu/docs/API_mission_stars.html

    Args:
        catalogpath (str):
            Full path to catalog VOTABLE. If None (default) uses default
            catalogfile in  EXOSIMS.StarCatalog directory.
        wdsfilepath (str):
            Full path to WDS catalog
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        Only StarCatalog prototype attributes are used.

    """

    def __init__(self, catalogpath=None, wdsfilepath=None, **specs):

        if catalogpath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = "mission_exocat_2019.08.22_11.37.24.votable"
            catalogpath = os.path.join(classpath, filename)

        if not os.path.exists(catalogpath):
            raise IOError("Catalog File %s Not Found." % catalogpath)

        # read votable
        with warnings.catch_warnings():
            # warnings for IPAC votables are out of control
            #   they are not moderated by pedantic=False
            #   they all have to do with units, which we handle independently anyway
            warnings.simplefilter(
                "ignore", astropy.io.votable.exceptions.VOTableSpecWarning
            )
            warnings.simplefilter(
                "ignore", astropy.io.votable.exceptions.VOTableChangeWarning
            )
            votable = parse(catalogpath)
        table = votable.get_first_table()
        data = table.array

        StarCatalog.__init__(self, ntargs=len(data), **specs)
        self._outspec["catalogpath"] = catalogpath
        self._outspec["wdsfilepath"] = wdsfilepath

        # list of astropy attributes
        # Distance to the planetary system in units of parsecs
        self.dist = data["st_dist"].data * u.pc
        # parallactic angle in units of mas
        self.parx = self.dist.to("mas", equivalencies=u.parallax())
        # Right Ascension of the planetary system in decimal degrees,
        # Declination of the planetary system in decimal degrees
        self.coords = SkyCoord(
            ra=data["ra"] * u.deg, dec=data["dec"] * u.deg, distance=self.dist
        )
        # Angular change in right ascension over time as seen from the center of mass
        # of the Solar System, units (mas/yr)
        self.pmra = data["st_pmra"].data * u.mas / u.yr
        # Angular change in declination over time as seen from the center of mass of
        # the Solar System, units (mas/yr)
        self.pmdec = data["st_pmdec"].data * u.mas / u.yr
        # Amount of energy emitted by a star per unit time, measured in units of solar
        # luminosities. The bolometric corrections are derived from V-K or B-V colors,
        # units [L_solar]
        self.L = data["st_lbol"].data

        # list of non-astropy attributes
        # Name of the star as given by the Hipparcos Catalog.
        self.Name = data["hip_name"].astype(str)
        # Classification of the star based on their spectral characteristics following
        # the Morgan-Keenan system
        self.Spec = data["st_spttype"].astype(str)
        self.Vmag = data["st_vmag"]  # V mag
        self.Jmag = data["st_j2m"]  # Stellar J (2MASS) Magnitude Value
        self.Hmag = data["st_h2m"]  # Stellar H (2MASS) Magnitude Value
        # Color of the star as measured by the difference between B and V bands,
        # units of [mag]
        self.BV = data["st_bmv"]
        self.Bmag = self.Vmag + data["st_bmv"]  # B mag based on BV color
        self.Kmag = self.Vmag - data["st_vmk"]  # K mag based on VK color
        # st_mbol Apparent magnitude of the star at a distance of 10 parsec
        # units of [mag]
        self.BC = -self.Vmag + data["st_mbol"]  # bolometric correction
        # absolute V mag
        self.MV = self.Vmag - 5.0 * (np.log10(self.dist.to("pc").value) - 1.0)
        # stellar_diameters in solar diameters
        self.stellar_diameters = data["st_rad"] * 2.0 * R_sun
        # appears in WDS (Washington Double Star) Catalog
        self.Binary_Cut = ~data["wds_sep"].mask

        # if given a WDS update file, ingest along with catalog
        if wdsfilepath is not None:
            wdsfilepathproc = os.path.normpath(os.path.expandvars(wdsfilepath))
            assert os.path.exists(wdsfilepathproc), (
                "WDS data file not found at %s" % wdsfilepath
            )

            wdsdat = astropy.io.ascii.read(wdsfilepathproc)

            # get HIP numbers of catalog
            HIPnums = np.zeros(len(data), dtype=int)
            for j, name in enumerate(data["hip_name"].astype(str)):
                tmp = re.match(r"HIP\s*(\d+)", name)
                if tmp:
                    HIPnums[j] = int(tmp.groups()[0])

            # find indices of wdsdata in catalog
            inds = np.zeros(len(wdsdat), dtype=int)
            for j, h in enumerate(wdsdat["HIP"].data):
                inds[j] = np.where(HIPnums == h)[0]

            # add attributes to catalog
            wdsatts = ['Close_Sep(")', "Close(M2-M1)", 'Bright_Sep(")', "Bright(M2-M1)"]
            catatts = ["closesep", "closedm", "brightsep", "brightdm"]

            for wdsatt, catatt in zip(wdsatts, catatts):
                tmp = wdsdat[wdsatt].data
                tmp[tmp == "___"] = "nan"
                tmp = tmp.astype(float)
                tmp2 = np.full(len(data), np.nan)
                tmp2[inds] = tmp
                setattr(self, catatt, tmp2)

            self.catalog_atts += catatts

        self.data = data
