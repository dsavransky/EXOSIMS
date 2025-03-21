# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.TargetList import TargetList
import warnings


class KnownRVPlanetsTargetList(TargetList):
    """Target list based on population of known RV planets from IPAC.
    Intended for use with the KnownRVPlanets family of modules.

    Args:
        **specs:
            :ref:`sec:inputspec`

    .. warning:
        fillPhotometry and getKnownPlanets are dissallowed inputs and will be set
        to False if present in the input specification.

    """

    def __init__(self, **specs):
        # define mapping between attributes we need and the IPAC data
        # table loaded in the Planet Population module
        self.atts_mapping = {
            "Name": "pl_hostname",
            "Spec": "st_spstr",
            "parx": "st_plx",
            "Umag": "st_uj",
            "Bmag": "st_bj",
            "Vmag": "st_vj",
            "Rmag": "st_rc",
            "Imag": "st_ic",
            "Jmag": "st_j",
            "Hmag": "st_h",
            "Kmag": "st_k",
            "dist": "st_dist",
            "BV": "st_bmvj",
            "L": "st_lum",  # log10(solLum)
            "pmra": "st_pmra",  # mas/year
            "pmdec": "st_pmdec",  # mas/year
            "rv": "st_radv",
        }

        # Enforce required planet population (KnownRVPlanets)
        assert (
            specs["modules"]["PlanetPopulation"] == "KnownRVPlanets"
        ), "KnownRVPlanetsTargetList must use KnownRVPlanets population"

        # Override any bad input attributes
        attributes_that_must_be_false = ["getKnownPlanets"]
        for att in attributes_that_must_be_false:
            if (att in specs) and (specs[att] is True):
                warnings.warn(
                    (
                        f"KnownRVPlanetsTargetList does not allow {att} "
                        "input to be True. Setting to False."
                    )
                )
                specs[att] = False

        TargetList.__init__(self, **specs)

    def set_catalog_attributes(self):
        """Hepler method that sets possible and required catalog attributes.

        Sets attributes:
            catalog_atts (list):
                Attributes to try to copy from star catalog.  Missing ones will be
                ignored and removed from this list.
            required_catalog_atts(list):
                Attributes that cannot be missing or nan.

        """

        # list of possible Star Catalog attributes
        self.catalog_atts = [
            "Name",
            "Spec",
            "parx",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "dist",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
            "hasKnownPlanet",
        ]

        # required catalog attributes
        self.required_catalog_atts = [
            "Name",
            "Vmag",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "dist",
        ]

    def populate_target_list(self, **specs):
        """This function is responsible for populating values from the star
        catalog into the target list attributes and enforcing attribute requirements.


        Args:
            **specs:
                :ref:`sec:inputspec`

        """

        PPop = self.PlanetPopulation
        OS = self.OpticalSystem

        tmp = PPop.allplanetdata[:]
        # filter out targets with planets outside of WA range
        dist = tmp["st_dist"].filled() * u.pc
        mask = (
            ~tmp["st_dist"].mask
            & (np.arctan(PPop.sma * (1 + PPop.eccen) / dist) > OS.IWA)
            & (np.arctan(PPop.sma * (1 - PPop.eccen) / dist) < OS.OWA)
        )
        tmp = tmp[mask]
        # filter out redundant targets
        tmp = tmp[np.unique(tmp["pl_hostname"].data, return_index=True)[1]]

        # filter missing Vmag and BV, for integration time calculation
        tmp = tmp[~tmp["st_vj"].mask]
        tmp = tmp[~tmp["st_bmvj"].mask]

        self.nStars = len(tmp)
        assert self.nStars, "Target list is empty: nStars = %r" % self.nStars

        for att in self.atts_mapping:
            ma = tmp[self.atts_mapping[att]]
            if isinstance(ma.fill_value, np.float64):
                setattr(self, att, ma.filled(np.nan))
            else:
                if (att == "Name") or (att == "Spec"):
                    setattr(self, att, ma.data.astype(str))
                else:
                    setattr(self, att, ma.data)
        # astropy units
        self.parx = self.parx * u.mas
        self.dist = self.dist * u.pc
        self.pmra = self.pmra * u.mas / u.yr
        self.pmdec = self.pmdec * u.mas / u.yr
        self.rv = self.rv * u.km / u.s

        self.BC = -2.5 * self.L - 26.832 - self.Vmag
        self.L = 10.0**self.L
        self.MV = self.Vmag - 5 * (np.log10(self.dist.to("pc").value) - 1)
        self.coords = SkyCoord(
            ra=tmp["ra"] * u.deg, dec=tmp["dec"] * u.deg, distance=self.dist
        )
        self.Binary_Cut = np.zeros(self.nStars, dtype=bool)
        self.hasKnownPlanet = np.ones(self.nStars, dtype=bool)

    def filter_target_list(self, filters):
        """Filtering is done as part of populating the table, so this
        method is overloaded to do nothing.
        """

        pass
