from EXOSIMS.Prototypes.TargetList import TargetList
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


class PlandbTargetList(TargetList):
    """Target List based on the planet population from the plandb database.
    Should be used with the Plandb family of modules (PlandbPlanets, PlandbUniverse)

    Args:
        **specs:
            user specified values

    """

    def __init__(self, **specs):
        # Mapping attribute name from the data columns
        # data columns are loaded from the pickle file in the...
        # PlanetPopulation module (PlandbPlanets)
        self.atts_mapping = {
            "Name": "hostname",
            "Spec": "st_spectype",
            "parx": "sy_plx",
            "Umag": "sy_umag",
            "Bmag": "sy_bmag",
            "Vmag": "sy_vmag",
            "Rmag": "sy_rmag",
            "Imag": "sy_imag",
            "Jmag": "sy_jmag",
            "Hmag": "sy_hmag",
            "Kmag": "sy_kmag",
            "dist": "sy_dist",
            "L": "st_lum",
            "pmra": "sy_pmra",
            "pmdec": "sy_pmdec",
            "rv": "st_radv",
        }

        TargetList.__init__(self, **specs)

    def set_catalog_attributes(self):
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
            "L",
            "pmra",
            "pmdec",
            "rv",
            "coords",
            "BC",
            "BV",
            "MV",
            "Binary_Cut",
        ]

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
        PPop = self.PlanetPopulation
        Comp = self.Completeness
        OS = self.OpticalSystem

        tmp = PPop.allplanetdata

        # filtering out targets (based on Working Angle)
        dist = tmp["sy_dist"].values * u.pc

        mask = (
            ~tmp["sy_dist"].isna()
            & (np.arctan(PPop.sma * (1 + PPop.eccen) / dist) > OS.IWA)
            & (np.arctan(PPop.sma * (1 - PPop.eccen) / dist) < OS.OWA)
        )
        tmp = tmp[mask]

        # filtering out redundant targets
        targetStars = np.unique(tmp["hostname"].values, return_index=True)[1]
        tmp = tmp.iloc[targetStars]

        # filter missing Vmag and BV, for integration time calculation
        tmp = tmp[~tmp["sy_vmag"].isna()]
        BV = tmp["sy_bmag"] - tmp["sy_vmag"]
        tmp = tmp[~BV.isna()]

        # number of Targets
        self.nStars = len(tmp)
        assert self.nStars, "Target List is empty: nStars = %r" % self.nStars

        # assigning attributes to the target list
        for att in self.atts_mapping:
            att_val = tmp[self.atts_mapping[att]].values
            mask = tmp[self.atts_mapping[att]].isna()
            if att_val.dtype == "float64":
                setattr(self, att, att_val)
                setattr(self, att, getattr(self, att).astype(float))
                getattr(self, att)[mask] = np.nanmedian(getattr(self, att))

            else:
                if (att == "Name") or (att == "Spec"):
                    setattr(self, att, att_val.astype(str))
                else:
                    setattr(self, att, att_val)

        # astropy units
        self.parx = self.parx * u.mas
        self.dist = self.dist * u.pc
        self.pmra = self.pmra * u.mas / u.yr
        self.pmdec = self.pmdec * u.mas / u.yr
        self.rv = self.rv * u.km / u.s

        self.BC = -2.5 * self.L - 26.832 - self.Vmag
        self.L = 10.0**self.L
        self.BV = self.Bmag - self.Vmag
        self.MV = self.Vmag - 5 * (np.log10(self.dist.to("pc").value) - 1)
        self.coords = SkyCoord(
            ra=tmp["ra"].values * u.deg,
            dec=tmp["dec"].values * u.deg,
            distance=self.dist,
        )
        self.Binary_Cut = np.zeros(self.nStars, dtype=bool)

        # populate completeness values
        self.comp0 = Comp.target_completeness(self)

        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()

        # include new attributes to the target list catalog attributes
        self.catalog_atts.append("comp0")

    def filter_target_list(self, **specs):
        """dummy function (for further filtering)"""

        pass
