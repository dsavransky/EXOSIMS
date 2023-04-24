from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from EXOSIMS.util.getExoplanetArchive import getHWOStars
import pandas
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


class HWOMissionStars(StarCatalog):
    """
    HWO Mission Star List.  Documentation available at:
    https://exoplanetarchive.ipac.caltech.edu/docs/2645_NASA_ExEP_Target_List_HWO_Documentation_2023.pdf    # noqa: E501

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk. Defaults False.
    """

    def __init__(self, forceNew=False, **specs):

        data = getHWOStars(forceNew=forceNew)

        StarCatalog.__init__(self, ntargs=len(data), **specs)
        self._outspec["forceNew"] = forceNew

        # assemble names
        self.Name = np.array(
            [
                f"{n}" if pandas.isna(cn) else f"{n} {cn}"
                for n, cn in zip(data["hip_name"].values, data["hip_compname"].values)
            ]
        )

        atts_mapping = {
            "Spec": ("st_spectype", None),
            "parx": ("sy_plx", u.mas),
            "dist": ("sy_dist", u.pc),
            "Vmag": ("sy_vmag", None),
            "BV": ("sy_bvmag", None),
            "Rmag": ("sy_rcmag", None),
            "Teff": ("st_teff", u.K),
            "diameter": ("st_diam", u.mas),
            "mass": ("st_mass", u.solMass),
            "metallicity": ("st_met", None),
            "logg": ("st_logg", None),
        }

        for att in atts_mapping:
            if atts_mapping[att][1] is None:
                setattr(self, att, data[atts_mapping[att][0]].values)
            else:
                setattr(
                    self, att, data[atts_mapping[att][0]].values * atts_mapping[att][1]
                )

        # grab luminosity and fill in remaining values
        self.L = 10 ** (data["st_lum"].values)
        self.BC = -2.5 * self.L - 26.832 - self.Vmag
        self.MV = self.Vmag - 5 * (np.log10(self.dist.to("pc").value) - 1)

        self.coords = SkyCoord(
            ra=data["ra"].values * u.deg,
            dec=data["dec"].values * u.deg,
            distance=self.dist,
        )

        self.hasKnownPlanet = data["sy_planets_flag"].values == "Y"
        self.Binary_Cut = data["wds_sep"].values < 10

        self.data = data

        # add available catalog attributes
        self.catalog_atts += ["Teff", "diameter", "mass", "metallicity", "logg"]
