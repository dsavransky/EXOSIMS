from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import pandas
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


class plandbcat(StarCatalog):
    """
    Imaging Mission Database Star Catalog.

    Args:
        star_data_path (str):
            Full path to pickle file with stellar data

    """

    def __init__(self, star_data_path=None, **specs):

        assert star_data_path is not None, "star_data_path must be set."

        stdata = pandas.read_pickle(star_data_path)

        StarCatalog.__init__(self, ntargs=len(stdata), **specs)

        atts_mapping = {
            "Name": ("st_name", None),
            "Spec": ("spectype", None),
            "parx": ("sy_plx", u.mas),
            "dist": ("sy_dist", u.pc),
            "pmra": ("sy_pmra", u.mas / u.year),
            "pmdec": ("sy_pmdec", u.mas / u.year),
            "rv": ("st_radv", u.km / u.s),
            "Umag": ("sy_umag", None),
            "Bmag": ("sy_bmag", None),
            "Vmag": ("sy_vmag", None),
            "Rmag": ("sy_rmag", None),
            "Imag": ("sy_imag", None),
            "Jmag": ("sy_jmag", None),
            "Hmag": ("sy_hmag", None),
            "Kmag": ("sy_kmag", None),
            "Teff": ("teff", u.K),
            "mass": ("mass", u.solMass),
            "metallicity": ("met", None),
            "logg": ("logg", None),
        }

        for att in atts_mapping:
            if atts_mapping[att][1] is None:
                setattr(self, att, stdata[atts_mapping[att][0]].values)
            else:
                setattr(
                    self,
                    att,
                    stdata[atts_mapping[att][0]].values * atts_mapping[att][1],
                )

        # replace missing specttypes with ''
        self.Spec[pandas.isna(self.Spec)] = ""

        # Ensure that Name and Spec are both strictly strings
        self.Name = self.Name.astype(str)
        self.Spec = self.Spec.astype(str)

        # Replace missing pm and rv vals with zeros
        # TODO: query replacement vals
        self.pmra[pandas.isna(self.pmra)] = 0 * (u.mas / u.year)
        self.pmdec[pandas.isna(self.pmdec)] = 0 * (u.mas / u.year)
        self.rv[pandas.isna(self.rv)] = 0 * (u.km / u.s)

        # weirdly missing some parallaxes, so just use distances for those
        assert np.all(~pandas.isna(self.dist))
        inds = pandas.isna(self.parx)
        self.parx[inds] = self.dist[inds].to(u.marcsec, equivalencies=u.parallax())

        # grab luminosity and fill in remaining values
        self.BV = self.Bmag - self.Vmag
        self.L = 10 ** (stdata["lum"].values)
        self.BC = -2.5 * self.L - 26.832 - self.Vmag
        self.MV = self.Vmag - 5 * (np.log10(self.dist.to("pc").value) - 1)

        self.coords = SkyCoord(
            ra=stdata["ra"].values * u.deg,
            dec=stdata["dec"].values * u.deg,
            distance=self.dist,
        )

        self.hasKnownPlanet = np.ones(len(stdata), dtype=bool)
        self.Binary_Cut = stdata["sy_snum"].values > 1

        self.data = stdata

        # add available catalog attributes
        self.catalog_atts += ["Teff", "mass", "metallicity", "logg"]
