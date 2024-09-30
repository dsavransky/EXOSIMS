from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import pandas
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from EXOSIMS.util.get_dirs import get_downloads_dir
import os
from urllib.request import urlretrieve
import tarfile


class HPIC(StarCatalog):
    """Habitable Worlds Observatory Preliminary Input Catalog
    https://exoplanetarchive.ipac.caltech.edu/docs/MissionStellar.html

    Args:
        **specs:
            :ref:`sec:inputspec`

    """

    def __init__(self, **specs):

        downloadsdir = get_downloads_dir()
        localfile = os.path.join(downloadsdir, "full_HPIC.txt")
        if not os.path.exists(localfile):
            url = r"https://exoplanetarchive.ipac.caltech.edu/data/Contributed/MissionStars/HPICv1.0.tgz"  # noqa: E501
            tgzpath = os.path.join(downloadsdir, url.split("/")[-1])

            # check if file might have been downloaded but not unarchived:
            if not os.path.exists(tgzpath):
                print("Downloading HPIC")
                _ = urlretrieve(url, tgzpath)
                assert os.path.exists(tgzpath), "HPIC download failed."

            with tarfile.open(tgzpath, "r") as f:
                f.extractall(path=downloadsdir)

            assert os.path.exists(localfile), "Could not find HPIC file on disk."

        data = pandas.read_csv(localfile, delimiter="|")
        StarCatalog.__init__(self, ntargs=len(data), **specs)

        atts_mapping = {
            "Name": ("star_name", None),
            "Spec": ("st_spectype", None),
            "parx": ("sy_plx", u.mas),
            "dist": ("sy_dist", u.pc),
            "Umag": ("sy_ujmag", None),
            "Bmag": ("sy_bmag", None),
            "Vmag": ("sy_vmag", None),
            "Rmag": ("sy_rcmag", None),
            "Imag": ("sy_icmag", None),
            "Jmag": ("sy_jmag", None),
            "Hmag": ("sy_hmag", None),
            "Kmag": ("sy_kmag", None),
            "Tmag": ("sy_tmag", None),
            "Gmag": ("sy_gaiamag", None),
            "Bpmag": ("sy_bpmag", None),
            "Rpmag": ("sy_rpmag", None),
            "Teff": ("st_teff", u.K),
            "mass": ("st_mass", u.solMass),
            "age": ("st_age", u.Gyr),
            "metallicity": ("st_met", None),
            "logg": ("st_logg", None),
        }

        for att in atts_mapping:
            if atts_mapping[att][1] is None:
                tmp = data[atts_mapping[att][0]].values
                if tmp.dtype.name == "object":
                    tmp[pandas.isna(tmp)] = ""
                    tmp = tmp.astype(str)
                setattr(self, att, tmp)
            else:
                setattr(
                    self, att, data[atts_mapping[att][0]].values * atts_mapping[att][1]
                )

        # fill in remaining values
        self.BV = self.Bmag - self.Vmag
        self.L = 10 ** (data["st_lum"].values)
        self.BC = -2.5 * self.L - 26.832 - self.Vmag
        self.MV = self.Vmag - 5 * (np.log10(self.dist.to("pc").value) - 1)
        self.diameter = np.arctan(2 * data["st_rad"].values * u.Rsun / (self.dist)).to(
            u.mas
        )

        self.coords = SkyCoord(
            ra=data["ra"].values * u.deg,
            dec=data["dec"].values * u.deg,
            distance=self.dist,
        )

        self.hasKnownPlanet = data["sy_planets_flag"].values == 1
        self.Binary_Cut = data["wds_sep"].values < 10

        self.data = data

        # add available catalog attributes
        self.catalog_atts += [
            "Tmag",
            "Gmag",
            "Bpmag",
            "Rpmag",
            "Teff",
            "diameter",
            "mass",
            "metallicity",
            "logg",
        ]
