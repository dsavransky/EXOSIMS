import numpy as np
import astropy.units as u
from EXOSIMS.Prototypes.TargetList import TargetList
import scipy.interpolate


class GaiaCatTargetList(TargetList):
    """Target list based on Gaia catalog inputs.

    Args:
        **specs:
            user specified values

    """

    def __init__(self, **specs):

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

        # call base method for the default set
        TargetList.set_catalog_attributes(self)

        # now add Gaia-specific ones
        self.catalog_atts.append("Teff")
        self.catalog_atts.append("Gmag")
        self.catalog_atts.append("BPmag")
        self.catalog_atts.append("RPmag")
        self.catalog_atts.append("RAerr")
        self.catalog_atts.append("DECerr")
        self.catalog_atts.append("parxerr")
        self.catalog_atts.append("astrometric_matched_observations")
        self.catalog_atts.append("visibility_periods_used")

        self.required_catalog_atts.append("Teff")

    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses

        This method calculates stellar mass via interpolation of data from:
        "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester)

        For more details see MeanStars documentation.

        Function called by reset sim
        """

        data = self.ms.data

        Teff_v_Msun = scipy.interpolate.interp1d(
            data["Teff"].data.astype(float),
            data["Msun"].data.astype(float),
            bounds_error=False,
            fill_value="extrapolate",
        )

        # 'True' stellar mass
        self.MsTrue = Teff_v_Msun(self.Teff) * u.solMass

        # normally distributed error to generate 'approximate' stellar mass
        err = (np.random.random(len(self.MV)) * 2.0 - 1.0) * 0.07
        self.MsEst = (1.0 + err) * self.MsTrue

        # if additional filters are desired, need self.catalog_atts fully populated
        if not hasattr(self.catalog_atts, "MsEst"):
            self.catalog_atts.append("MsEst")
        if not hasattr(self.catalog_atts, "MsTrue"):
            self.catalog_atts.append("MsTrue")
