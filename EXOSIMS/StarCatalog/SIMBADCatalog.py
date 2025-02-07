# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import os
import pickle
from scipy.io import loadmat
from astropy.coordinates import SkyCoord
import numpy as np


class SIMBADCatalog(StarCatalog):
    """SIMBAD Catalog class

    This class provides the functions to populate the star catalog used in
    EXOSIMS from the SIMBAD star catalog data.

    """

    def populatepkl(self, pklpath, **specs):
        """Populates the star catalog and returns True if successful

        This method populates the star catalog from a pickled dictionary file
        housed in the StarCatalog directory and returns True if successful.

        Args:
            pklpath (string):
                path to the pickled dictionary file with extension .pkl
            **specs:
                :ref:`sec:inputspec`

        Returns:
            bool (boolean):
                True if successful, False if not

        """

        if os.path.exists(pklpath):
            try:
                with open(pklpath, "rb") as ff:
                    x = pickle.load(ff)
            except UnicodeDecodeError:
                with open(pklpath, "rb") as ff:
                    x = pickle.load(ff, encoding="latin1")
            if "Name" in x:
                ntargs = len(x["Name"])
                StarCatalog.__init__(self, ntargs=ntargs, **specs)

                for att in x:
                    # list of astropy attributes
                    if att in ("dist", "parx", "pmra", "pmdec", "rv"):
                        unit = getattr(self, att).unit
                        setattr(self, att, np.array(x[att]) * unit)
                    # list of non-astropy attributes
                    elif att in self.__dict__:
                        setattr(self, att, np.array(x[att]))
                # astropy SkyCoord object
                self.coords = SkyCoord(
                    x["radeg"], x["decdeg"], x["dist"], unit="deg,deg,pc"
                )

                success = True
            else:
                self.vprint(
                    "pickled dictionary file %s must contain key 'Name'" % pklpath
                )
                success = False
        else:
            self.vprint(
                "Star catalog pickled dictionary file %s not in StarCatalog directory"
                % pklpath
            )
            success = False

        return success

    def SIMBAD_mat2pkl(self, matpath, pklpath):
        """Writes pickled dictionary file from .mat file

        This method takes a .mat star catalog, converts it to a Python
        dictionary, pickles the dictionary, and stores it in the StarCatalog
        directory.

        Args:
            matpath (string):
                path to .mat file
            pklpath (str):
                pat to .pkl file to be written

        Returns:
            bool (boolean):
                True if successful, False if not

        Stores pickled dictionary file with same name as .mat file (and
        extension of .pkl) containing lists of required values needed to
        populate the Star Catalog object in StarCatalog directory.

        """

        if os.path.exists(matpath):
            # dictionary mapping MATLAB structure fields to required Python
            # object attribute names
            mat2pkl = {
                "NAME": "Name",
                "TYPE": "Type",
                "SPEC": "Spec",
                "PARX": "parx",
                "UMAG": "Umag",
                "BMAG": "Bmag",
                "VMAG": "Vmag",
                "RMAG": "Rmag",
                "IMAG": "Imag",
                "JMAG": "Jmag",
                "HMAG": "Hmag",
                "KMAG": "Kmag",
                "DIST": "dist",
                "BVNEW": "BV",
                "MV": "MV",
                "BC": "BC",
                "L": "L",
                "RADEG": "radeg",
                "DECDEG": "decdeg",
                "PMRA": "pmra",
                "PMDEC": "pmdec",
                "RV": "rv",
                "BINARY_CUT": "Binary_Cut",
            }
            y = {}  # empty dictionary to be pickled
            x = loadmat(matpath, squeeze_me=True, struct_as_record=False)
            x = x["S"]
            for field in mat2pkl:
                if field == "BINARY_CUT":
                    bc = x.BINARY_CUT.tolist()
                    y["Binary_Cut"] = [False] * len(bc)
                    for i in range(len(bc)):
                        if isinstance(bc[i], str) and bc[i] == "cut":
                            y["Binary_Cut"][i] = True
                else:
                    y[mat2pkl[field]] = getattr(x, field).tolist()
            # store pickled y dictionary in file
            with open(pklpath, "wb") as f:
                pickle.dump(y, f)
            success = True
        else:
            self.vprint("%s does not exist in StarCatalog directory" % matpath)
            success = False

        return success
