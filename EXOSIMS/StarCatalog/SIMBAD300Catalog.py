# -*- coding: utf-8 -*-
from EXOSIMS.StarCatalog.SIMBADCatalog import SIMBADCatalog
from EXOSIMS.util.get_dirs import get_cache_dir
import os
import inspect


class SIMBAD300Catalog(SIMBADCatalog):
    """SIMBAD300 Catalog class

    This class populates the star catalog used in EXOSIMS from the SIMBAD300
    catalog.

    """

    def __init__(self, cachedir=None, **specs):
        self.cachedir = get_cache_dir(cachedir)
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        filename = "SIMBAD300"
        pklpath = os.path.join(self.cachedir, filename + ".pkl")
        matpath = os.path.join(classpath, filename + ".mat")

        # check if given filename exists as .pkl file already
        if os.path.exists(pklpath):
            self.populatepkl(pklpath, **specs)
            self.vprint("Loaded %s.pkl star catalog" % filename)
        # check if given filename exists as a .mat file but not .pkl file
        elif os.path.exists(matpath):
            self.SIMBAD_mat2pkl(matpath, pklpath)
            self.populatepkl(pklpath, **specs)
            self.vprint("Loaded %s.mat star catalog" % filename)
        # otherwise print error
        else:
            self.vprint("Could not load SIMBAD300 star catalog")
