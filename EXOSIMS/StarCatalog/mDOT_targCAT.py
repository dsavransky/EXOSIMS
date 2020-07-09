# -*- coding: utf-8 -*-
import os, inspect
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_downloads_dir
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.table import vstack
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle


class mDOT_targCAT(StarCatalog):
    """
    mDOT targets Catalog class

    This class contains the functions necessary to populate the StarCatalog
    with a given mDOT target list, using SIMBAD and astroquery.

    """

    def __init__(self, targlist="mDOT_targs_NH.txt", **specs):
        """
        Constructor for mDOT_targCAT

        Args:
            targlist (string):
                Path to list of catalog names. Defaults to North Hemisphere
                list as detailed in mDOT concept study, Appendix 9.1,
                table A9.3. File should be readable by the pandas.read_csv
                method and organized such that there is a single column titled
                "Name" under which the names (legible by SIMBAD) are listed.

        """
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        filename = targlist
        catalogpath = os.path.join(classpath, filename)

        self.downloadsdir = get_downloads_dir()
        pklname = filename.replace('.txt', '')
        pklpath = os.path.join(self.downloadsdir, pklname + '.pkl')

        # check if given filename exists as .pkl file already
        if os.path.exists(pklpath):
            print('Loading %s.pkl star catalog from downloads' % pklname)
            infile = open(pklpath, 'rb')
            table = pickle.load(infile)
            infile.close()
        else:
            print('No pickled target list found, astroquery-ing SIMBAD')
            # import target list from txt file
            names = pd.read_csv(catalogpath)

            # set up astroquery class
            customSimbad = Simbad()
            customSimbad.add_votable_fields('ra(d)', 'dec(d)', 'parallax',
                                            'pmra', 'pmdec', 'rv_value',
                                            'sptype', 'diameter', 'flux(U)',
                                            'flux(B)', 'flux(V)', 'flux(R)',
                                            'flux(I)', 'flux(J)', 'flux(K)')
            customSimbad.remove_votable_fields('coordinates')

            # query for each target in list
            results = []
            for i in range(len(names['Name'])):
                name = names['Name'][i]
                result = customSimbad.query_object(name)
                results.append(result)

            # stack table
            table = vstack(results)

            # pickle table to download dir
            outfile = open(pklpath, 'wb')
            pickle.dump(table, outfile)
            outfile.close()
            print('Pickled astroquery results to downloads.')

        StarCatalog.__init__(self, ntargs=len(table), **specs)

        # list of astropy attributes
        self.dist = (1/(1000*table['PLX_VALUE'].data))*u.pc  # calculates dist
        self.parx = self.dist.to('mas', equivalencies=u.parallax())  # parallax
        self.coords = SkyCoord(ra=table['RA_d'].data*u.deg,
                               dec=table['DEC_d'].data*u.deg,
                               distance=self.dist)
        self.pmra = table['PMRA'].data*u.mas/u.yr  # proper motion in RA
        self.pmdec = table['PMDEC'].data*u.mas/u.yr  # proper motion in DEC
        self.rv = table['RV_VALUE'].data*u.km/u.s  # radial velocity

        # list of non-astropy attributes
        self.Name = table['MAIN_ID'].astype(str)  # star names
        self.Spec = table['SP_TYPE'].astype(str)  # spectral types
        self.Umag = table['FLUX_U'].data  # U magnitude
        self.Bmag = table['FLUX_B'].data  # B magnitude
        self.Vmag = table['FLUX_V'].data  # V magnitude
        self.Rmag = table['FLUX_R'].data  # R magnitude
        self.Imag = table['FLUX_I'].data  # I magnitude
        self.Jmag = table['FLUX_J'].data  # J magnitude
        self.Kmag = table['FLUX_K'].data  # K magnitude
        self.BV = self.Bmag - self.Vmag   # B-V Johnson magnitude
        # absolute V magnitude
        self.MV = self.Vmag - 5.*(np.log10(self.dist.to('pc').value) - 1.)
        # self.BC = np.zeros(ntargs)  # bolometric correction
        # self.L = np.ones(ntargs)  # stellar luminosity in ln(SolLum)

        # sanity check to print names of target stars:
        print('mDOT targets gathered for this simulation are:')
        for i in range(len(table['MAIN_ID'])):
            print(self.Name[i], ": ", self.Spec[i])
