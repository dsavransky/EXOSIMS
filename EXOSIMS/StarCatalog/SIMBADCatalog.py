# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import os
import cPickle as pickle
from scipy.io import loadmat
from astropy.coordinates import SkyCoord
import numpy as np

class SIMBADCatalog(StarCatalog):
    """SIMBAD Catalog class
    
    This class provides the functions to populate the star catalog used in 
    EXOSIMS from the SIMBAD star catalog data."""
    
    def populatepkl(self, filename):
        """Populates the star catalog and returns True if successful
        
        This method populates the star catalog from a pickled dictionary file 
        housed in the StarCatalog directory and returns True if successful.
        
        Args:
            filename (str):
                name of the pickled dictionary file with extension .pkl
        
        Returns:
            bool (bool):
                True if successful, False if not
        
        """
        
        # pickled dictionary stored in filename must contain at least the first 
        # key ('Name') defined
        # 'Name' - list of star names
        # 'Type' - list of star types
        # 'Spec' - list of spectral types
        # 'parx' - list of parallax in milliarcseconds
        # 'Umag' - list of U magnitude
        # 'Bmag' - list of B magnitude
        # 'Vmag' - list of V magnitude
        # 'Rmag' - list of R magnitude
        # 'Imag' - list of I magnitude
        # 'Jmag' - list of J magnitude
        # 'Hmag' - list of H magnitude
        # 'Kmag' - list of K magnitude
        # 'dist' - list of distance in parsecs
        # 'BV' - list of B-V Johnson magnitude
        # 'MV' - list of absolute V magnitude
        # 'BC' - list of bolometric correction
        # 'L' - list of stellar luminosity in Solar luminosities
        # 'radeg' - list of right ascension in degrees
        # 'decdeg' - list of declination in degrees
        # 'pmra'- list of proper motion in right assension in milliarcseconds/year
        # 'pmdec' - list of proper motion in declination in milliarcseconds/year
        # 'rv' - list of radial velocity in kilometers/second
        # 'Binary_Cut' - boolean list where True is companion closer than 10 arcsec
        
        nan = float('nan') # used to fill missing data
        
        if os.path.exists(filename):
            x = pickle.load(open(filename, 'rb'))
            if 'Name' in x:
                self.Name = x['Name']
                # store attributes as list
                atts = self.__dict__.keys() 
                for attr in atts:
                    if attr in x:
                        if attr != 'radeg' or attr != 'decdeg':
                            setattr(self, attr, np.array(x[attr]))
                    else:
                        if attr != 'coords':
                            setattr(self, attr, np.array([nan]*len(self.Name)))
                            print "Warning, %s not in %s list set to 'nan'" % (attr, filename)
                self.coords = SkyCoord(ra=x['radeg'], dec=x['decdeg'], unit='deg')
                success = True
            else:
                print "pickled dictionary file %s must contain key 'Name'" % filename
                success = False
        else:
            print 'Star catalog pickled dictionary file %s not in StarCatalog directory' % filename
            success = False
        
        return success
    
    def SIMBAD_mat2pkl(self, matpath, pklpath):
        """Writes pickled dictionary file from .mat file 
        
        This method takes a .mat star catalog, converts it to a Python 
        dictionary, pickles the dictionary, and stores it in the StarCatalog 
        directory.
        
        Args:
            matpath (str):
                path to .mat file
            pklpath (str):
                pat to .pkl file to be written
        
        Returns:
            bool (bool):
                True if successful, False if not
        
        Stores pickled dictionary file with same name as .mat file (and 
        extension of .pkl) containing lists of required values needed to 
        populate the Star Catalog object in StarCatalog directory.
        
        """
        
        if os.path.exists(matpath):
            # dictionary mapping MATLAB structure fields to required Python 
            # object attribute names
            mat2pkl = {'NAME':'Name', 'TYPE':'Type', 'SPEC':'Spec', 'PARX':'parx',
            'UMAG':'Umag', 'BMAG':'Bmag', 'VMAG':'Vmag', 'RMAG':'Rmag', 'IMAG':'Imag',
            'JMAG':'Jmag', 'HMAG':'Hmag', 'KMAG':'Kmag', 'DIST':'dist', 'BVNEW':'BV',
            'MV':'MV', 'BC':'BC', 'L':'L', 'RADEG':'radeg', 'DECDEG':'decdeg',
            'PMRA':'pmra', 'PMDEC':'pmdec', 'RV':'rv', 'BINARY_CUT':'Binary_Cut'}
            y = {} # empty dictionary to be pickled 
            x = loadmat(matpath, squeeze_me=True, struct_as_record=False)
            x = x['S']
            for field in mat2pkl:
                if field == 'BINARY_CUT':
                    bc = x.BINARY_CUT.tolist()
                    y['Binary_Cut'] = [False]*len(bc)
                    for i in xrange(len(bc)):
                        if bc[i] == 'cut':
                            y['Binary_Cut'][i] = True
                else:
                    y[mat2pkl[field]] = getattr(x, field).tolist()
            # store pickled y dictionary in file
            pickle.dump(y, open(pklpath, 'wb'))
            success = True
        else:
            print '%s does not exist in StarCatalog directory' % matpath
            success = False
            
        return success
