# -*- coding: utf-8 -*-
from EXOSIMS.StarCatalog.SIMBADCatalog import SIMBADCatalog
import os, inspect

class SIMBAD300Catalog(SIMBADCatalog):
    """SIMBAD300 Catalog class
    
    This class populates the star catalog used in EXOSIMS from the SIMBAD300 
    catalog."""
    
    def __init__(self, **specs):
       
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        filename = 'SIMBAD300'
        SIMBADCatalog.__init__(self)
        pklpath = os.path.join(classpath, filename+'.pkl')
        matpath = os.path.join(classpath, filename+'.mat')
        
        # check if given filename exists as .pkl file already
        if os.path.exists(pklpath):
            self.populatepkl(pklpath)
            print 'Loaded %s star catalog' % pklpath
        # check if given filename exists as a .mat file but not .pkl file
        elif os.path.exists(matpath):
            self.SIMBAD_mat2pkl(matpath, pklpath)
            self.populatepkl(pklpath)
            print 'Loaded %s star catalog' % filename
        # otherwise print error
        else:
            print 'Could not load SIMBAD300 star catalog'
