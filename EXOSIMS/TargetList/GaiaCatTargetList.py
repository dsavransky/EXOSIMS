import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.TargetList import TargetList
import os
import astropy.io
import inspect
import scipy.interpolate

class GaiaCatTargetList(TargetList):
    """Target list based on Gaia catalog inputs.
    
    Args: 
        \*\*specs: 
            user specified values
    
    """

    def __init__(self, **specs):

        TargetList.__init__(self, **specs)



    def populate_target_list(self,**specs):
        """ This function is actually responsible for populating values from the star 
        catalog (or any other source) into the target list attributes.

        Same as Protoype, but adds Teff to attributes

        Copy directly from star catalog and remove stars with any NaN attributes
        Calculate completeness and max integration time, and generates stellar masses.
        
        """
        
        SC = self.StarCatalog
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        Comp = self.Completeness

        self.catalog_atts.append('Teff')
        self.catalog_atts.append('Gmag')
        self.catalog_atts.append('BPmag')
        self.catalog_atts.append('RPmag')
        self.catalog_atts.append('RAerr')
        self.catalog_atts.append('DECerr')
        self.catalog_atts.append('parxerr')
        self.catalog_atts.append('astrometric_matched_observations')
        self.catalog_atts.append('visibility_periods_used')

        # bring Star Catalog values to top level of Target List
        for att in self.catalog_atts:
            if type(getattr(SC, att)) == np.ma.core.MaskedArray:
                setattr(self, att, getattr(SC, att).filled(fill_value=float('nan')))
            else:
                setattr(self, att, getattr(SC, att))
        
        # number of target stars
        self.nStars = len(self.Name)
        if self.explainFiltering:
            print("%d targets imported from star catalog."%self.nStars)
    
        if self.fillPhotometry:
            self.fillPhotometryVals()

        # filter out nan attribute values from Star Catalog
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering."%self.nStars)

        # populate completeness values
        self.comp0 = Comp.target_completeness(self)
        # populate minimum integration time values
        self.tint0 = OS.calc_minintTime(self)
        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()
        
        # include new attributes to the target list catalog attributes
        self.catalog_atts.append('comp0')
        self.catalog_atts.append('tint0')


    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses
        
        This method calculates stellar mass via interpolation of data from:
        "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester) 
        Version 2017.09.06

        *Function called by reset sim
        
        """

        #Looking for file EEM_dwarf_UBVIJHK_colors_Teff.txt in the TargetList folder
        filename = 'EEM_dwarf_UBVIJHK_colors_Teff.txt'
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        classpath = os.path.normpath(os.path.join(classpath, '..', 
                'TargetList'))
        datapath = os.path.join(classpath, filename)
        assert os.path.isfile(datapath),'Could not locate %s in TargetList directory.'%filename

        data = astropy.io.ascii.read(datapath,fill_values=[('...',np.nan),('....',np.nan),('.....',np.nan)])
        
        Teff_v_Msun = scipy.interpolate.interp1d(data['Teff'].data.astype(float),data['Msun'].data.astype(float),bounds_error=False,fill_value='extrapolate')

        
        # 'approximate' stellar mass
        self.MsTrue = Teff_v_Msun(self.Teff)*u.solMass
        
        # normally distributed 'error'
        err = (np.random.random(len(self.MV))*2. - 1.)*0.07
        self.MsEst = (1. + err)*self.MsTrue
        
        # if additional filters are desired, need self.catalog_atts fully populated
        if not hasattr(self.catalog_atts,'MsEst'):
            self.catalog_atts.append('MsEst')
        if not hasattr(self.catalog_atts,'MsTrue'):
            self.catalog_atts.append('MsTrue')


