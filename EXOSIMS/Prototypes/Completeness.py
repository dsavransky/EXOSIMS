# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u

class Completeness(object):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        specs: 
            user specified values
            
    Attributes:
        dMagLim (float):
            Limiting planet-to-star delta magnitude for completeness
        minComp (float):
            Minimum completeness value for inclusion in target list
        cachedir (str):
            Path to EXOSIMS cache folder
    
    """

    _modtype = 'Completeness'
 
    def __init__(self, dMagLim=25, minComp=0.1, cachedir=None, **specs):
        
        #start the outspec
        self._outspec = {}
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))

        # find the cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 
       
        #if specs contains a completeness_spec then we are going to generate separate instances
        #of planet population and planet physical model for completeness and for the rest of the sim
        if 'completeness_specs' in specs:
            if specs['completeness_specs'] == None:
                specs['completeness_specs'] = {}
                specs['completeness_specs']['modules'] = {}
            if not 'modules' in specs['completeness_specs']:
                specs['completeness_specs']['modules'] = {}
            if not 'PlanetPhysicalModel' in specs['completeness_specs']['modules']:
                specs['completeness_specs']['modules']['PlanetPhysicalModel'] = specs['modules']['PlanetPhysicalModel']
            if not 'PlanetPopulation' in specs['completeness_specs']['modules']:
                specs['completeness_specs']['modules']['PlanetPopulation'] = specs['modules']['PlanetPopulation']
            self.PlanetPopulation = get_module(specs['completeness_specs']['modules']['PlanetPopulation'],'PlanetPopulation')(**specs['completeness_specs'])
            self._outspec['completeness_specs'] = specs.get('completeness_specs')
        else:
            self.PlanetPopulation = get_module(specs['modules']['PlanetPopulation'],'PlanetPopulation')(**specs)

        # copy phyiscal model object up to attribute
        self.PlanetPhysicalModel = self.PlanetPopulation.PlanetPhysicalModel
        
        # loading attributes
        self.dMagLim = float(dMagLim)
        self.minComp = float(minComp)
        
        # populate outspec
        self._outspec['dMagLim'] = self.dMagLim
        self._outspec['minComp'] = self.minComp
        self._outspec['cachedir'] = self.cachedir

    def __str__(self):
        """String representation of Completeness object
        
        When the command 'print' is used on the Completeness object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__:
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Completeness class object attributes'

    def target_completeness(self, TL):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList module):
                TargetList class object
            
        Returns:
            comp0 (float ndarray): 
                Completeness values for each target star
        
        """
        
        comp0 = np.array([0.2]*TL.nStars)
        
        return comp0
        
    def gen_update(self, TL):
        """Generates any information necessary for dynamic completeness 
        calculations (completeness at successive observations of a star in the
        target list)
        
        Args:
            TL (TargetList module):
                TargetList class object
        
        """
        # Prototype does not use precomputed updates, so set these to zeros
        self.updates = np.zeros((TL.nStars, 5))

    def completeness_update(self, TL, sInds, visits, dt):
        """Updates completeness value for stars previously observed
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            visits (integer array):
                Number of visits for each star
            dt (astropy Quantity array):
                Time since previous observation
        
        Returns:
            comp0 (float ndarray):
                Completeness values for each star
        
        """
        # prototype returns the "virgin" completeness value
        comp0 = TL.comp0[sInds]
        
        return comp0

    def revise_updates(self, ind):
        """Keeps completeness update values only for targets remaining in 
        target list during filtering (called from TargetList.filter_target_list)
        
        Args:
            ind (ndarray):
                1D numpy ndarray of indices to keep
        
        """
        
        self.updates = self.updates[ind,:]

    def comp_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None):
        """Calculates completeness values per integration time
        
        Note: Prototype does no calculations and always returns the same value
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
                
        Returns:
            comp (float ndarray):
                Completeness values
        
        """
        
        sInds = np.array(sInds, ndmin=1, copy=False)
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) in [1, len(sInds)], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [1, len(sInds)], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [1, len(sInds)], "fEZ must be constant or have same length as sInds"
        assert len(WA) in [1, len(sInds)], "WA must be constant or have same length as sInds"
        
        return np.array([0.2]*len(sInds))

    def dcomp_dt(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None):
        """Calculates derivative of completeness with respect to integration time
        
        Note: Prototype does no calculations and always returns the same value
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
                
        Returns:
            dcomp (astropy Quantity array):
                Derivative of completeness with respect to integration time (units 1/time)
 
        """
        
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) in [1, len(sInds)], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [1, len(sInds)], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [1, len(sInds)], "fEZ must be constant or have same length as sInds"
        assert len(WA) in [1, len(sInds)], "WA must be constant or have same length as sInds"
        
        return np.array([0.02]*len(sInds))/u.d
