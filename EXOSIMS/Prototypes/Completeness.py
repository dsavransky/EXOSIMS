# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
import numpy as np

class Completeness(object):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    """

    _modtype = 'Completeness'
    _outspec = {}
    
    def __init__(self, **specs):
        
        # load the vprint funtion (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # import Planet Population and Physical Model class objects
        Pop = get_module(specs['modules']['PlanetPopulation'],'PlanetPopulation')(**specs)
        self.PlanetPopulation = Pop
        self.PlanetPhysicalModel = Pop.PlanetPhysicalModel

    def __str__(self):
        """String representation of Completeness object
        
        When the command 'print' is used on the Completeness object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
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

    def comp_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode):
        """Calculates completeness for integration time

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
            comp (array):
                Completeness values
        
        """
        
        sInds = np.array(sInds, ndmin=1, copy=False)
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(intTimes) == len(fZ) or len(fZ) == 1, "fZ must be constant or have same length as intTimes"
        assert len(intTimes) == len(fEZ) or len(fEZ) == 1, "fEZ must be constant or have same length as intTimes"
        assert len(WA) == 1, "WA must be constant"

        return np.array([0.2]*len(intTimes))
        

    def dcomp_dt(self, intTimes, TL, sInds, fZ, fEZ, WA, mode):
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
            dcomp (array):
                Derivative of completeness with respect to integration time
        
        """

        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(intTimes) == len(fZ) or len(fZ) == 1, "fZ must be constant or have same length as intTimes"
        assert len(intTimes) == len(fEZ) or len(fEZ) == 1, "fEZ must be constant or have same length as intTimes"
        assert len(WA) == 1, "WA must be constant"

        return np.array([0.02]*len(intTimes))

