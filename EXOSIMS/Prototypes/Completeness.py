# -*- coding: utf-8 -*-
import numpy as np
from EXOSIMS.util.get_module import get_module

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
        # import PlanetPopulation class
        Pop = get_module(specs['modules']['PlanetPopulation'],'PlanetPopulation')(**specs)
        self.PlanetPopulation = Pop # planet population object class
        self.PlanetPhysicalModel = Pop.PlanetPhysicalModel

    def __str__(self):
        """String representation of Completeness object
        
        When the command 'print' is used on the Completeness object, this 
        method will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Completeness class object attributes'

    def target_completeness(self, TL):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList): 
                TargetList class object
            
        Returns:
            comp0 (ndarray): 
                1D numpy array of completeness values for each target star
        
        """
        
        comp0 = np.array([0.2]*TL.nStars)
        
        return comp0
        
    def gen_update(self, TL):
        """Generates any information necessary for dynamic completeness 
        calculations (completeness at successive observations of a star in the
        target list)
        
        Args:
            TL (TargetList):
                TargetList module
        
        """
        pass

    def completeness_update(self, sInd, TL, obsbegin, obsend, nexttime):
        """Updates completeness value for stars previously observed
        
        Args:
            sInd (int):
                index of star just observed
            TL (TargetList):
                TargetList module
            obsbegin (Quantity):
                time of observation begin
            obsend (Quantity):
                time of observation end
            nexttime (Quantity):
                time of next observational period
        
        Returns:
            comp0 (ndarray):
                1D numpy ndarray of completeness values for each star in the 
                target list
        
        """
        # prototype returns the "virgin" completeness value
        return TL.comp0