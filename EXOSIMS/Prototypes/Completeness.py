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
        
        # initialize number of visits per star
        self.visits = np.array([0]*TL.nStars)

    def completeness_update(self, TL, sInds, dt):
        """Updates completeness value for stars previously observed
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            dt (astropy Quantity):
                Time since initial completeness
        
        Returns:
            comp0 (float ndarray):
                Completeness values for each star
        
        """
        # prototype returns the "virgin" completeness value
        comp0 = TL.comp0[sInds]
        
        return comp0
