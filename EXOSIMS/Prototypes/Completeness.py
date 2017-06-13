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
        # import Planet Population and Physical Model class objects
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
            dt (astropy Quantity):
                Time since initial completeness
        
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