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
    
    Attributes:
        minComp (float): 
            minimum completeness level for inclusion in target list
        
    """

    _modtype = 'Completeness'
    _outspec = {}
    
    def __init__(self, minComp=0.1, Nplanets=1e8, **specs):
        # import PlanetPopulation class
        Pop = get_module(specs['modules']['PlanetPopulation'], 'PlanetPopulation')
        self.PlanetPopulation = Pop(**specs) # planet population object class
        self.PlanetPhysicalModel = self.PlanetPopulation.PlanetPhysicalModel
       
        self.minComp = float(minComp)
        self.Nplanets = Nplanets # total number of planets for completeness calcs
        self._outspec['minComp'] = self.minComp
        self._outspec['Nplanets'] = self.Nplanets

    
    def __str__(self):
        """String representation of Completeness object
        
        When the command 'print' is used on the Completeness object, this 
        method will return the values contained in the object
        
        """

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Completeness class object attributes'
        
    def target_completeness(self, targlist):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            targlist (TargetList): 
                TargetList class object
            
        Returns:
            comp0 (ndarray): 
                1D numpy array of completeness values for each target star
        
        """
        
        comp0 = np.array([0.2]*len(targlist.Name))
        
        return comp0
        
    def completeness_update(self, s_ind, targlist, obsbegin, obsend, nexttime):
        """Updates completeness value for stars previously observed
        
        Args:
            s_ind (int):
                index of star just observed
            targlist (TargetList):
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
        comp0 = targlist.comp0
        
        return comp0
