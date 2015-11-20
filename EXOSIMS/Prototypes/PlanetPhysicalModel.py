# -*- coding: utf-8 -*-
class PlanetPhysicalModel(object):
    """Planet Physical Model class template
    
    This class contains all variables and functions necessary to perform 
    Planet Physical Model Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    """

    _modtype = 'PlanetPhysicalModel'
    
    def __init__(self, **specs):
        
        # default values
        # replace default values with user specification values if any
        # initialize values updated by functions
        # set values derived from quantities above
        pass
    
    def __str__(self):
        """String representation of Planet Physical Model object
        
        When the command 'print' is used on the Planet Physical Model object, 
        this method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Planet Physical Model class object attributes'
