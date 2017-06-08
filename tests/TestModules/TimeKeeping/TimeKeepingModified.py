# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.TimeKeeping import TimeKeeping

class TimeKeepingModified(TimeKeeping):
    """TimeKeeping non-prototype class template.
    
    This class exists to test import of non-EXOSIMS modules using
    the get_module interface.  It presently takes no arguments
    and does nothing besides initialize itself.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        (none)
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, **specs):
        pass

