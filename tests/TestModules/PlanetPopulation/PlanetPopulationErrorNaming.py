# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation

# Note: wrong class name has been given within the module (they should match)
class PlanetPopulationErrorNamingBadName(PlanetPopulation):
    """PlanetPopulation non-prototype class template (error-containing).

    This class exists to test import of non-EXOSIMS modules using
    the get_module interface.  It presently takes no arguments
    and does nothing besides initialize itself.

    The class has an erroneous name: the class implemented here has a different
    name from the module.

    Args:
        \*\*specs:
            user specified values

    Attributes:
        (none)
    """

    _modtype = "PlanetPopulation"
    _outspec = {}

    def __init__(self, **specs):
        pass
