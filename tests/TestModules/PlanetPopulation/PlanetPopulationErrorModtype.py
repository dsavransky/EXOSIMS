# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation


class PlanetPopulationErrorModtype(PlanetPopulation):
    """PlanetPopulation non-prototype class template (error-containing).

    This class exists to test import of non-EXOSIMS modules using
    the get_module interface.  It presently takes no arguments
    and does nothing besides initialize itself.

    It contains an erroroneous _modtype attribute.

    Args:
        \*\*specs:
            user specified values

    Attributes:
        (none)
    """

    # Note: wrong module type has been given
    _modtype = "TimeKeeping"
    _outspec = {}

    def __init__(self, **specs):
        pass
