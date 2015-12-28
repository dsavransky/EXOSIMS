# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import EXOSIMS.util.statsFun as statsFun
import numpy as np
import astropy.units as u
import astropy.constants as const


class EarthTwinHabZone1(PlanetPopulation):
    """
    Population of Earth twins (1 R_Earth, 1 M_Eearth, 1 p_Earth)
    On circular Habitable zone orbits (0.7 to 1.5 AU)
    """
    def __init__(self, arange=[0.7, 1.5], erange=[0,0],\
            Rrange=[1,1],Mprange=[1,1],prange=[0.367,0.367],**specs):

        PlanetPopulation.__init__(self, arange=arange, erange=erange,\
                Rrange=Rrange,Mprange=Mprange,prange=prange,**specs)

