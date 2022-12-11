from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import astropy.constants as const

class multiSS(SurveySimulation):

    def __init__(self,coef):

        self.coef = coef
        
