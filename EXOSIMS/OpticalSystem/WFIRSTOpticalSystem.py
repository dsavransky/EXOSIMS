# -*- coding: utf-8 -*-
from EXOSIMS.OpticalSystem.KasdinBraems import KasdinBraems
import astropy.units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class WFIRSTOpticalSystem(KasdinBraems):
    """WFIRST OpticalSystem class
    
    This class contains all variables and methods specific to the WFIRST
    optical system needed to perform OpticalSystem Definition Module calculations
    in exoplanet mission simulation.
        
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        KasdinBraems.__init__(self, **specs)
        
        self.Imager = None
        self.Spectro = None
        self.ImagerSyst = None
        self.SpectroSyst = None
        
        for inst in self.scienceInstruments:
            if 'imag' in inst['type'].lower():
                self.Imager = inst
            if 'spec' in inst['type'].lower():
                self.Spectro = inst
        for syst in self.starlightSuppressionSystems:
            if 'hlc' in syst['type'].lower():
                self.ImagerSyst = syst
            if 'spc' in syst['type'].lower():
                self.SpectroSyst = syst
        
        assert self.Imager, "No imager defined."
        assert self.Spectro, "No spectrograph defined."
        assert self.ImagerSyst, "No hybrid Lyot coronagraph defined."
        assert self.SpectroSyst, "No shaped pupil coronagraph defined."
