# -*- coding: utf-8 -*-
from EXOSIMS.OpticalSystem.OptSys_Kasdin import OptSys_Kasdin
from astropy import units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class OptSys_Nemati(OptSys_Kasdin):
    """WFIRST Optical System class
    
    This class contains all variables and methods specific to the WFIRST
    optical system needed to perform Optical System Module calculations
    in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        OptSys_Kasdin.__init__(self, **specs)

    def calc_intTime(self, targlist, sInd, I, dMag, WA):
        """Finds integration time for a specific target system,
        based on Nemati 2014 (SPIE).
        
        Args:
            targlist:
                TargetList class object
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I:
                Numpy ndarray containing inclinations of the planets of interest
            dMag:
                Numpy ndarray containing differences in magnitude between planets 
                and their host star
            WA:
                Numpy ndarray containing working angles of the planets of interest
        
        Returns:
            intTime:
                1D numpy array of integration times (astropy Quantity with 
                units of day)
        
        """

        inst = self.Imager
        syst = self.HLC
        lam = inst['lam']

        # nb of pixels for photometry aperture = 1/sharpness
        PSF = syst['PSF'](lam, WA)
        Npix = (np.sum(PSF))**2/np.sum(PSF**2)
        C_p, C_b = self.Cp_Cb(targlist, sInd, I, dMag, WA, inst, syst, Npix)

        # Nemati14+ method
        PP = targlist.PostProcessing                # post-processing module
        SNR = PP.SNchar                             # SNR threshold for characterization
        ppFact = PP.ppFact                          # post-processing contrast factor
        Q = syst['contrast'](lam, WA)
        SpStr = C_p*10.**(0.4*dMag)*Q*ppFact        # spatial structure to the speckle
        C_b += C_p*inst['ENF']**2                   # Cb must include the planet
        intTime = SNR**2*C_b / (C_p**2 - (SNR*SpStr)**2);
        
        return intTime.to(u.day)
