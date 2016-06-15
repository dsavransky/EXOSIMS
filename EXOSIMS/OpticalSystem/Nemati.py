# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import astropy.units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class Nemati(OpticalSystem):
    """Nemati Optical System class
    
    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Nemati 2014.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        OpticalSystem.__init__(self, **specs)

    def calc_intTime(self, TL, sInds, dMag, WA, fEZ, fZ):
        """Finds integration time for a specific target system,
        based on Nemati 2014 (SPIE).
        
        Args:
            TL (object):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest, with the length of 
                the number of planets of interest
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
        
        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day
        
        """
        
        # check type of sInds
        sInds = np.array(sInds)
        if not sInds.shape:
            sInds = np.array([sInds])
        
        # use the imager to calculate the integration time
        inst = self.Imager
        syst = self.ImagerSyst
        lam = inst['lam']
        
        # nb of pixels for photometry aperture = 1/sharpness
        PSF = syst['PSF'](lam, WA)
        Npix = (np.sum(PSF))**2/np.sum(PSF**2)
        C_p, C_b = self.Cp_Cb(TL, sInds, dMag, WA, fEZ, fZ, inst, syst, Npix)
        
        # Nemati14+ method
        PP = TL.PostProcessing                      # post-processing module
        SNR = PP.SNchar                             # SNR threshold for characterization
        ppFact = PP.ppFact                          # post-processing contrast factor
        Q = syst['contrast'](lam, WA)
        SpStr = C_p*10.**(0.4*dMag)*Q*ppFact        # spatial structure to the speckle
        C_b += C_p*inst['ENF']**2                   # Cb must include the planet
        intTime = SNR**2*C_b / (C_p**2 - (SNR*SpStr)**2);
        
        # negative values correspond to infinite integration times
        intTime[intTime < 0] = np.inf
        
        return intTime.to('day')

    def calc_charTime(self, TL, sInds, dMag, WA, fEZ, fZ):
        """Finds characterization time for a specific target system,
        based on Nemati 2014 (SPIE).
        
        Args:
            TL (object):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest, with the length of 
                the number of planets of interest
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
        
        Returns:
            charTime (astropy Quantity array):
                Characterization times in units of day
        
        """
        
        # check type of sInds
        sInds = np.array(sInds)
        if not sInds.shape:
            sInds = np.array([sInds])
        
        # use the spectro to calculate the characterization time
        inst = self.Spectro
        syst = self.SpectroSyst
        lam = inst['lam']
        
        # nb of pixels for photometry aperture = 1/sharpness
        PSF = syst['PSF'](lam, WA)
        Npix = (np.sum(PSF))**2/np.sum(PSF**2)
        C_p, C_b = self.Cp_Cb(TL, sInds, dMag, WA, fEZ, fZ, inst, syst, Npix)
        
        # Nemati14+ method
        PP = TL.PostProcessing                      # post-processing module
        SNR = PP.SNimag                             # SNR threshold for imaging/detection
        ppFact = PP.ppFact                          # post-processing contrast factor
        Q = syst['contrast'](lam, WA)
        SpStr = C_p*10.**(0.4*dMag)*Q*ppFact        # spatial structure to the speckle
        C_b += C_p*inst['ENF']**2                   # Cb must include the planet
        charTime = SNR**2*C_b / (C_p**2 - (SNR*SpStr)**2);
        
        # negative values correspond to infinite characterization times
        charTime[charTime < 0] = np.inf
        
        return charTime.to('day')
