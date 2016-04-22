# -*- coding: utf-8 -*-
from EXOSIMS.OpticalSystem.Nemati import Nemati
import astropy.units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class KasdinBraems(Nemati):
    """KasdinBraems Optical System class
    
    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Kasdin & Braems 2006.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        Nemati.__init__(self, **specs)

    def calc_intTime(self, targlist, sInds, I, dMag, WA):
        """Finds integration time for a specific target system,
        based on Kasdin and Braems 2006.
        
        Args:
            targlist:
                TargetList class object
            sInds (integer ndarray):
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
        syst = self.ImagerSyst
        lam = inst['lam']

        # nb of pixels for photometry aperture = 1/sharpness
        PSF = syst['PSF'](lam, WA)
        Npix = (np.sum(PSF))**2/np.sum(PSF**2)
        C_p, C_b = self.Cp_Cb(targlist, sInds, I, dMag, WA, inst, syst, Npix)

        # Kasdin06+ method
        Pbar = PSF/np.max(PSF)
        P1 = np.sum(Pbar)
        Psi = np.sum(Pbar**2)/(np.sum(Pbar))**2
        Xi = np.sum(Pbar**3)/(np.sum(Pbar))**3
        Qbar = C_p/C_b*P1
        PP = targlist.PostProcessing                # post-processing module
        K = st.norm.ppf(1-PP.FAP)                   # false alarm threshold
        gamma = st.norm.ppf(1-PP.MDP)               # missed detection threshold
        deltaAlphaBar = ((inst['pitch']/inst['focal'])**2 / (lam/self.pupilDiam)**2)\
                .decompose()                        # dimensionless pixel size
        T = syst['throughput'](lam, WA) * self.attenuation**2
        Ta = T*self.shapeFac*deltaAlphaBar*P1       # Airy throughput 
        beta = C_p/T
        intTime = 1./beta*(K - gamma*np.sqrt(1.+Qbar*Xi/Psi))**2/(Qbar*Ta*Psi)

        return intTime.to('day')
