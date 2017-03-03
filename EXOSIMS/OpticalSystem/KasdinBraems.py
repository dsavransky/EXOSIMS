# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import astropy.units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class KasdinBraems(OpticalSystem):
    """KasdinBraems Optical System class
    
    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Kasdin & Braems 2006.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        OpticalSystem.__init__(self, **specs)

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode):
        """Finds integration times of target systems for a specific observing 
        mode (imaging or characterization), based on Kasdin and Braems 2006.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
        
        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day
        
        """
        
        # electron counts
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode)
        # for characterization, Cb must include the planet
        if mode['detectionMode'] == False:
            C_b = C_b + C_p*mode['inst']['ENF']**2
        
        # Kasdin06+ method
        inst = mode['inst']                         # scienceInstrument
        syst = mode['syst']                         # starlightSuppressionSystem
        lam = mode['lam']
        PSF = syst['PSF'](lam, WA)  # this is the only place where PSF is used!
        Pbar = PSF/np.max(PSF)
        P1 = np.sum(Pbar)
        Psi = np.sum(Pbar**2)/(np.sum(Pbar))**2
        Xi = np.sum(Pbar**3)/(np.sum(Pbar))**3
        PPro = TL.PostProcessing                    # post-processing module
        K = st.norm.ppf(1-PPro.FAP)                 # false alarm threshold
        gamma = st.norm.ppf(1-PPro.MDP)             # missed detection threshold
        deltaAlphaBar = ((inst['pitch']/inst['focal'])**2 / (lam/self.pupilDiam)**2)\
                .decompose()                        # dimensionless pixel size
        Tcore = syst['core_thruput'](lam, WA)
        Ta = Tcore*self.shapeFac*deltaAlphaBar*P1   # Airy throughput
        # calculate integration time based on Kasdin&Braems2006
        with np.errstate(divide='ignore',invalid='ignore'):
            Qbar = np.true_divide(C_p*P1,C_b)
            beta = np.true_divide(C_p,Tcore)
            intTime = np.true_divide((K - gamma*np.sqrt(1.+Qbar*Xi/Psi))**2, \
                    (beta*Qbar*Ta*Psi))
        # integration times (NAN and negative values correspond to infinity)
        intTime[np.isnan(intTime)] = np.inf*u.day
        intTime[intTime < 0] = np.inf*u.day
        
        return intTime.to('day')
