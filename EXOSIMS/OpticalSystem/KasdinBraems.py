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
        specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        OpticalSystem.__init__(self, **specs)
        

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None):
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
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode, TK=TK)
        
        # Kasdin06+ method
        inst = mode['inst']                         # scienceInstrument
        syst = mode['syst']                         # starlightSuppressionSystem
        lam = mode['lam']
        
        # load PSF and use it to calculate the sharpness parameters
        PSF = syst['PSF'](lam, WA)
        if np.std(PSF) > 0:
            Pbar = PSF/np.max(PSF)
            P1 = np.sum(Pbar)
            Psi = np.sum(Pbar**2)/(np.sum(Pbar))**2
            Xi = np.sum(Pbar**3)/(np.sum(Pbar))**3
        # if PSF is a flat image, e.g. default ones(3,3)
        # then use values corresponding to HLC data
        # TODO: find a better default PSF value (Airy pattern)
        else:
            P1 = 11.7044848279412
            Psi = 0.0448393
            Xi = 0.0028994
        
        PPro = TL.PostProcessing                    # post-processing module
        K = st.norm.ppf(1 - PPro.FAP)               # false alarm threshold
        gamma = st.norm.ppf(1 - PPro.MDP)           # missed detection threshold
        deltaAlphaBar = ((inst['pixelSize']/inst['focal'])**2 \
                /(lam/self.pupilDiam)**2).decompose() # dimensionless pixel size
        Tcore = syst['core_thruput'](lam, WA)
        Ta = Tcore*self.shapeFac*deltaAlphaBar*P1   # Airy throughput
        # calculate integration time based on Kasdin&Braems2006
        with np.errstate(divide='ignore', invalid='ignore'):
            Qbar = np.true_divide(C_p*P1, C_b)
            beta = np.true_divide(C_p, Tcore)
            intTime = np.true_divide((K - gamma*np.sqrt(1. + Qbar*Xi/Psi))**2,
                    beta*Qbar*Ta*Psi)
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = 0.*u.d
        # negative values are set to zero
        intTime[intTime < 0] = 0.*u.d
        
        return intTime.to('day')
