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

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode):
        """Finds integration times of target systems for a specific observing 
        mode (imaging or characterization), based on Nemati 2014 (SPIE).
        
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
        
        # get SNR threshold
        SNR = mode['SNR']
        # calculate integration time based on Nemati 2014
        with np.errstate(divide='ignore',invalid='ignore'):
            intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = 0.*u.d
        # negative values are set to zero
        intTime[intTime < 0] = 0.*u.d
        
        return intTime.to('day')
    
    def calc_contrast_per_intTime(self, t_int, TL, sInds, fZ, fEZ, WA, mode, dMag=25.0):
        """Finds instrument achievable contrast for given integration time(s) 
        and working angle(s).
        
        Instrument contrast is returned as an m x n array where m corresponds 
        to each star in sInds and n corresponds to each working angle in WA.
        
        Args:
            t_int (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
                
        Returns:
            C_inst (ndarray):
                Instrument contrast for given integration time and working angle
                
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        # reshape WA
        WA = np.array(WA.value,ndmin=1)*WA.unit
        
        # get scienceInstrument and starlightSuppressionSystem
        inst = mode['inst']
        syst = mode['syst']
        
        # get mode wavelength
        lam = mode['lam']
        # get mode bandwidth (including any IFS spectral resolving power)
        deltaLam = lam/inst['Rs'] if 'spec' in inst['name'].lower() else mode['deltaLam']
        
        # if the mode wavelength is different than the wavelength at which the system 
        # is defined, we need to rescale the working angles
        if lam != syst['lam']:
            WA = WA*lam/syst['lam']
        
        # get star magnitude
        sInds = np.array(sInds,ndmin=1)
        mV = TL.starMag(sInds,lam)
        
        # get signal to noise ratio
        SNR = mode['SNR']
        
        # spectral flux density = F0 * A * Dlam * QE * T (non-coro attenuation)
        C_F0 = self.F0(lam)*self.pupilArea*deltaLam*inst['QE'](lam)*self.attenuation
        
        # get core_thruput
        core_thruput = syst['core_thruput'](lam,WA)
        
        C_inst = np.zeros((len(sInds),len(WA)))
        for i in xrange(len(sInds)):
            C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds[i], fZ, fEZ, dMag, WA, mode)
            C_inst[i,:] = SNR*np.sqrt(C_b/t_int[i] + C_sp**2)/(C_F0*10.0**(-0.4*mV[i])*core_thruput)
        
        return C_inst
