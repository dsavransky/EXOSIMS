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
        
        # get SNR threshold
        SNR = mode['SNR']
        # calculate integration time based on Nemati 2014
        with np.errstate(divide='ignore', invalid='ignore'):
            intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = 0.*u.d
        # negative values are set to zero
        intTime[intTime < 0] = 0.*u.d
        
        return intTime.to('day')

    def calc_dMag_per_intTime(self, t_int, TL, sInds, fZ, fEZ, WA, mode):
        """Finds achievable dMag for one integration time per star in the input 
        list at one or more working angles.
        
        Achievable dMag is returned as an m x n array where m corresponds to 
        each star in sInds and n corresponds to each working angle in WA.
        
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
                            
        Returns:
            dMag (ndarray):
                Achievable dMag for given integration time and working angle
                
        """
        
        # reshape sInds, WA, t_int
        sInds = np.array(sInds, ndmin=1)
        WA = np.array(WA.value, ndmin=1)*WA.unit
        t_int = np.array(t_int.value, ndmin=1)*t_int.unit
        assert len(t_int) == len(sInds), "t_int and sInds must be same length"
        
        # get scienceInstrument and starlightSuppressionSystem
        inst = mode['inst']
        syst = mode['syst']
        
        # get mode wavelength
        lam = mode['lam']
        # get mode bandwidth
        # include any IFS spectral resolving power, which is equivalent to 
        # using f_sr from Nemati
        deltaLam = lam/inst['Rs'] if 'spec' in inst['name'].lower() else mode['deltaLam']
        
        # if the mode wavelength is different than the wavelength at which the system 
        # is defined, we need to rescale the working angles
        if lam != syst['lam']:
            WA = WA*lam/syst['lam']
        
        # get star magnitude
        mV = TL.starMag(sInds, lam)
        
        # get signal to noise ratio
        SNR = mode['SNR']
        
        # spectral flux density = F0 * A * Dlam * QE * T (attenuation due to optics)
        attenuation = inst['optics']*syst['optics']
        C_F0 = self.F0(lam)*self.pupilArea*deltaLam*inst['QE'](lam)*attenuation
        
        # get core_thruput
        core_thruput = syst['core_thruput'](lam, WA)
        
        dMag = np.zeros((len(sInds), len(WA)))
        for i in xrange(len(sInds)):
            _, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds[i], fZ, fEZ, self.dMagLim, WA, mode)
            dMag[i,:] = -2.5*np.log10((SNR*np.sqrt(C_b/t_int[i] + C_sp**2) \
                    /(C_F0*10.0**(-0.4*mV[i])*core_thruput)).decompose().value)
        
        return dMag
    
    def ddMag_dt(self, t_int, TL, sInds, fZ, fEZ, WA, mode):
        """Finds derivative of achievable dMag with respect to integration time
        
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
            
        Returns:
            ddMagdt (ndarray):
                Derivative of achievable dMag with respect to integration time
        
        """
        # reshape sInds, WA, t_int
        sInds = np.array(sInds, ndmin=1)
        WA = np.array(WA.value, ndmin=1)*WA.unit
        t_int = np.array(t_int.value, ndmin=1)*t_int.unit
        assert len(t_int) == len(sInds), "t_int and sInds must be same length"
        
        ddMagdt = np.zeros((len(sInds), len(WA)))
        for i in xrange(len(sInds)):
            _, Cb, Csp = self.Cp_Cb_Csp(TL, sInds[i], fZ, fEZ, self.dMagLim, WA, mode)
            ddMagdt[i,:] = 2.5/(2.0*np.log(10.0))*(Cb/(Cb*t_int[i] + (Csp*t_int[i])**2)).to('1/s').value
            
        return ddMagdt/u.s
