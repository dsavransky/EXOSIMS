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

        # default detectors and imagers
        self.Imager = self.scienceInstruments[0]
        self.ImagerSyst = self.starlightSuppressionSystems[0]
        self.Spectro = self.scienceInstruments[-1]
        self.SpectroSyst = self.starlightSuppressionSystems[-1]

    def Cp_Cb(self, targlist, sInds, I, dMag, WA, inst, syst, Npix):
        """ Calculates electron count rates for planet signal and background noise.

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
            inst:
                Selected scienceInstrument
            syst:
                Selected starlightSuppressionSystem
            Npix:
                Number of noise pixels

        Returns:
            C_p:
                1D numpy array of planet signal electron count rate (units of s^-1)
            C_b:
                1D numpy array of background noise electron count rate (units of s^-1)
        
        """
        
        lam = inst['lam']                           # central wavelength
        deltaLam = inst['deltaLam']                 # bandwidth
        QE = inst['QE'](lam)                        # quantum efficiency
        Q = syst['contrast'](lam, WA)               # contrast
        T = syst['throughput'](lam, WA) / inst['Ns'] \
                * self.attenuation**2               # throughput
        mV = self.starMag(targlist,sInds,lam)       # star visual magnitude
        zodi = targlist.ZodiacalLight               # zodiacalLight module
        fZ = zodi.fZ(targlist,sInds,lam)            # surface brightness of local zodi
        fEZ = zodi.fEZ(targlist,sInds,I)            # surface brightness of exo-zodi
        X = np.sqrt(2)/2                            # aperture photometry radius (in lam/D)
        Theta = (X*lam/self.pupilDiam*u.rad).to('arcsec') # angular radius (in arcseconds)
        Omega = np.pi*Theta**2                      # solid angle subtended by the aperture

        # electron count rates [ s^-1 ]
        C_F0 = self.F0(lam)*QE*T*self.pupilArea*deltaLam
        C_p = C_F0*10.**(-0.4*(mV + dMag))          # planet signal
        C_s = C_F0*10.**(-0.4*mV)*Q                 # residual suppressed starlight (coro)
        C_z = C_F0*(fZ+fEZ)*Omega                   # zodiacal light = local + exo
        C_id = Npix*inst['idark']                   # dark current
        C_cc = Npix*inst['CIC']/inst['texp']        # clock-induced-charge
        C_sr = Npix*(inst['sread']/inst['Gem'])**2/inst['texp'] # readout noise
        C_b = inst['ENF']**2*(C_s + C_z + C_id + C_cc) + C_sr   # total noise budget
        
        return C_p, C_b

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

    def calc_charTime(self, targlist, sInds, I, dMag, WA):
        """Finds characterization time for a specific target system,
        based on Nemati 2014 (SPIE).
        
        
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
            charTime (Quantity):
                1D numpy ndarray of characterization times (default units of day)
        
        """
        
        inst = self.Spectro
        syst = self.SpectroSyst
        lam = inst['lam']

        # nb of pixels for photometry aperture = 1/sharpness
        PSF = syst['PSF'](lam, WA)
        Npix = (np.sum(PSF))**2/np.sum(PSF**2)
        C_p, C_b = self.Cp_Cb(targlist, sInds, I, dMag, WA, inst, syst, Npix)

        # Nemati14+ method
        PP = targlist.PostProcessing                # post-processing module
        SNR = PP.SNchar                             # SNR threshold for characterization
        ppFact = PP.ppFact                          # post-processing contrast factor
        Q = syst['contrast'](lam, WA)
        SpStr = C_p*10.**(0.4*dMag)*Q*ppFact        # spatial structure to the speckle
        C_b += C_p*inst['ENF']**2                   # Cb must include the planet
        charTime = SNR**2*C_b / (C_p**2 - (SNR*SpStr)**2);

        return charTime.to('day')
