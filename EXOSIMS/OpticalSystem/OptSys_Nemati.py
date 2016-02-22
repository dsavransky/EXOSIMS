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

    def calc_intTime(self, targlist, starInd, dMagPlan, WA, I):
        """Finds integration time for a specific target system 
        
        This method is called from a method in the SurveySimulation class
        object.
        
        Args:
            targlist:
                TargetList class object
            starInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            dMagPlan:
                Numpy ndarray containing differences in magnitude between planets 
                and their host star
            WA:
                Numpy ndarray containing working angles of the planets of interest
            I:
                Numpy ndarray containing inclinations of the planets of interest
        
        Returns:
            intTime:
                1D numpy array of integration times (astropy Quantity with 
                units of day)
        
        """

        PP = targlist.PostProcessing                # post-processing module
        ppFact = PP.ppFact;                         # post-processing contrast factor
        Vmag = targlist.Vmag[starInd];              # star visual magnitude
        zodi = targlist.ZodiacalLight               # zodiacalLight module
        EZlevel = zodi.fzodi(starInd,I,targlist);   # exozodi level
        EZmag = zodi.exozodiMag;                    # 1 zodi brightness in mag per asec2
        Omega = (0.7*self.lam.to(u.m))**2*self.shapeFac/self.pupilArea\
                *(u.rad.to(u.arcsec))**2;           # pixel size in square arcseconds

        # values taken from the imaging camera
        for syst in self.scienceInstruments:
            if 'imag' in syst['type']:
                QEfunc = syst['QE'];
                darkCurrent = syst['darkCurrent'];
                CIC = syst['CIC'];
                readNoise = syst['readNoise'];
                texp = syst['texp'];
                ENF = syst['ENF'];
                G_EM = syst['G_EM'];

        # values derived from the normalized PSF
        PSF = self.PSF(self.lam, WA);
        Pbar = PSF/np.max(PSF);
        P1 = np.sum(Pbar);
        Psi = np.sum(Pbar**2)/(np.sum(Pbar))**2;
        Xi = np.sum(Pbar**3)/(np.sum(Pbar))**3;
        # nb of pixels for photometry aperture = 1/sharpness
        Npix = 1./Psi;

        # throughput, contrast
        T = self.attenuation**2*self.throughput(self.lam, WA);
        Q = ppFact*self.contrast(self.lam, WA);

        # average irradiance in detection band [photons /s /m2 /nm]
        F0 = 3631*1.51e7/self.lam *u.photon/u.s/u.m**2; #zero-magnitude star = 3631 Jy
        I_psf = F0*10.**(-Vmag/2.5);
        I_pl = F0*10.**(-(Vmag + dMagPlan)/2.5); 
        I_CG = I_psf * Q;
        I_zodi = F0*10.**(-EZmag/2.5)*EZlevel*Omega;

        # electron rates [ s^-1 ]
        QE = QEfunc(self.lam) /u.photon;
        r_pl = I_pl*QE*T*self.pupilArea*self.deltaLam;
        r_CG = I_CG*QE*T*self.pupilArea*self.deltaLam;
        r_zodi = I_zodi*QE*T*self.pupilArea*self.deltaLam;
        r_dark = darkCurrent*Npix;
        r_cic = CIC*Npix/texp;
        r_read = (readNoise/G_EM)**2*Npix/texp;
        
        # Nemati14+ method
        r_noise = ENF**2*(r_pl + r_CG + r_zodi + r_dark + r_cic) + r_read;
        SNR = PP.SNR;                               # SNR threshold for imaging/detection

        intTime = (SNR**2*r_noise)/(r_pl**2 - SNR**2*r_CG**2);

        return intTime.to(u.day)

