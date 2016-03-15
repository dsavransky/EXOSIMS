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
        fzodi = zodi.fzodi(starInd,I,targlist);     # local + exozodi level

        X = np.sqrt(2)/2                            # aperture photometry radius (in lam/D)
        Theta = X*self.lam.to(u.m)/self.pupilDiam*u.rad.to(u.arcsec)\
                                                    # aperture photometry angular radius
        Omega = np.pi*Theta**2                      # solid angle subtended by the aperture

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

        # nb of pixels for photometry aperture = 1/sharpness
        PSF = self.PSF(self.lam, WA);
        Pbar = PSF/np.max(PSF);
        Npix = (np.sum(Pbar))**2/np.sum(Pbar**2);
        # throughput, contrast, quantum efficiency
        T = self.throughput(self.lam, WA) * self.attenuation**2;
        Q = self.contrast(self.lam, WA) * ppFact;
        QE = QEfunc(self.lam) /u.photon;
        # average irradiance in detection band [photons /s /m2 /nm]
        F0 = 3631*1.51e7/self.lam *u.photon/u.s/u.m**2;     # zero-magnitude star = 3631 Jy
        # broadband electron count rate of F0
        C_F0 = F0*QE*T*self.pupilArea*self.deltaLam;
        # electron rates [ s^-1 ]
        r_pl = C_F0*10.**(-0.4*(Vmag + dMagPlan));
        r_CG = C_F0*10.**(-0.4*Vmag)*Q;
        r_zl = C_F0*fzodi*Omega;                            # zodiacal light = local + exo
        r_dark = darkCurrent*Npix;
        r_cic = CIC*Npix/texp;
        r_read = (readNoise/G_EM)**2*Npix/texp;
        
        # Nemati14+ method
        r_noise = ENF**2*(r_pl + r_CG + r_zl + r_dark + r_cic) + r_read;
        SNR = PP.SNR;                               # SNR threshold for imaging/detection

        intTime = (SNR**2*r_noise)/(r_pl**2 - SNR**2*r_CG**2);

        return intTime.to(u.day)

